"""AI 客户端：统一封装 chat/completions 兼容接口（OpenAI / DeepSeek / 自定义）。

设计要点：
- 配置来自数据库 KV（加密存储），可用 Web 界面修改
- 指数退避重试（429/5xx/网络错误/空响应）
- 滑动窗口速率限制
- 密钥绝不落日志
- 深度思考支持：DeepSeek 等推理模型返回 reasoning_content，content 为空时自动扩容重试
- JSON 模式：chat_json 用 response_format 强制输出合法 JSON
"""
import asyncio
import json
import logging
import random
import time
from typing import Any, Optional

import httpx

from core.database import Database

log = logging.getLogger(__name__)

# 常见 OpenAI 兼容接口的默认路径
_PROVIDER_PATHS = {
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com/v1",
}

# 推理模型输出较大，JSON 任务需要更大空间（含 reasoning_content）
_JSON_MAX_TOKENS = 8192
_JSON_MIN_TOKENS = 4096
# 扩容重试的 max_tokens 硬上限：DeepSeek（deepseek-chat/reasoner）上限为 8192，
# 直接翻倍到 16384 会被 400 拒绝——扩容必须封顶（留 192 余量防边界拒绝）
_MAX_TOKEN_CAP = 8000

# ============ 按功能设定 AI 参数（不同任务对温度/token 需求不同） ============
# 温度：越低越确定（分析/优化/复盘要精确、避免编造），越高越发散（设计/挖掘要探索）。
# token：推理模型需要更大空间输出 reasoning + JSON。
AI_FEATURE_PARAMS: dict[str, dict] = {
    "market_analysis": {"temperature": 0.2, "max_tokens": 4096},   # 解读要客观、证据导向
    "param_optimize": {"temperature": 0.2, "max_tokens": 4096},    # 参数微调要保守精确
    "strategy_design": {"temperature": 0.7, "max_tokens": 8192},   # 设计要探索新思路
    "strategy_iterate": {"temperature": 0.6, "max_tokens": 8192},  # 反思迭代要有批判但不失控
    "trade_review": {"temperature": 0.2, "max_tokens": 4096},      # 复盘要客观归因
    "factor_mine": {"temperature": 0.8, "max_tokens": 8192},       # 因子挖掘要发散创新
}


class AINotConfigured(Exception):
    """AI 尚未配置（缺少 API Key 等）。"""


class AICallError(Exception):
    """AI 调用失败（网络 / 状态码 / 解析错误）。"""


class _EmptyContentError(Exception):
    """AI 只输出了 reasoning 没有 content（推理模型 max_tokens 不足）。"""


class _SlidingWindowLimiter:
    """滑动窗口限速器：窗口内最多 n 次、两次间隔至少 min_interval 秒。"""

    def __init__(self, rpm: int = 30, min_interval: float = 2.0) -> None:
        self.max_per_window = max(1, int(rpm))
        self.min_interval = min_interval
        self._times: list[float] = []

    async def acquire(self) -> None:
        now = time.time()
        window = 60.0
        self._times = [t for t in self._times if now - t < window]
        if self.min_interval > 0 and self._times:
            wait = self._times[-1] + self.min_interval - now
            if wait > 0:
                await asyncio.sleep(wait)
        while len(self._times) >= self.max_per_window:
            await asyncio.sleep(0.2)
            now = time.time()
            self._times = [t for t in self._times if now - t < window]
        self._times.append(time.time())


class AIClient:
    """OpenAI chat/completions 兼容客户端。

    配置优先级：数据库 KV（Web 界面保存）> .env 默认值。
    """

    def __init__(self, db: Database, defaults: Optional[dict] = None) -> None:
        self._db = db
        self._default = defaults or {}
        self._cache: dict[str, str] = {}
        self._cache_at = 0.0
        self._lock = asyncio.Lock()
        self._limiter = _SlidingWindowLimiter(
            rpm=int(self._default.get("ai_rate_limit_rpm", 30)),
            min_interval=float(self._default.get("ai_rate_limit_min_interval", 2.0)),
        )
        # 长生命周期 HTTP 连接（按事件循环缓存）：连接复用避免每次请求 TCP+TLS 握手。
        # 后台线程（AI 任务）与主循环共用同一 AIClient，httpx client 绑定创建时的
        # 事件循环，故按 loop 分开缓存；close() 在进程退出时统一关闭。
        self._http_clients: dict[int, httpx.AsyncClient] = {}

    def invalidate_cache(self) -> None:
        """配置变更后清空缓存。"""
        self._cache.clear()
        self._cache_at = 0.0

    def _client_for_loop(self) -> httpx.AsyncClient:
        """获取当前事件循环的 HTTP 客户端（惰性创建，连接复用）。"""
        loop = asyncio.get_running_loop()
        key = id(loop)
        c = self._http_clients.get(key)
        if c is None:
            c = httpx.AsyncClient(timeout=httpx.Timeout(90.0))
            self._http_clients[key] = c
        return c

    async def close(self) -> None:
        """关闭全部 HTTP 连接（进程退出时调用）。"""
        for c in self._http_clients.values():
            try:
                await c.aclose()
            except Exception:  # noqa: BLE001
                pass
        self._http_clients.clear()

    async def _load_config(self) -> dict:
        """读取 AI 配置：优先数据库 KV，其次 .env 默认。"""
        async with self._lock:
            now = time.time()
            if self._cache and now - self._cache_at < 30:
                return dict(self._cache)
            # 并行读取全部配置项（曾 5 次顺序 kv_get）
            provider, base_url, api_key, model, temperature, max_tokens = await asyncio.gather(
                self._db.kv_get("ai_provider"), self._db.kv_get("ai_base_url"),
                self._db.kv_get_secret("ai_api_key"), self._db.kv_get("ai_model"),
                self._db.kv_get("ai_temperature"), self._db.kv_get("ai_max_tokens"),
            )
            provider = provider or self._default.get("ai_provider", "openai")
            base_url = base_url or self._default.get("ai_base_url", "")
            api_key = api_key or self._default.get("ai_api_key", "")
            model = model or self._default.get("ai_model", "gpt-4o-mini")
            temperature = float(temperature or self._default.get("ai_temperature", 0.7))
            max_tokens = int(max_tokens or self._default.get("ai_max_tokens", 2048))
            self._cache = {
                "provider": provider, "base_url": base_url, "api_key": api_key,
                "model": model, "temperature": temperature, "max_tokens": max_tokens,
            }
            self._cache_at = now
            return dict(self._cache)

    async def _resolve_url(self, cfg: dict) -> str:
        base = (cfg["base_url"] or _PROVIDER_PATHS.get(cfg["provider"], "")).rstrip("/")
        if not base:
            raise AINotConfigured("未配置 AI Base URL")
        return base + "/chat/completions"

    async def _post(self, messages: list[dict], json_mode: bool = False,
                    extra_max_tokens: Optional[int] = None,
                    feature: Optional[str] = None,
                    temperature: Optional[float] = None) -> tuple[str, str]:
        """发送请求，返回 (content, reasoning_content)。

        - json_mode=True 时强制 response_format=json_object，并自动扩容 max_tokens
        - feature：指定 AI 功能名，用该功能预设的 temperature / max_tokens
          （不同功能对确定性与 token 需求不同，由代码层统一管理，不再让用户手调）
        - temperature：显式覆盖（优先级最高）
        - 推理模型 content 为空时（reasoning 占满预算），自动扩容重试一次
        """
        cfg = await self._load_config()
        if not cfg["api_key"]:
            raise AINotConfigured("未配置 AI API Key（请在「AI 与 API」页设置）")
        url = await self._resolve_url(cfg)

        # 按功能预设参数（若 feature 存在且预设了）；temperature 显式传参最优先
        fp = AI_FEATURE_PARAMS.get(feature or "", {})
        temp = temperature if temperature is not None else fp.get("temperature", cfg["temperature"])
        max_tokens = extra_max_tokens or fp.get("max_tokens", cfg["max_tokens"])
        # JSON 任务自动扩容：保证 reasoning + JSON 都有空间
        if json_mode and max_tokens < _JSON_MIN_TOKENS:
            max_tokens = _JSON_MAX_TOKENS

        headers = {"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"}
        base_body: dict[str, Any] = {
            "model": cfg["model"],
            "messages": messages,
            "temperature": temp,
            "max_tokens": max_tokens,
        }
        if json_mode:
            base_body["response_format"] = {"type": "json_object"}
            if not _contains_json_instruction(messages):
                messages = [dict(m) for m in messages]
                messages[-1]["content"] = messages[-1]["content"] + "\n\n（必须输出严格 JSON，不要输出任何多余文字）"

        retries = int(self._default.get("ai_max_retries", 3))
        last_err: Optional[Exception] = None
        # 长生命周期连接复用（避免每次请求 TCP+TLS 握手）
        client = self._client_for_loop()
        for attempt in range(retries + 1):
            await self._limiter.acquire()
            body = dict(base_body)
            if attempt == 1 and isinstance(last_err, _EmptyContentError):
                # 扩容重试：推理模型 reasoning 占满预算导致 content 为空。
                # 封顶 _MAX_TOKEN_CAP：DeepSeek max_tokens 上限 8192，翻倍会直接 400
                body["max_tokens"] = min(max(max_tokens * 2, _JSON_MAX_TOKENS * 2), _MAX_TOKEN_CAP)
                log.warning("[ai] content 为空，扩容 max_tokens=%d 重试", body["max_tokens"])
            try:
                resp = await client.post(url, headers=headers, json=body)
                if resp.status_code == 200:
                    data = resp.json()
                    message = (data.get("choices") or [{}])[0].get("message") or {}
                    content = (message.get("content") or "").strip()
                    reasoning = (message.get("reasoning_content") or "").strip()
                    if not content:
                        # 推理模型常见：只输出了思考没有正文
                        raise _EmptyContentError("AI 只输出了推理没有内容")
                    return content, reasoning
                detail = resp.text[:400] or resp.reason_phrase
                if resp.status_code in (429, 500, 502, 503, 504):
                    last_err = RuntimeError(f"HTTP {resp.status_code}: {detail}")
                    # 指数退避 + 抖动：多任务并发触发临时错误时避免同步波峰
                    wait = 2 ** attempt + random.uniform(0, 0.5)
                    if resp.status_code == 429:
                        # 429 优先遵循服务端 Retry-After
                        ra = resp.headers.get("Retry-After")
                        if ra and ra.strip().isdigit():
                            wait = min(float(ra.strip()), 30.0)
                    log.warning("[ai] 临时错误(第%d/%d次) %s，%.1fs 后重试", attempt + 1, retries, last_err, wait)
                    await asyncio.sleep(wait)
                    continue
                raise AICallError(f"AI 接口返回 {resp.status_code}，具体原因: {detail}")
            except _EmptyContentError as e:
                last_err = e
                if attempt < retries:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                raise AICallError(f"AI 调用 {retries + 1} 次后仍无正文输出（可能是推理模型 max_tokens 限制）")
            except httpx.TransportError as e:
                last_err = e
                wait = 2 ** attempt + random.uniform(0, 0.5)
                log.warning("[ai] 网络错误(第%d/%d次) %s，%.1fs 后重试", attempt + 1, retries, e, wait)
                await asyncio.sleep(wait)
        raise AICallError(f"AI 调用重试 {retries} 次后仍失败: {last_err}")

    async def chat(self, messages: list[dict], feature: Optional[str] = None,
                   temperature: Optional[float] = None) -> str:
        """普通对话，返回文本。"""
        content, _ = await self._post(messages, feature=feature, temperature=temperature)
        return content

    async def chat_json(self, messages: list[dict], feature: Optional[str] = None,
                        temperature: Optional[float] = None) -> dict:
        """要求 AI 输出 JSON，稳健解析 + 强制 JSON 模式 + 截断补全重试。

        feature: 功能名，决定 temperature/max_tokens 预设。
        """
        content, _ = await self._post(messages, json_mode=True, feature=feature, temperature=temperature)
        data = self._parse_json(content)
        if data is not None:
            return data
        # JSON 不完整（可能是 reasoning 占用预算导致 content 截断）：追加"补全"重试一次。
        # 只回灌截断尾部（最长 1000 字符）而非全文——截断内容本身就是占满预算的
        # reasoning 噪声，全文回灌成本翻倍且加重 429 场景
        log.warning("[ai] JSON 解析失败，尝试补全重试: %.120s...", content)
        repair = [dict(m) for m in messages]
        repair.append({"role": "assistant", "content": content[-1000:]})
        repair.append({"role": "user", "content": (
            "你上面的输出被截断了，不是完整 JSON。请基于你已输出的内容，"
            "补全为一个完整合法的 JSON 对象，只输出 JSON，不要任何解释。"
        )})
        try:
            content2, _ = await self._post(repair, json_mode=True, feature=feature, temperature=temperature)
        except AICallError:
            raise AICallError(f"AI 返回无法解析为 JSON: {content[:200]}")
        data = self._parse_json(content2)
        if data is None:
            raise AICallError(f"AI 返回无法解析为 JSON（补全后仍失败）: {content2[:200]}")
        return data

    async def chat_with_reasoning(self, messages: list[dict], feature: Optional[str] = None,
                                  temperature: Optional[float] = None) -> dict:
        """返回 {content, reasoning}，供需要展示/利用深度思考的功能使用。"""
        content, reasoning = await self._post(messages, feature=feature, temperature=temperature)
        return {"content": content, "reasoning": reasoning}

    async def chat_json_validated(self, messages: list[dict], feature: str,
                                  ctx: Optional[dict] = None,
                                  max_repairs: int = 2) -> dict:
        """校验式 JSON 调用：AI 输出必须通过量化规则校验，否则喂回错误要求修正。

        这是"防幻觉 + 每代正向提升"的核心机制：
        - 第 1 次调用 → 校验 → 失败则把具体错误喂回 AI 要求修正
        - 有界重试 max_repairs 次；仍失败抛 AICallError（宁可失败也不接受脏输出）
        ctx: 校验上下文（param_schema / summary / 市场快照等）
        """
        from .validator import ValidationError, feedback_message, validate_ai_output
        attempt = 0
        msgs = [dict(m) for m in messages]
        while attempt <= max_repairs:
            data = await self.chat_json(msgs, feature=feature)
            try:
                validate_ai_output(feature, data, ctx or {})
                return data  # 校验通过
            except ValidationError:
                if attempt >= max_repairs:
                    raise AICallError(f"[{feature}] AI 输出 {max_repairs + 1} 次未通过量化校验，已拒绝")
                fb = feedback_message(feature, data, ctx or {})
                log.warning("[ai] %s 校验失败(第%d次)，反馈修正: %.100s",
                            feature, attempt + 1, fb.replace("\n", " "))
                msgs = [dict(m) for m in messages]
                msgs.append({"role": "assistant", "content": data})
                msgs.append({"role": "user", "content": fb + "\n请严格按修正要求重新输出完整 JSON。"})
                attempt += 1
        raise AICallError(f"[{feature}] 校验重试超限")

    @staticmethod
    def _parse_json(text: str) -> Optional[dict]:
        t = text.strip()
        # 提取 ```json ... ``` 代码块
        if "```" in t:
            for block in t.split("```"):
                block = block.strip()
                if block.startswith("json"):
                    block = block[4:].strip()
                try:
                    return json.loads(block)
                except json.JSONDecodeError:
                    continue
        # 直接解析
        try:
            return json.loads(t)
        except json.JSONDecodeError:
            pass
        # 尝试从第一个 { 到最后一个 }
        start, end = t.find("{"), t.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(t[start:end + 1])
            except json.JSONDecodeError:
                return None
        return None

    async def test_connection(self, override: Optional[dict] = None) -> str:
        """测试连接：发送一句简单问候，返回模型回复文本。

        override: 可选临时配置（{provider, base_url, model, api_key}）——
        用于"表单值未保存先测试"场景，不落库、不改缓存。
        """
        if override and (override.get("api_key") or override.get("base_url") or override.get("model")):
            # 临时覆盖：直接用传入值测试（跳过 _load_config 的 KV 读取）
            temp = dict(self._default)
            temp.update({k: v for k, v in override.items() if v})
            cfg = {
                "provider": temp.get("provider", "openai"),
                "base_url": temp.get("base_url", "") or temp.get("ai_base_url", ""),
                "api_key": temp.get("api_key", "") or temp.get("ai_api_key", ""),
                "model": temp.get("model", "gpt-4o-mini"),
                "temperature": float(temp.get("temperature", 0.7)),
                "max_tokens": int(temp.get("max_tokens", 2048)),
            }
            if not cfg["api_key"]:
                raise AINotConfigured("未提供 API Key（请填写后测试，或先保存配置）")
            url = await self._resolve_url(cfg)
            headers = {"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"}
            body = {"model": cfg["model"], "messages": [
                {"role": "system", "content": "你是一个简短的连接测试助手。"},
                {"role": "user", "content": "请只回复两个字：正常"},
            ], "temperature": cfg["temperature"], "max_tokens": cfg["max_tokens"]}
            client = self._client_for_loop()
            try:
                resp = await client.post(url, headers=headers, json=body)
            except httpx.TransportError as e:
                raise AICallError(f"网络错误: {e}")
            if resp.status_code != 200:
                raise AICallError(f"AI 接口返回 {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
            content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            return content.strip() or "（空回复，但连接正常）"
        reply = await self.chat([
            {"role": "system", "content": "你是一个简短的连接测试助手。"},
            {"role": "user", "content": "请只回复两个字：正常"},
        ])
        return reply.strip()


def _contains_json_instruction(messages: list[dict]) -> bool:
    """检查提示词是否已要求 JSON 输出（避免重复注入指令）。"""
    for m in messages:
        if "JSON" in (m.get("content") or "") or "json" in (m.get("content") or ""):
            return True
    return False
