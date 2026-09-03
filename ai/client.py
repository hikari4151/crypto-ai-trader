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
import weakref
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
# 扩容重试的 max_tokens 硬上限默认值（仅当模型不在 _MODEL_MAX_TOKEN_CAPS 中时使用）
# 曾写死 8000：DeepSeek 上限 8192 留 192 余量，但 GPT-4o 可到 32000，按模型动态判定更合理
_DEFAULT_MAX_TOKEN_CAP = 8000
# 按模型名前缀映射的 max_tokens 上限（用于扩容重试时动态封顶）
_MODEL_MAX_TOKEN_CAPS: dict[str, int] = {
    "gpt-4o": 32000,
    "gpt-4": 16000,
    "gpt-3.5": 16000,
    "deepseek-chat": 8192,
    "deepseek-reasoner": 8192,
    "deepseek-r1": 8192,
    "o1": 32000,
    "o1-mini": 32000,
    "o3": 32000,
    "o3-mini": 32000,
}


def _max_token_cap_for_model(model: str) -> int:
    """根据模型名选择扩容上限；未知模型回退默认 8000。"""
    for prefix, cap in _MODEL_MAX_TOKEN_CAPS.items():
        if model.startswith(prefix):
            return cap
    return _DEFAULT_MAX_TOKEN_CAP


def _supports_json_schema(model: str) -> bool:
    """该模型是否支持 response_format=json_schema（P1 结构化输出）。

    DeepSeek 等明确不支持 json_schema 的模型直接禁用；OpenAI/o 系列与未知
    模型默认放行（即使报 400 也会由调用方自动回退 json_object，成本可忽略）。
    """
    if model.startswith("deepseek"):
        return False
    return True

# ============ 按功能设定 AI 参数（不同任务对温度/token 需求不同） ============
# 温度：越低越确定（分析/优化/复盘要精确、避免编造），越高越发散（设计/挖掘要探索）。
# token：推理模型需要更大空间输出 reasoning + JSON。
# 2026-08 优化：strategy_design 现要求输出完整 Pine 代码（token 需求大增），
# strategy_iterate 同；trade_review/factor_mine 保持。
AI_FEATURE_PARAMS: dict[str, dict] = {
    "market_analysis": {"temperature": 0.2, "max_tokens": 4096},   # 解读要客观、证据导向
    "param_optimize": {"temperature": 0.2, "max_tokens": 4096},    # 参数微调要保守精确
    "strategy_design": {"temperature": 0.7, "max_tokens": 12000},  # 设计要探索新思路 + 完整 Pine 代码
    "strategy_iterate": {"temperature": 0.6, "max_tokens": 12000}, # 反思迭代要有批判但不失控 + Pine
    "trade_review": {"temperature": 0.2, "max_tokens": 4096},      # 复盘要客观归因
    "factor_mine": {"temperature": 0.8, "max_tokens": 8192},       # 因子挖掘要发散创新
}

# 推理模型前缀（DeepSeek reasoner 等）忽略 temperature 参数，保留字段但标记。
# P4-A4：改前缀匹配——精确集合会让变体名（deepseek-r1-0528、deepseek-reasoner-xxx 等）
# 漏判，推理 API 收到 temperature 直接 400 崩溃。与 _max_token_cap_for_model 同风格。
REASONING_MODEL_PREFIXES = ("deepseek-reasoner", "deepseek-r1", "o1", "o3", "o4-mini", "gpt-5")


def _is_reasoning_model(model: str) -> bool:
    """模型是否属于推理模型（忽略 temperature 参数）。前缀匹配，兼容变体名。"""
    return any(model.startswith(p) for p in REASONING_MODEL_PREFIXES)

# P4：校验失败重试时注入的"合规示例输出"（few-shot）。
# 仅作格式/字段参考，数值必须来自实际输入（反幻觉规则仍生效）；用于复杂
# schema 时显著提高一次修正通过率，降低重试次数与 token/延迟浪费。
_FEWSHOT_EXAMPLES: dict[str, str] = {
    "market_analysis": (
        '{"regime":"震荡","bias":"neutral","confidence":0.35,'
        '"summary":"价格在区间内震荡，量能萎缩，暂无方向性机会。",'
        '"warnings":[],"signals":["上下影线交替，无明显突破"],'
        '"trade_plan":null,"evidence":[{"name":"rsi","value":52.1}]}'
    ),
    "param_optimize": (
        '{"params":{"fast_period":12,"slow_period":30},"reason":"近期趋势延续性增强，适当缩短快线以更快捕捉拐点",'
        '"focus":"价格行为与关键位"}'
    ),
    "strategy_design": (
        '{"name":"ai_breakout","title":"突破追涨策略","description":"回踩关键位后突破入场",'
        '"logic":"price>resistance 且 volume>均量1.5倍时买入，跌破支撑止损",'
        '"params":{"size_pct":0.2,"stop_loss_pct":0.02,"take_profit_pct":0.05},'
        '"risk_tips":["震荡市假突破多，需配合量能确认"],"pine_code":""}'
    ),
    "strategy_iterate": (
        '{"name":"dual_ma_v1","title":"双均线改进版","description":"在双均线上加量能过滤",'
        '"logic":"金叉且量比>1.2才入场，死叉或量比<0.8离场",'
        '"params":{"fast_period":10,"slow_period":30,"size_pct":0.3},'
        '"risk_tips":["横盘期频繁交叉",""],"critique":"原策略在震荡市反复止损",'
        '"improvements":["加入量能过滤","提高死叉离场阈值"],"summary":"减少震荡市噪声交易"}'
    ),
    "trade_review": (
        '{"score":65,"strengths":["盈利单止损执行到位"],"problems":["逆势加仓导致回撤"],'
        '"action_items":["只在趋势确认后入场","单日止损上限2%"],"summary":"整体合格，主要问题在仓位管理"}'
    ),
    "factor_mine": (
        '{"factors":[{"name":"momentum_3d","title":"三日动量","category":"momentum",'
        '"expression":"close/close.shift(3)-1","logic":"捕捉短中期动量延续"}]}'
    ),
}


class AINotConfigured(Exception):
    """AI 尚未配置（缺少 API Key 等）。"""


class AICallError(Exception):
    """AI 调用失败（网络 / 状态码 / 解析错误）。"""


class _EmptyContentError(Exception):
    """AI 只输出了 reasoning 没有 content（推理模型 max_tokens 不足）。"""


class _SlidingWindowLimiter:
    """滑动窗口限速器：窗口内最多 n 次、两次间隔至少 min_interval 秒。

    优化：使用 asyncio.Event 替代忙等待轮询，释放事件循环 CPU。
    """

    def __init__(self, rpm: int = 30, min_interval: float = 2.0) -> None:
        self.max_per_window = max(1, int(rpm))
        self.min_interval = min_interval
        self._times: list[float] = []
        self._lock = asyncio.Lock()
        self._wake = asyncio.Event()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.time()
            window = 60.0
            self._times = [t for t in self._times if now - t < window]
            if self.min_interval > 0 and self._times:
                wait = self._times[-1] + self.min_interval - now
                if wait > 0:
                    await asyncio.sleep(wait)
                    now = time.time()
                    self._times = [t for t in self._times if now - t < window]
            while len(self._times) >= self.max_per_window:
                # 等待 Event 被 set（窗口中最旧条目过期后由 set() 唤醒），
                # 带超时兜底避免信号丢失
                self._wake.clear()
                self._lock.release()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=1.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
                finally:
                    await self._lock.acquire()
                now = time.time()
                self._times = [t for t in self._times if now - t < window]
            self._times.append(time.time())
            # 通知因窗口满而等待的其它协程：最旧条目可能已过期
            self._wake.set()


class AIClient:
    """OpenAI chat/completions 兼容客户端。

    配置优先级：数据库 KV（Web 界面保存）> .env 默认值。
    """

    def __init__(self, db: Database, defaults: Optional[dict] = None) -> None:
        self._db = db
        self._default = defaults or {}
        self._cache: dict[str, str] = {}
        self._cache_at = 0.0
        # 注：不再用 asyncio.Lock 保护配置缓存——后台 worker 线程（独立事件循环）
        # 与主循环共用本实例，asyncio.Lock 跨循环 acquire 会抛异常（见 _load_config）
        # P4-D4：限速器同 _http_clients 按事件循环缓存。_SlidingWindowLimiter 内部用
        # asyncio.Lock/Event，绑定创建时的循环；主循环与后台 AI worker（独立事件循环）
        # 共用本 AIClient 时，跨循环 acquire 会抛 "bound to a different event loop"，
        # AI 调用间歇性崩溃。按 loop 各建一份限速器即可消除（限速统计按 loop 隔离，
        # 对整体限速影响可忽略）。
        self._limiters: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, _SlidingWindowLimiter] = weakref.WeakKeyDictionary()
        self._limiter_rpm = int(self._default.get("ai_rate_limit_rpm", 30))
        self._limiter_min_interval = float(self._default.get("ai_rate_limit_min_interval", 2.0))
        # 长生命周期 HTTP 连接（按事件循环缓存）：连接复用避免每次请求 TCP+TLS 握手。
        # 后台线程（AI 任务）与主循环共用同一 AIClient，httpx client 绑定创建时的
        # 事件循环，故按 loop 分开缓存；close() 在进程退出时统一关闭。
        # P1-4 修复：键用 loop 对象本身且为弱引用键字典（曾用 id(loop) 永不淘汰，
        # 旧 loop 被 GC 后新 loop 可复用同一 id → 取到绑定已关闭循环的 client，
        # 且每个后台任务泄漏一个 client）。loop 死亡即被 WeakKeyDictionary 自动
        # 淘汰；worker 退出时另有 close_loop(loop) 显式关闭并移除（见下）。
        # 已实测：本环境 httpx.AsyncClient 不持有 loop 强引用，弱引用键可正常回收。
        self._http_clients: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, httpx.AsyncClient] = weakref.WeakKeyDictionary()

    def _limiter_for_loop(self) -> _SlidingWindowLimiter:
        """获取当前事件循环的限速器（按 loop 惰性创建，P4-D4）。"""
        loop = asyncio.get_running_loop()
        lim = self._limiters.get(loop)
        if lim is None:
            lim = _SlidingWindowLimiter(rpm=self._limiter_rpm,
                                        min_interval=self._limiter_min_interval)
            self._limiters[loop] = lim
        return lim

    async def close(self) -> None:
        """关闭全部 HTTP 连接（进程退出时调用）。"""
        # 遍历 keys 快照 + pop：弱引用键在迭代期间被 GC 淘汰时，
        # 直接遍历 values() 会抛 RuntimeError（字典迭代中尺寸变化）
        for loop in list(self._http_clients.keys()):
            c = self._http_clients.pop(loop, None)
            if c is not None:
                try:
                    await c.aclose()
                except Exception:  # noqa: BLE001
                    pass

    async def close_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """关闭并移除绑定指定事件循环的 HTTP 客户端（后台 worker 退出前调用）。

        幂等：条目不存在时静默返回。供 B/D 模块的 worker finally 使用——
        每个后台任务新建独立事件循环，任务结束必须显式释放该 loop 的 client，
        否则连接池随条目残留（即使 WeakKeyDictionary 会在 loop 死亡后回收，
        显式关闭可立即释放 TCP 连接而非等 GC）。
        """
        c = self._http_clients.pop(loop, None)
        if c is not None:
            try:
                await c.aclose()
            except Exception:  # noqa: BLE001
                pass

    def invalidate_cache(self) -> None:
        """配置变更后清空缓存。"""
        self._cache.clear()
        self._cache_at = 0.0

    def _client_for_loop(self) -> httpx.AsyncClient:
        """获取当前事件循环的 HTTP 客户端（惰性创建，连接复用）。"""
        loop = asyncio.get_running_loop()
        c = self._http_clients.get(loop)
        # 弱引用键字典按 loop 身份匹配；c.is_closed 兜底重建（close() 后复用路径）
        if c is None or c.is_closed:
            c = httpx.AsyncClient(timeout=httpx.Timeout(90.0))
            self._http_clients[loop] = c
        return c

    async def _load_config(self) -> dict:
        """读取 AI 配置：优先数据库 KV，其次 .env 默认。

        无锁实现：曾用 asyncio.Lock，后台 AI worker 线程（独立事件循环）与主循环
        共用本实例 → 锁绑定第一个循环，其它循环 acquire 抛
        "bound to a different event loop"，AI 调用间歇性/永久失败。
        KV 读取幂等，并发重复读无害；GIL 下 dict 赋值原子。
        """
        now = time.time()
        if self._cache and now - self._cache_at < 30:
            return dict(self._cache)
        # 并行读取全部配置项（曾 5 次顺序 kv_get）
        provider, base_url, api_key, model, temperature, max_tokens, model_map = await asyncio.gather(
            self._db.kv_get("ai_provider"), self._db.kv_get("ai_base_url"),
            self._db.kv_get_secret("ai_api_key"), self._db.kv_get("ai_model"),
            self._db.kv_get("ai_temperature"), self._db.kv_get("ai_max_tokens"),
            self._db.kv_json_get("ai_model_map"),
        )
        provider = provider or self._default.get("ai_provider", "openai")
        base_url = base_url or self._default.get("ai_base_url", "")
        api_key = api_key or self._default.get("ai_api_key", "")
        model = model or self._default.get("ai_model", "gpt-4o-mini")
        temperature = float(temperature or self._default.get("ai_temperature", 0.7))
        max_tokens = int(max_tokens or self._default.get("ai_max_tokens", 2048))
        # P5：按功能路由模型（feature -> 模型名）。KV `ai_model_map` 优先，settings 兜底。
        model_map = (model_map if isinstance(model_map, dict) and model_map
                     else (self._default.get("ai_model_map") or {}))
        # P4-D3：默认 model_map 分流——用户未配置时，若主模型是强模型（gpt-4o/o1/
        # deepseek-reasoner 等），轻任务（复盘/校验/市场解读/参数优化）自动落到便宜
        # 快模型，避免"高成本模型跑低成本任务"的浪费；重任务（策略设计/迭代/因子
        # 挖掘）保持用主模型。用户显式配置 ai_model_map 时完全以其为准。
        if not model_map:
            mm = {feat: model for feat in AI_FEATURE_PARAMS}
            _strong = (model.startswith(("gpt-4o", "o1", "o3", "deepseek-reasoner", "deepseek-r1"))
                       and "mini" not in model and "flash" not in model)
            if _strong:
                cheap = "gpt-4o-mini"
                for _light in ("trade_review", "validate_pine", "market_analysis", "param_optimize"):
                    mm[_light] = cheap
                log.info("[ai] 主模型 %s 为强模型，轻任务默认分流到 %s（可在 AI 配置页覆盖）", model, cheap)
            model_map = mm
        self._cache = {
            "provider": provider, "base_url": base_url, "api_key": api_key,
            "model": model, "temperature": temperature, "max_tokens": max_tokens,
            "model_map": dict(model_map or {}),
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
        # P5：按功能路由模型（feature -> 模型名，见 _load_config 的 model_map）。
        # 在 api_key 检查前应用；后续 REASONING_MODELS / json_schema 能力判定均按路由后模型。
        if feature and cfg.get("model_map", {}).get(feature):
            cfg["model"] = str(cfg["model_map"][feature])
        if not cfg["api_key"]:
            raise AINotConfigured("未配置 AI API Key（请在「AI 与 API」页设置）")
        url = await self._resolve_url(cfg)

        # 按功能预设参数（若 feature 存在且预设了）；temperature 显式传参最优先
        fp = AI_FEATURE_PARAMS.get(feature or "", {})
        temp = temperature if temperature is not None else fp.get("temperature", cfg["temperature"])
        max_tokens = extra_max_tokens or fp.get("max_tokens", cfg["max_tokens"])
        # JSON 任务自动扩容：保证 reasoning + JSON 都有空间。
        # P4-D5：小任务不过度分配——此前 json_mode 且 max_tokens<4096 时一律顶到
        # 8192，简单校验任务（validate_pine 预设 2048，几十 token 的答案）也被分配
        # 8192 上限，模型倾向输出更长、推理模型会用满预算。改为按功能定下限：
        # 大输出功能仍扩到 _JSON_MAX_TOKENS；小任务只扩到其预设值与 _JSON_MIN_TOKENS
        # 的较大者（约 4096），避免输出预算超配。
        if json_mode and max_tokens < _JSON_MIN_TOKENS:
            fp_default = fp.get("max_tokens", 0)
            if fp_default and fp_default < _JSON_MAX_TOKENS:
                # 小任务（预设输出空间小于 8192）：只扩到预设与最小安全值中较大者
                max_tokens = max(fp_default, _JSON_MIN_TOKENS)
            else:
                max_tokens = _JSON_MAX_TOKENS
        # P4-A3：发送前把 max_tokens clamp 到模型上限。此前 strategy_design/iterate
        # 预设 12000，超过 DeepSeek 系列上限 8192，发送时直接 400 必失败（_max_token_cap
        # 只在扩容重试路径用到，发送前从不 clamp）；未知模型回退默认上限。
        model_cap = _max_token_cap_for_model(cfg["model"])
        if max_tokens > model_cap:
            log.warning("[ai] max_tokens=%d 超过模型 %s 上限 %d，clamp 到 %d",
                        max_tokens, cfg["model"], model_cap, model_cap)
            max_tokens = model_cap

        headers = {"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"}
        base_body: dict[str, Any] = {
            "model": cfg["model"],
            "messages": messages,
            "max_tokens": max_tokens,
        }
        # C1：推理模型（DeepSeek-R1/o1/o3 等）不支持 temperature（传了会 400 或静默忽略），
        # 显式跳过；非推理模型才携带 temperature（P4-A4：前缀匹配，兼容模型变体名）
        if not _is_reasoning_model(cfg["model"]):
            base_body["temperature"] = temp
        if json_mode:
            # P1：若启用 ai_use_json_schema 且该功能有 JSON Schema 且模型支持，
            # 用 response_format=json_schema 做机器级字段约束；否则 json_object。
            # 不支持/报 400 时自动回退 json_object（见下方 schema_active 分支）。
            schema_active = False
            from .schemas import schema_for
            sch = schema_for(feature or "") if feature else None
            if (bool(self._default.get("ai_use_json_schema", False))
                    and sch is not None and _supports_json_schema(cfg["model"])):
                schema_active = True
                base_body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": feature or "output", "strict": False, "schema": sch},
                }
            else:
                base_body["response_format"] = {"type": "json_object"}
            if not _contains_json_instruction(messages):
                messages = [dict(m) for m in messages]
                messages[-1]["content"] = messages[-1]["content"] + "\n\n（必须输出严格 JSON，不要输出任何多余文字）"

        retries = int(self._default.get("ai_max_retries", 3))
        last_err: Optional[Exception] = None
        # 长生命周期连接复用（避免每次请求 TCP+TLS 握手）
        client = self._client_for_loop()
        for attempt in range(retries + 1):
            await self._limiter_for_loop().acquire()
            body = dict(base_body)
            if isinstance(last_err, _EmptyContentError):
                # 扩容重试：推理模型 reasoning 占满预算导致 content 为空。
                # 封顶：按模型动态选择（DeepSeek 8192，GPT-4o 32000）
                cap = _max_token_cap_for_model(cfg.get("model", ""))
                # P2-9：扩容结果写回 base_body 并沿用——曾只在 attempt==1 扩容一次，
                # 后续重试退回小 max_tokens，扩容重试形同虚设
                base_body["max_tokens"] = min(max(max_tokens * 2, _JSON_MAX_TOKENS * 2), cap)
                log.warning("[ai] content 为空，扩容 max_tokens=%d 重试", base_body["max_tokens"])
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
                    # 指数退避 + 抖动（封顶 30s：attempt 过大时 2**attempt 溢出）
                    wait = min(2 ** attempt, 30.0) + random.uniform(0, 0.5)
                    if resp.status_code == 429:
                        # 429 优先遵循服务端 Retry-After
                        ra = resp.headers.get("Retry-After")
                        if ra and ra.strip().isdigit():
                            wait = min(float(ra.strip()), 30.0)
                    log.warning("[ai] 临时错误(第%d/%d次) %s，%.1fs 后重试", attempt + 1, retries, last_err, wait)
                    await asyncio.sleep(wait)
                    continue
                # P1：json_schema 被服务端拒绝（400，多因模型不支持）→ 回退 json_object 重试
                if resp.status_code == 400 and schema_active:
                    schema_active = False
                    base_body["response_format"] = {"type": "json_object"}
                    log.warning("[ai] 模型不支持 json_schema(400)，回退 json_object 重试: %.200s", detail)
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
                wait = min(2 ** attempt, 30.0) + random.uniform(0, 0.5)
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
                # 修正消息必须把 AI 输出序列化为字符串（assistant content 只接受 str，
                # 传 dict 会被 OpenAI 兼容接口 400 拒绝 → 修复重试机制整体失效）
                msgs.append({"role": "assistant",
                             "content": json.dumps(data, ensure_ascii=False)})
                # P4：注入合规示例（few-shot），提升复杂 schema 的一次修正通过率
                example = _FEWSHOT_EXAMPLES.get(feature)
                retry_prompt = fb + "\n请严格按修正要求重新输出完整 JSON。"
                if example:
                    retry_prompt = (
                        "参考一个合规输出示例（仅作格式/字段参考，数值请勿照抄）：\n"
                        f"{example}\n\n" + retry_prompt)
                msgs.append({"role": "user", "content": retry_prompt})
                attempt += 1
        raise AICallError(f"[{feature}] 校验重试超限")

    @staticmethod
    def _parse_json(text: str) -> Optional[dict]:
        t = text.strip()
        # 提取 ```json ... ``` 代码块（大小写不敏感）
        if "```" in t:
            for block in t.split("```"):
                block = block.strip()
                if block.lower().startswith("json"):
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

    async def list_models(self, base_url: str, api_key: str) -> list[str]:
        """拉取 OpenAI 兼容 /models 端点返回的可用模型列表（Cherry Studio 式配置）。

        base_url/api_key 为表单临时值：不保存、不落日志、不回显。
        失败抛 AICallError（带友好提示）；不支持 models 端点的服务可保留手动输入。
        """
        base = (base_url or "").strip().rstrip("/")
        if not base:
            raise AINotConfigured("请先填写 API 地址（Base URL）")
        if not api_key:
            raise AINotConfigured("请先填写 API Key")
        headers = {"Authorization": f"Bearer {api_key}"}
        client = self._client_for_loop()
        try:
            r = await client.get(base + "/models", headers=headers,
                                 timeout=httpx.Timeout(20.0))
            if r.status_code == 401:
                raise AICallError("API Key 无效（401）：请检查密钥后重试")
            if r.status_code == 404:
                raise AICallError("该服务不支持 /models 端点（404），请手动输入模型名")
            r.raise_for_status()
            data = r.json()
            models = sorted({m.get("id", "") for m in (data.get("data") or []) if m.get("id")})
            if not models:
                raise AICallError("该服务返回的模型列表为空")
            return models
        except AICallError:
            raise
        except httpx.HTTPStatusError as e:
            raise AICallError(f"模型列表拉取失败: HTTP {e.response.status_code}")
        except httpx.HTTPError as e:
            raise AICallError(f"模型列表拉取失败（网络不可达或地址错误）: {e}")

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
    """检查提示词是否已要求 JSON 输出（避免重复注入指令）。

    使用更精确的匹配：```json 代码块标记、JSON 格式关键词、respond in JSON 等。
    """
    keywords = ["```json", "```JSON", "json format", "JSON format",
                "respond in json", "respond in JSON",
                "输出 json", "输出 JSON", "返回 json", "返回 JSON"]
    for m in messages:
        content = m.get("content") or ""
        for kw in keywords:
            if kw in content:
                return True
    return False
