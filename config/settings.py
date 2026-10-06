"""统一配置：从 .env + config/config.yaml 加载（yaml 键名与字段同名，扁平结构）。

非敏感默认值可被 Web 界面覆盖。优先级：环境变量 > .env > config.yaml > 代码默认。
"""
import json
import logging
import socket
import threading
from functools import lru_cache
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import YamlConfigSettingsSource

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# 常见本地代理端口（Clash / V2Ray / FClash 等），用于自动探测
_COMMON_PROXY_PORTS = (7890, 7891, 7892, 7897, 7898, 1080, 10808, 10809, 8888, 33210, 2080, 2081, 8080)

# P2-4：代理探测结果进程级缓存（探测是阻塞 socket 连接，最坏
# 13 端口 × 0.4s ≈ 5s；缓存后仅首次访问触发，lifespan 异步预热避开请求路径）
_PROXY_CACHE: Optional[str] = None
_PROXY_READY: bool = False
_PROXY_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def probe_local_proxy() -> str:
    """自动探测本机正在监听的 HTTP(S) 代理端口，返回如 http://127.0.0.1:7890，未找到返回空串。

    探测结果是进程级缓存，只做一次端口连通性检查（耗时约几十毫秒）。
    """
    for port in _COMMON_PROXY_PORTS:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.4):
                url = f"http://127.0.0.1:{port}"
                log.info("自动探测到本地代理: %s", url)
                return url
        except OSError:
            continue
    return ""


def normalize_proxy_url(url: str) -> str:
    """代理 URL 规范化：`socks5://` → `socks5h://`（远端 DNS 解析）。

    本机 DNS 常被污染/劫持（2026-09-27 实测 `www.okx.com → 169.254.0.2`），
    `socks5://` 会把域名在**本机**解析出的废 IP 送给隧道，请求必然超时；
    `socks5h://` 改由代理侧解析（等价 curl `--socks5-hostname`），requests/urllib3
    与 httpx 均支持该 scheme。已是 socks5h 或其他协议则原样返回。
    """
    if not url or "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    if scheme.lower() == "socks5":
        return "socks5h://" + rest
    return url


def ccxt_async_proxy_cfg(proxy: Optional[str]) -> dict:
    """把应用级代理翻译成 ccxt **异步**客户端可接受的配置（只返回其一）。

    ccxt async 的 REST 走 aiohttp：`aiohttp_proxy` 只支持 http/https；socks 必须用
    `socksProxy`（ccxt 内部改走 aiohttp_socks，并识别 `socks5h://` 保留远端 DNS）。
    两者同时设置 ccxt 会直接抛 `NotSupported: you have set multiple proxies`。

    2026-09-27 实测：socks 误塞进 aiohttp_proxy → aiohttp 不认 socks scheme →
    `ClientOSError [WinError 64]` → `/api/exchanges/test` 502（用户看到的
    「连接失败（网络受限）」）。改为 socksProxy 后 6/6 全通。
    """
    if not proxy:
        return {}
    scheme = proxy.split("://", 1)[0].lower() if "://" in proxy else ""
    if scheme.startswith("socks"):
        return {"socksProxy": proxy}
    return {"aiohttp_proxy": proxy}


class _Utf8YamlSource(YamlConfigSettingsSource):
    """UTF-8 读取 config.yaml：pydantic-settings 默认以系统编码（中文 Windows=GBK）
    文本模式打开，UTF-8 中文注释直接 UnicodeDecodeError（启动即崩）。
    """

    def _read_file(self, file_path):
        import yaml
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        yaml_file=str(ROOT / "config" / "config.yaml"),
        extra="ignore",
        case_sensitive=False,
    )

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings,
                                   dotenv_settings, file_secret_settings):
        """注册 config/config.yaml 为配置源（曾仅声明 yaml_file 无 source，文件被静默忽略）。"""
        return (init_settings, env_settings, dotenv_settings,
                _Utf8YamlSource(settings_cls), file_secret_settings)

    # 安全
    master_key: str = ""       # 可选：固定主密钥（.env 配置），留空则自动生成 data/master.key
    # Web API 访问令牌：留空则启动时自动生成并打印到日志（前端通过 Cookie 自动携带）
    api_token: str = ""
    # 数据库（默认绝对路径，避免从其他目录启动时数据分裂到 CWD）
    database_url: str = f"sqlite+aiosqlite:///{ROOT / 'data' / 'trader.db'}"
    # AI
    ai_provider: str = "openai"
    ai_base_url: str = "https://api.openai.com/v1"
    ai_api_key: str = ""
    ai_model: str = "gpt-4o-mini"
    ai_max_retries: int = 3
    ai_rate_limit_rpm: int = 30
    ai_rate_limit_min_interval: float = 2.0
    # P1：结构化输出用真 JSON Schema（response_format=json_schema）而非自由文本描述。
    # 默认关（向后兼容）：开启后仅对"支持 json_schema 的模型"生效，
    # 不支持/报错时自动回退 json_object。DeepSeek 等不支持 json_schema 的模型不受影响。
    ai_use_json_schema: bool = False
    # P5：按功能路由模型（feature -> 模型名），如 {"market_analysis": "gpt-4o", "factor_mine": "gpt-4o-mini"}。
    # 高价值决策用强模型、批量化用便宜模型；可同时在数据库 KV `ai_model_map` 覆盖。
    ai_model_map: dict[str, str] = {}
    # 交易
    paper_trading: bool = True
    log_level: str = "INFO"
    log_dir: str = str(ROOT / "data" / "logs")
    # Web
    web_host: str = "127.0.0.1"  # 默认仅本机访问；需要局域网访问时用 --host 0.0.0.0 或改 .env
    web_port: int = 8000
    # 网络代理（用于访问 OKX / Bitget / Bybit 等受限交易所）
    # 留空则自动探测本机常见 Clash/V2Ray 端口；也可显式指定如 http://127.0.0.1:7890
    proxy_url: str = ""
    # 默认
    default_exchange: str = "binance"
    default_symbol: str = "BTC/USDT"
    default_timeframe: str = "1h"
    default_start_cash: float = 10000.0
    trading_auto_start: bool = False
    ai_market_analysis_interval: int = 600
    ai_optimize_interval: int = 86400
    ai_review_interval: int = 604800
    default_strategy: str = "dual_ma"
    # 交易所断链自动平仓：None=按当前周期自动取 2 根 K 线时长；0 或负数=显式关闭；>0=自定义秒数
    max_stale_seconds: Optional[int] = None
    # 纸面/模拟成交滑点（默认与回测引擎一致 0.0005，避免纸面比回测更乐观；
    # 仅对市价单生效，限价单按限价成交不叠加滑点）
    paper_slippage: float = 0.0005
    # 成交时点：True = 信号在下一根K线开盘成交（与回测一致，杜绝前视、更保守）；
    # False = 当前K线收盘即时成交（默认，向后兼容）。
    trade_on_open: bool = False
    # 实盘服务端兜底止损（仅 live 模式生效）：成交后在交易所挂触发式市价卖单，
    # 进程崩溃/断网/机器休眠时交易所侧仍有一道防线
    protective_stop_enabled: bool = True
    # 策略 params 里没有 stop_loss_pct 时使用的止损距离（相对持仓均价）
    protective_stop_default_pct: float = 0.07
    # 在引擎软止损基础上再往深处让出的幅度：保证正常情况下由引擎先平仓，
    # 兜底单只在本地链路真的失效时才成交
    protective_stop_buffer_pct: float = 0.02
    # AI 参数热更新验证门：应用 AI 建议参数前先跑回测 + 过拟合校验，通过才应用
    ai_optimize_validate: bool = True
    # 验证用历史K线根数（建议 >= 400 以便过拟合检测可执行）
    ai_optimize_validation_bars: int = 800

    # 持续进化引擎（默认开启，间隔秒）
    evolve_enabled: bool = True
    evolve_factor_miner_interval: int = 60     # 因子挖掘训练间隔：60s（连续训练）
    evolve_strategy_drl_interval: int = 60      # 策略 DRL 训练间隔：60s（连续训练）
    evolve_rolling_window: int = 5000             # 滚动窗口K线数
    evolve_factor_miner_episodes: int = 8         # 每轮因子挖掘训练轮数（连续训练取小值）
    # E-EXP（2026-09 实验落地）：因子挖掘连续 OOS 被拒达到此阈值后，下一轮强制
    # from-scratch 重训（base_agent=None）。证据：生产 9/13-9/14 连续 15+ 轮被拒
    # （|rank_ic| 0.004~0.028<0.03）而同窗口 masked from-scratch 过门槛率 44%
    # （.optim/exp_training/replicate_prod_windows.py）——续训循环会困在老化基线上，
    # 周期重置可打破"永久连拒"。0=关闭。
    evolve_factor_miner_reset_after_rejects: int = 5
    # 因子挖掘训练目标口径（2026-09-29 成本对齐修复）：
    #   "ic"  = 旧行为：|rank_ic| × √(1/max(turnover,0.125))——抽象选择质量，低换手
    #           奖励会虚高分数（现场实测 fitness 0.35 而真实 IC 仅 ~0.07），与真实
    #           回测收益脱节（5m ETH 300+ 笔交易 -34.6%）。
    #   "net" = 验证段可交易净收益（方向收益 − 换手成本）× scale：训练目标直接
    #           对齐"扣成本后能赚多少"，高换手组合被真实成本压制。
    # 注意：改口径会触发锚点不可比 → 自动重建基线（config 指纹含此项）。
    evolve_factor_miner_fitness_mode: str = "net"
    evolve_factor_miner_cost_per_turn: float = 0.002   # 方向翻转双边费率（fee 0.1% ×2）
    evolve_factor_miner_net_fitness_scale: float = 500.0
    # 策略 DRL 每轮训练轮数。8 → 32 → 128（2026-10 修复配套 + 成本标定后的训练量）：
    # 实测（.optim/exp_f3_history.py、exp_g_deep_train.py）同一份数据下
    #   episodes=8   → val -0.0031、OOS -0.0084（学不出东西）
    #   episodes=40  → val +0.0130、OOS -0.0032
    #   episodes=150 → val +0.0196、OOS +0.0161（仅手续费口径；含滑点 -0.0129）
    # 8 轮远在学习曲线的起点之前；1h/k=8 下 32 轮实测 14.2s，128 轮 ≈57s，
    # 仍在 evolve_train_time_budget=120s 预算内（预算本身就是 CPU 上限，超时提前收尾）。
    evolve_strategy_drl_episodes: int = 128
    # 策略 DRL 网络规模：导出 Pine 时权重必须整段内联，故默认取「装得进 Pine」的组合。
    # 旧配置 state_window=20 + hidden=[64,64] = 21184 权重 ≈700KB 脚本 → 进化产物
    # 永远没有可运行的自动交易代码（图上买卖点无从谈起）。改回大网络即放弃 Pine 导出。
    evolve_strategy_drl_state_window: int = 4
    # 训练环境最小交易区（TradingEnv.min_trade_zone）：目标仓位差异小于该比例不开仓。
    # 0.05（旧默认）→ 0.10：让模型学会少动、降换手/成本（v44/v47 长窗亏损主因之一
    # 是高换手成本侵蚀；buy_zone 部署侧扫描无效，须在训练侧解决）。
    evolve_strategy_drl_min_trade_zone: float = 0.10
    evolve_strategy_drl_hidden: list[int] = [16, 16]
    # 策略 DRL 经验回放（跨训练轮存活）：每轮只训 8ep + 滚动窗口移动，若缓冲每轮
    # 重建，旧行情经验在轮间被完全丢弃——这是"灾难性遗忘"的加速器。>0 时引擎持有
    # 缓冲区实例并逐轮透传，上一轮乃至更早窗口的轨迹继续参与 PPO 更新（旧轨迹经
    # 重要性采样比率自然衰减，PPO 兼容）。0=禁用（保持旧行为）。
    #
    # 内存口径（实测 2026-09-28，5m/5000根/state_window=4）：单条轨迹 ≈1.24MB
    # （2939 步 × 54 维 float64）。P0 并行采样上量后 n_episodes=16、episodes=128 →
    # 每轮推入 128×16 ≈2048 条 ≈2.5GB——**不可能全留**（2048×1.24MB 常驻不可行）。
    # 2026-10 审计修复（MED）：capacity 是"保留最近一小段"的内存取舍而非"完整跨轮
    # 记忆"——默认 128 ≈155MB，保留最近 ~1/16 轮（约 8 次 PPO 迭代 ×16 并行轨迹）；
    # 老轨迹经重要性采样比率自然衰减（PPO 兼容）。要更强跨轮记忆按此公式调大，
    # 400≈484MB（不建议）。
    evolve_strategy_drl_replay_capacity: int = 128
    evolve_strategy_drl_replay_mix_ratio: float = 0.3   # 每批中来自回放的历史轨迹比例
    # P0 多标的混合训练（2026-10 经验广度）：开启后策略 DRL 每轮把 evolve_symbols
    # 池中的全部标的喂给 train_drl，每个 episode 随机选标的×随机时间窗切片，
    # 网络共享——补"见过多种价格行为"的经验广度（人类几年=牛熊震荡多标的多场景）。
    # 主标的仍承担 val/OOS/部署语义；额外标的只作为经验来源。0=关闭（单标的训练）。
    evolve_strategy_drl_multi_symbol: bool = True
    # 每条轨迹来自额外标的的概率（0=只用主标的；1=只用额外标的）。主标的经验密度
    # 更高、部署语义最强，默认 0.5 让两者均衡。
    evolve_strategy_drl_multi_mix_ratio: float = 0.5
    evolve_rollback_threshold: float = 0.95       # 新模型低于最佳模型此比例时回退（仅旧 fitness 为正时生效）
    evolve_max_versions: int = 10                 # 保留的历史版本数
    evolve_symbols: list[str] = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]  # 训练标的池（留单个=固定品种训练）
    evolve_timeframe: str = ""                  # 训练时间周期（空则跟随 default_timeframe；指定如 "4h" 则独立训练周期）
    evolve_oos_min_bars: int = 250                # OOS 安检最小样本数（<此值视为无样本外证据）
    evolve_vol_penalty: float = 40.0              # 策略 DRL 波动率惩罚（20→40 实验提升：
                                             # .optim/exp_training/sweep_drl_rewards.py，30ep 双标的：
                                             # OOS 收益 BTC -0.006→+0.0005、ETH +0.0005→+0.0057，
                                             # 配合 reward_losing_penalty=0.5；baseline 在 BTC 实为负）
                                             # 2026-10 复核（.optim/exp_vol_sweep.py，5 标池 3-seed，
                                             # levels 20/40/60/80 逐 seed 结果完全一致）：本参数只在
                                             # reward_mode=absolute 的旧口径参与 reward；生产 excess
                                             # 口径的风险项由 evolve_strategy_drl_risk_coef 驱动，
                                             # vol_penalty 为死参数（调整无效，防未来误调）。
    evolve_cross_symbol_oos: bool = True          # 是否启用跨标的传导 OOS（P1-8）
    evolve_min_new_bars: int = 2                  # 增量训练触发阈值：新增 K 线达此根数才训练（1h 周期 2 根≈2 小时）
    # P0：单轮训练时间预算（秒）。>0 时训练到预算即提前结束（续训场景"学一点就停"
    # 优于"从头重训"，超时后 OOS 门照常把关防过拟合）；0 = 不限时跑满 episodes。
    # 配合续训（base_agent）让 60s 周期内的每轮训练都变得轻量，成本大幅下降。
    # 第2项完善：默认 120s 单轮预算——实测大窗口（5000 根）单轮 PPO 在低配环境
    # 可超分钟级，60s 周期下无预算上限会让后台训练常驻吃 CPU。面板可热调。
    # 2026-10 策略连拒排查：120s 预算 + 观测开销叠加可能挤压策略训练充分性；
    # 实验（exp_15m_tb/factorblock）证明完整训练 OOS 转正 +2/3 过硬门 → 提到 300s。
    evolve_train_time_budget: float = 300.0
    # P1：增量门按训练周期自动换算最小新增K线根数——训练间隔至少覆盖
    # evolve_min_train_gap_sec 秒（5m 周期 2 根=10 分钟一轮仍偏密，换算后更贴合
    # 训练成本与行情节奏）。0 = 关闭自适应，仅用 evolve_min_new_bars。
    evolve_min_train_gap_sec: int = 0
    # 第2项完善：连续 OOS 被拒（训练正常但安检门连续拦截）自动降频阈值。
    # 被拒说明当前数据窗口暂时供不出有效因子，60s 一轮全量重训纯烧 CPU——
    # 连续达到该次数后该管线训练间隔放大 5 倍（如 60s→300s）；成功一轮即重置。
    # 0 = 关闭（保持固定间隔）。
    evolve_reject_downshift_after: int = 6
    # P1：交易所故障期是否用合成数据训练（True=保留兜底训练；False=直接跳过，
    # 彻底省掉故障期 60s 一轮的全量重训 CPU 成本）。demo 模型本就不会被部署，
    # 故障期训练只烧电不产出，默认关。
    evolve_train_on_demo: bool = False
    # P0：锚点 fitness 量级上限。demo 合成数据能跑出 1267 这类量级，真实行情
    # 0.3 已算不错；data_source 缺失的老锚点无法识破时会永久锁死迭代。
    # 超过此上限的锚点判为不可比（禁止回退比较，等新模型重建基线）。
    # 0 = 关闭该检查。
    evolve_anchor_max_fitness: float = 50.0
    # P0-族群：策略 DRL 冠军/挑战者 K=2 族群训练开关。开启后训练轮次在两个
    # 谱系间交替（每轮仍只训一个模型，墙钟成本不变）：冠军谱系照常部署/注册；
    # 挑战者谱系独立槽位（strategy_drl_alt）提供多样性，须显著优于冠军才晋升，
    # 防"运气好"的模型靠噪音上位（选择只看 OOS，不看训练段）。
    evolve_strategy_drl_population: bool = True
    # 训练能力参数（2026-09-21，诊断 §15 剩余杠杆）：批量收集轨迹数（n_episodes）
    # 与 PPO 更新轮数（ppo_epochs）。4→8 提升样本效率/去方差，观察模型是否从
    # "只在训练段交易"变成 val/OOS 段也交易。两参数不进 _training_identity，
    # 改动不触发锚点身份不可比。0=回退引擎默认（4/4）。
    # P0 并行采样上量（2026-10）：默认 16——多标的混合训练后每 episode 是
    # 独立的"标的×时间窗"切片，批量越大、每轮见过的局面越多，PPO 梯度越稳。
    # 并行线程数放开（见 collect_workers，默认自动按 CPU 核数），16 条轨迹不再
    # 是 4 条串行收 4 轮，而是并行一次收齐，墙钟成本几乎不变。
    evolve_strategy_drl_n_episodes: int = 16
    # 并行收集线程数：0=自动（min(n_episodes, max(4, cpu_count))）；>0 显式指定。
    # 此前硬编码 min(n_episodes,4) 是吞吐瓶颈。线程数过高在超线程机抢内存带宽，
    # 自动模式封顶物理核数；低配机可显式设 2-4 省 CPU。
    evolve_strategy_drl_collect_workers: int = 0
    # P1 课程学习（能力层次）：训练早期只从低难度（低波动/趋势清晰）窗口采样，
    # 随训练进度逐步放开高波动段——先学"什么时候该持仓"，再学"什么时候该降仓"。
    # 难度 = 窗口内已实现波动率分位。False=关闭（采样行为与旧路径完全一致）；
    # 开启后 curriculum_start 为起始难度分位（0=只采最易窗口，1=不限）。
    # 注意：开启会走标的池采样（单标的主标的也按难度过滤），与多标的正交叠加。
    evolve_strategy_drl_curriculum: bool = False
    evolve_strategy_drl_curriculum_start: float = 0.3
    # P1 时序网络升级：actor 骨干。默认 mlp（持续进化行为不变）；
    # gru=GRU 序列编码（能力更强，Pine 导出走蒸馏——GRU teacher→MLP student）。
    evolve_strategy_drl_backbone: str = "mlp"
    evolve_strategy_drl_gru_hidden: int = 32
    evolve_strategy_drl_ppo_epochs: int = 8
    # ============ 2026-10 根本性逻辑错误修复（诊断报告 docs/DRL训练根本性逻辑错误-诊断报告.md）============
    # 背景（实测）：旧口径下奖励 = 组合收益(bp) - 40·(仓位·vol%)²——风险项单位与收益项差 10⁴、
    # 且是对持仓的无条件扣分，空仓奖励恒为 0。实测行情涨 8.6%~26.9%、持仓账户收益为正，
    # 但持仓累计奖励全是大负数、躺平恰好 0 = 全局最优 ⇒ DRL 把全部算力学成"永不交易"，
    # 产线 501 轮里 452 轮被 OOS 门拒（fitness 多为 0.0，即一手没做）。
    #   "excess" = 基准相对奖励：奖励 = tanh(相对基准的超额收益) - risk_coef·(偏离基准·vol%)²，
    #              空仓在上涨行情里同样要付机会成本，"躺平"不再是免费的全局最优。
    #   "absolute" = 旧行为（保留以便回退/对照实验）。
    # 注意：reward_mode / decision_interval / risk_coef 属于训练语义，已进 _training_identity，
    # 改动会让旧锚点判为不可比（不再拿"躺平模型"的 fitness 0 去回退新模型）。
    evolve_strategy_drl_reward_mode: str = "excess"
    evolve_strategy_drl_benchmark_position: float = 1.0   # 基准仓位（1.0=买入持有）
    # 跟踪误差惩罚系数（bp 量级 / (%波动率)²）。旧 vol_penalty=40 在 1h/vol≈0.6% 时
    # 满仓每根约扣 14bp，是行情自身漂移的数十倍；0.5 时约 0.18bp，只做平局裁决。
    evolve_strategy_drl_risk_coef: float = 0.5
    # 决策间隔（每 k 根K线决策一次，其间保持仓位）。旧行为 k=1：每根都重新决策，
    # 初始近均匀策略平均 |Δ仓位|≈0.4 ⇒ 每根 0.06% 摩擦 ⇒ 2939 根亏 83%（实测 -82%）。
    # 也就是说"随机试错"本身是自杀，优势里"别交易"的信号压过真实行情信号。
    # k>1：探索成本降 k 倍，且每个决策覆盖 k 根行情（同样墙钟见 k 倍行情 = 样本效率）。
    # 2026-10 成本标定：与 evolve_timeframe=1h 配套取 8（81 小时持有）——
    # 成本壁垒表要求年化夏普 0.40（可争取）；5m 上同样换手纪律要 12~48（不可能）。
    # 见 docs/DRL成本标定-决策频率与滑点复核.md §2。
    evolve_strategy_drl_decision_interval: int = 8
    # 策略 DRL 训练用的历史深度（K线根数）。0=沿用 evolve_rolling_window（旧行为：
    # 5m 下只有 17 天，且每 60s 在同一段上重训）。>0 时拉取该深度并随机采样时间窗，
    # 让本地 3.4 年历史真正进入训练，而不是 17 天反复背。
    evolve_strategy_drl_history_bars: int = 30000
    # 单条 episode 的时间窗长度（0=整段训练段）。与 history_bars 配合：
    # 每 episode 随机取一个 window_bars 长的年代切片，批量 16 条即覆盖 16 个不同年代。
    evolve_strategy_drl_episode_window: int = 3000
    # ============ 2026-10 验收口径与基准对齐（docs/DRL成本标定-决策频率与滑点复核.md §4）============
    # 旧判据 `oos_ret > 0` 与训练目标（基准相对超额）和动作空间（只能做多/空仓）互相矛盾：
    # OOS 是熊市时，多头策略最优解 = 全程空仓 = 0%，而 0 被 `>0` 拒掉 ⇒ 结构上不可能
    # 有任何模型过门（产线实测 452/501 轮被拒，fitness 多为 0.0）。
    # True：`oos_ret - 同段买入持有 > 0` 才算过门——熊市里保护住资本合格，
    #       牛市里"只会满仓躺赢"（超额 0）被拒，只有真择时能过。
    # 报告缺基准字段（老模型/精简报告）时自动退回旧绝对判据。
    evolve_strategy_drl_deploy_requires_benchmark: bool = True
    # 活跃度守卫：OOS 段实收换手基数低于此值判为"几乎没交易"，不构成择时能力。
    # 必要性：基准相对验收下，"从不交易"在下跌窗口会白拿正超额（那是没参与，不是会择时）。
    # 单位=被收费的名义/权益（1.0 ≈ 满仓换一次）。可用 oos_trades / 段末持仓比例回退。
    evolve_strategy_drl_deploy_min_oos_turnover: float = 0.5
    # 分段稳健性（2026-10）：正超额子段必须 ≥ ceil(OOS段数 × 此比例)，默认 2/3。
    # 必要性：中位数超额会被一段灾难行情掩盖——如 [-0.45, +0.02, +0.03] 中位 +0.02>0，
    # 旧门可过，但 3 段里 2 段亏损，"择时能力"并不存在。
    # <=0 = 关闭（回到只查中位超额）；缺逐段字段的老报告自动跳过（向前兼容）。
    evolve_strategy_drl_deploy_min_positive_fraction: float = 2.0 / 3.0
    # fitness 口径：True=验证段**超额**（val_ret − 同段买入持有），False=旧的绝对收益。
    # 同一轮内基准是常数、排序等价（选模不变），但跨轮/跨窗口可比性只有超额能给：
    # 牛市的 0% 与熊市的 0% 在绝对值口径下会被当成同一个数去比锚点。
    evolve_strategy_drl_fitness_excess: bool = True
    # ============ 2026-10 做空（动作空间 −1~+1）============
    # 默认 **False**。为什么默认关：撮合内核 backtest/_matching.py 是现货语义
    #（`sell` 要求 position>0 且 qty ≤ position），纸面/实盘同样没有做空路径——
    # 训练会做空、实盘执行不了 = 训练与部署彻底不是一回事。
    # True 只用于**离线实验/评估**（量出"做空到底能不能在熊市创造超额"）；
    # 含做空档的模型会被部署门显式阻断（_strategy_deploy_gate），Pine 导出同样拒绝。
    # 要真正部署做空，需先完成交易内核改造（撮合/纸面/实盘/Pine 四处，见
    # docs/DRL做空支持-评估与落地.md）。
    evolve_strategy_drl_short_enabled: bool = False
    # ============ 2026-10 训练种子与多 seed 选模（解决"单轮判据像抽奖"）============
    # 实测（docs/DRL做空支持-评估与落地.md §4，生产配置 3 个 seed）：
    # 验证段超额 13.1%/6.4%/16.0%、OOS 超额 +2.4%/−1.5%/−5.3%——单次训练成败接近抛硬币。
    # 固定种子：单 seed 时结果可复现、跨重启可比（旧值 None=每轮随机）。
    # seeds>1：本轮训 N 个 seed、按**验证段超额中位数**选模（OOS 不参与选模——
    # 它是选完之后的独立安检，用它选模等于把样本外泄漏进选择）。
    # 总墙钟 ≈ N × 单 seed 时长（每个 seed 仍受 evolve_train_time_budget 约束）。
    evolve_strategy_drl_seed: int = 42
    evolve_strategy_drl_seeds: int = 1
    # walk-forward 同品种多折观测（2026-09-19，用户选"1"）：每轮 strategy DRL
    # 真实训练后，用部署中的最佳模型在最近 N 根K线上做多折评估（不改接受逻辑），
    # 结果写 models/walkforward_obs.json + 日志摘要，作为"换一段历史也赚"的常规可见指标。
    evolve_strategy_drl_walkforward_obs: bool = True
    evolve_strategy_drl_walkforward_folds: int = 2
    evolve_strategy_drl_walkforward_bars: int = 6000
    evolve_walkforward_min_trades: int = 8
    # 连续拒绝 from-scratch 重置（对齐 factor_miner 先例，诊断报告 §7 建议第2条）：
    # 策略 DRL（仅冠军轮）连续 OOS 被拒达阈值后，下一轮强制
    # 从零重训（base_agent=None），打破"永久连拒"困在老基线上的死循环。0=关闭。
    evolve_strategy_drl_reset_after_rejects: int = 5
    # GAP-01 实盘验收观测（2026-09-19，诊断报告建议第6条(b)）：每轮 strategy DRL
    # 真实训练后，对部署中的最佳模型跑同品种多折 walk-forward × 两档成本（基准/翻倍），
    # 输出 PASS/FAIL（单标的部署豁免跨品种，见 docs/LIVE_READINESS.md E7）。
    # 结果写 models/gap01_obs.json + 日志；不改接受逻辑。
    # 2026-09-21：口径固定为 4 折 × 15000 根（与 scripts/walkforward_rl_evolve.py
    # 一致）——原 2 折 × 6000 根是"最近窗口红利"，v36 假过考试（4 折 FAIL）。
    evolve_gap01_obs: bool = True
    evolve_gap01_folds: int = 4
    evolve_gap01_bars: int = 15000
    evolve_gap01_min_trades: int = 8
    evolve_gap01_cost_multiplier: float = 2.0
    # GAP-01 FAIL 软信号（2026-09-20）：连续 N 轮 GAP-01 未通过（FAIL/INCONCLUSIVE）
    # 触发策略 DRL 冠军轮 from-scratch 重学（补考不过→重学）；PASS 清零、触发清零。0=关闭。
    evolve_gap01_fail_reset_after: int = 3
    # 多窗口连续 PASS 判定（2026-09-21，§16.5 遗留1）：同一部署模型连续 N 次
    # GAP-01 观测全部 PASS 才算"稳定 PASS"（gap01_stable=True，实盘候选依据）；
    # 非 PASS（FAIL/INCONCLUSIVE）或部署版本变化即清零。0=关闭。
    evolve_gap01_stable_pass_required: int = 3
    # 长窗回放观测（2026-09-21，§16.5 升级）：每轮训练后对部署模型跑 30/90 天
    # 真实数据引擎撮合回放，两窗收益均正才 longwindow_verdict=PASS（防"近期
    # 窗口效应"冒充实盘候选——v44 短窗 PASS 但 30 天 -26.6% / 90 天 -67.8%）。
    evolve_strategy_drl_longwindow_obs: bool = True
    evolve_strategy_drl_longwindow_30d_bars: int = 8640
    evolve_strategy_drl_longwindow_90d_bars: int = 25920
    # 长窗观测频率（2026-09-22，训练速度研究）：实测训练仅 ~9.5s/轮，瓶颈在
    # 30/90 天回放（每轮 +数分钟）——默认每 5 轮跑一次长窗观测，进化节奏大幅回升；
    # 1=每轮（旧行为，观察密集但循环极慢）。
    evolve_strategy_drl_longwindow_every_n: int = 5
    # 部署长窗门槛（2026-09-22，§20 落地）：候选模型部署前须过 30 天长窗回放
    #（30d>0 才允许部署；FAIL 仅存档不部署，防高换手短窗模型顶替长窗稳健模型——
    # v50/52/53/55 类长窗 -10%~-56% 却因 OOS 门正被部署的教训）。False=逃生关闭。
    evolve_strategy_drl_deploy_requires_longwindow: bool = True
    # 部署长窗门是否要求 90 天档（2026-09-22，§20.3 升级）：True=候选须 30d 与
    # 90d 均正才部署（与观测口径一致）；False=仅 30d 门槛（默认，控制冠军轮成本）。
    evolve_strategy_drl_deploy_longwindow_90d: bool = False
    # 优化（自测实验，实测数据）：进化组合因子"回测摘要"的窗口上限（根K线）。
    # 摘要属纯展示用途，耗时随窗口线性增长（5m：1000 根 1.9s / 3000 根 5.9s /
    # 5000 根 10s，热点在 combo 因子每根全量重算），训练间隔 60s 时全窗口
    # 回测会显著占 CPU。>0 = 只回测最近 N 根；0（默认）= 全窗口（语义不变）。
    evolve_factor_summary_bars: int = 0
    # AI 数学期望信号门（2026-09-22，§19 落地）：buy 信号在风控通过后，若
    # ai.math_analysis 的期望收益 < signal_math_ev_min 则拦截新开多（负期望
    # 抑制买入，sell/止损/减仓不受影响）。True=默认生效；False=逃生关闭。
    signal_math_ev_gate: bool = True
    signal_math_ev_min: float = 0.0

    @property
    def data_dir(self) -> Path:
        return DATA_DIR

    def _resolve_proxy_sync(self) -> str:
        """实际探测逻辑（阻塞 socket 连接，最坏 13 端口 × 0.4s ≈ 5s）。
        仅首次经 _ensure_proxy 调用，结果进程级缓存。"""
        if self.proxy_url and self.proxy_url.strip():
            url = normalize_proxy_url(self.proxy_url.strip())
            try:
                parts = urlsplit(url if "://" in url else "http://" + url)
                host = parts.hostname or "127.0.0.1"
                port = parts.port or 80
                with socket.create_connection((host, int(port)), timeout=0.4):
                    return url
            except (OSError, ValueError):
                log.warning("显式代理 %s 不可达（按 host 探测），回退自动探测", url)
                return probe_local_proxy()
            return url
        return probe_local_proxy()

    def _ensure_proxy(self) -> str:
        """缓存化访问：探测结果进程级只算一次（线程安全），后续 O(1) 返回。"""
        global _PROXY_CACHE, _PROXY_READY
        with _PROXY_LOCK:
            if not _PROXY_READY:
                try:
                    _PROXY_CACHE = self._resolve_proxy_sync()
                except Exception as e:  # noqa: BLE001
                    log.warning("[settings] 代理探测失败，置空: %s", e)
                    _PROXY_CACHE = ""
                _PROXY_READY = True
            return _PROXY_CACHE or ""

    def reset_proxy_cache(self) -> None:
        """运行时更新 proxy_url 后重置探测缓存（resolved_proxy 下次重新探测生效）。"""
        global _PROXY_CACHE, _PROXY_READY
        with _PROXY_LOCK:
            _PROXY_CACHE = ""
            _PROXY_READY = False

    @property
    def resolved_proxy(self) -> str:
        """实际生效的代理地址（探测结果进程级缓存，仅首次可能阻塞）：
        - 显式配置：按 URL 的 host 探测（曾写死 127.0.0.1，远程代理被静默丢弃）；
          可达性检查失败仅告警并回退自动探测，但显式配置仍优先返回。
        - 留空：自动探测本机常见代理端口。
        - P2-4：结果缓存一次（首次调用最坏 ~5s），lifespan 经 prewarm_proxy
          异步预热，请求路径不再在事件循环上同步阻塞。
        """
        return self._ensure_proxy()

    async def prewarm_proxy(self) -> None:
        """异步预热：把阻塞 socket 探测放线程池（asyncio.to_thread），
        服务启动时调用一次，首个行情请求不再同步阻塞数秒。幂等。"""
        import asyncio
        try:
            await asyncio.to_thread(self._ensure_proxy)
        except Exception as e:  # noqa: BLE001
            log.warning("[settings] 代理预热失败: %s", e)


settings = Settings()


def _load_proxy_override() -> None:
    """启动时恢复网页端设置的代理覆盖（data/proxy_config.json，运行时生效）。"""
    p = ROOT / "data" / "proxy_config.json"
    if not p.exists():
        return
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        v = str(d.get("proxy_url") or "").strip()
        if v:
            settings.proxy_url = v
            settings.reset_proxy_cache()
            log.info("[settings] 已恢复网页端代理覆盖: %s", v)
    except Exception as e:  # noqa: BLE001
        log.warning("[settings] 代理覆盖加载失败: %s", e)


_load_proxy_override()
