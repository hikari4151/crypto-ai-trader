# 持续进化引擎设计文档

## 概述

让 DRL 因子挖掘和策略训练实现"越训练越强"的持续进化能力。核心思路：训练好的模型自动保存、下次启动加载继续训练、新数据滚动窗口增量训练、最佳模型自动回退保护。

## 架构

```
┌──────────────────────────────────────────────────────────────┐
│                   持续进化引擎 (EvolveEngine)                   │
│                                                              │
│  ┌─────────────────────┐    ┌──────────────────────────┐    │
│  │  因子挖掘持续训练     │    │  策略 DRL 持续训练         │    │
│  │  (FactorMinerLoop)   │    │  (StrategyDRLLoop)       │    │
│  │  · 每 N 秒检查新数据  │    │  · 每 N 秒检查新数据       │    │
│  │  · 加载最佳模型继续   │    │  · 加载最佳模型继续        │    │
│  │  · 滚动窗口训练       │    │  · 因子作为额外特征输入    │    │
│  │  · 最佳模型回退       │    │  · 最佳模型回退            │    │
│  └────────┬────────────┘    └───────────┬──────────────┘    │
│           │                             │                     │
│           ▼                             ▼                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                模型仓库 (ModelZoo)                     │   │
│  │  ├─ factor_miner_best.pkl    (因子挖掘最佳模型)        │   │
│  │  ├─ strategy_agent_best.pkl  (策略训练最佳模型)        │   │
│  │  └─ model_versions/          (历史版本快照)            │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │             控制接口 (Control API)                      │   │
│  │  POST /api/evolve/pause    — 暂停持续训练               │   │
│  │  POST /api/evolve/resume   — 恢复持续训练               │   │
│  │  GET  /api/evolve/status   — 查询训练状态               │   │
│  │  POST /api/evolve/rollback — 回退到指定版本             │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. `drl/model_zoo.py` — 模型仓库

管理模型保存、加载、版本回退。

```python
class ModelZoo:
    """模型仓库：持久化训练好的智能体，支持版本管理。"""

    MODELS_DIR: Path  # data/models/

    def save_agent(agent: ACAgent, name: str, version: int) -> Path
        """保存智能体到磁盘，返回路径。"""

    def load_agent(name: str, version: Optional[int] = None) -> Optional[ACAgent]
        """加载智能体。version=None 时加载最新（best）版本。"""

    def best_version(name: str) -> int
        """返回当前最佳版本号。"""

    def list_versions(name: str) -> list[dict]
        """列出所有版本及其元数据（fitness/时间戳/参数）。"""

    def rollback(name: str, version: int) -> bool
        """回退到指定版本（设为最佳）。"""
```

保存格式：使用 `agent.to_dict()` 序列化为 JSON，压缩后存为 `.json.gz`（或 pickle）。

### 2. `drl/evolve_engine.py` — 持续进化引擎

核心调度引擎，集成到 `TradingEngine` 中。

```python
class EvolveEngine:
    """持续进化引擎：管理因子挖掘和策略训练的持续循环。"""

    def __init__(self, db, bus, model_dir: Path):
        self.zoo = ModelZoo(model_dir)
        self._paused = False
        self._factor_miner_task: Optional[asyncio.Task] = None
        self._strategy_task: Optional[asyncio.Task] = None

    async def start(self):
        """启动后台训练循环。"""

    async def stop(self):
        """停止训练循环。"""

    async def pause(self):
        """暂停训练（当前轮完成后停止）。"""

    async def resume(self):
        """恢复训练。"""

    def status(self) -> dict:
        """返回当前训练状态。"""
```

### 3. 因子挖掘持续训练循环

```python
async def _factor_miner_loop(self):
    """因子挖掘持续训练循环。"""
    while self.running:
        if self._paused:
            await asyncio.sleep(10)
            continue
        try:
            # 1. 检查是否有新数据
            df = await self._latest_data(n_bars=5000)
            if df is None or len(df) < 500:
                await asyncio.sleep(60)
                continue

            # 2. 加载已有最佳模型（首次时 None）
            agent = self.zoo.load_agent("factor_miner")
            best_agent = agent

            # 3. 计算因子矩阵
            mat = compute_factor_matrix(df)

            # 4. 训练（使用滚动窗口）
            result = await asyncio.to_thread(
                train_factor_miner, df, mat=mat,
                cfg={**self._base_cfg, "episodes": 20, "seed": None},
                on_progress=self._on_progress
            )

            # 5. 最佳模型回退保护
            if best_agent is not None:
                new_fitness = result["history"][-1]["best_fitness"]
                old_fitness = self._eval_agent_fitness(best_agent, mat, df)
                if new_fitness < old_fitness * 0.95:
                    # 新模型退化超过 5%，回退
                    self.zoo.save_agent(best_agent, "factor_miner", is_best=True)
                    log.warning(...)
                    continue

            # 6. 保存最佳模型
            best_fitness = result["history"][-1]["best_fitness"]
            version = self.zoo.save_agent(
                result["agent"], "factor_miner",
                meta={"fitness": best_fitness, "timestamp": time.time()}
            )

        except Exception as e:
            log.exception(...)
        await asyncio.sleep(self._factor_miner_interval)
```

### 4. 策略 DRL 持续训练循环

同因子挖掘结构，但使用 `train_drl` 并加载因子挖掘产出的组合因子作为额外特征。

### 5. 控制 API

`web/api/evolve.py` — 持续进化控制路由

```python
router = APIRouter(prefix="/api/evolve", tags=["evolve"])

@router.post("/pause")
async def pause(db=Depends(get_db)):
    engine.evolve.pause()
    return {"ok": True}

@router.post("/resume")
async def resume(db=Depends(get_db)):
    engine.evolve.resume()
    return {"ok": True}

@router.get("/status")
async def status(db=Depends(get_db)):
    return engine.evolve.status()

@router.get("/versions")
async def versions(name: str = "factor_miner"):
    return engine.evolve.zoo.list_versions(name)

@router.post("/rollback")
async def rollback(name: str, version: int):
    return {"ok": engine.evolve.zoo.rollback(name, version)}
```

## 数据流

```
新K线到达 → ws_market.upsert()
    ↓
TradingEngine._on_candle() → 策略信号
    ↓
EvolveEngine._factor_miner_loop()  (每 interval 秒)
    ├─ 取最近 5000 根 K 线
    ├─ 计算因子矩阵
    ├─ 加载上次最佳模型
    ├─ 继续训练 20 轮
    ├─ 验证段评估新模型 vs 最佳模型
    ├─ 若退化则回退
    └─ 保存最佳模型
    ↓
EvolveEngine._strategy_drl_loop()  (每 interval 秒)
    ├─ 取最近 5000 根 K 线
    ├─ 加载因子挖掘产出的组合因子
    ├─ 加载上次最佳策略模型
    ├─ 继续训练 20 轮
    ├─ 验证段评估新模型 vs 最佳模型
    ├─ 若退化则回退
    └─ 保存最佳模型
```

## 训练参数

```yaml
# config/config.yaml 新增
evolve:
  enabled: true                    # 是否启用持续进化
  factor_miner_interval: 604800    # 因子挖掘训练间隔（秒，默认 7 天）
  strategy_drl_interval: 86400     # 策略DRL训练间隔（秒，默认 1 天）
  rolling_window: 5000             # 滚动窗口K线数
  factor_miner_episodes: 20        # 每轮因子挖掘训练轮数
  strategy_drl_episodes: 20        # 每轮策略DRL训练轮数
  rollback_threshold: 0.95         # 新模型低于最佳模型此比例时回退
  max_versions: 10                 # 保留的历史版本数
```

## 文件变动清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `drl/model_zoo.py` | 新建 | 模型仓库（保存/加载/版本管理） |
| `drl/evolve_engine.py` | 新建 | 持续进化引擎 |
| `web/api/evolve.py` | 新建 | 控制 API 路由 |
| `engine/trading_engine.py` | 修改 | 集成 EvolveEngine |
| `config/settings.py` | 修改 | 新增 evolve 配置字段 |
| `config/config.yaml` | 修改 | 新增 evolve 默认配置 |
| `web/main.py` | 修改 | 注册 evolve 路由 |
| `web/static/index.html` | 修改 | 前端增加训练控制面板 |

## 过拟合防护

1. **最佳模型回退**：每个训练周期完成后，在未见过的验证段上评估新模型。如果 fitness 低于最佳模型 95%，自动回退。
2. **滚动窗口**：只用最近 N 根 K 线训练，避免模型记住太旧的模式。
3. **训练/验证/OOS 三段切分**：沿用现有 `train_drl` 的防过拟合设计。
4. **版本快照**：保留最近 10 个版本，可手动回退到任意历史版本。
5. **暂停机制**：用户可随时暂停训练，观察当前模型表现。