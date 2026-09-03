# Phase 0 基线存档（2026-08-14）

> **⚠️ 本文档为历史锚点。** 当前有效基线与"禁止回退清单"见 [REDEV_GUIDE.md](REDEV_GUIDE.md) §1/§3：
> 2026-09-01 复测锚定 ret=**-0.206239** / trades=**37**（本文件的 -0.224999 为成本模型增强前的旧值）；
> pytest 已从 9 例增长到 434 例。

重构开始前的验证链基线，后续每个 Phase 的改动必须与此对比。

## 验证链结果

| 项 | 结果 | 备注 |
|---|---|---|
| compileall | ✅ 全通过 | `python -m compileall -q ai backtest core drl engine exchange factors indicators strategies web config scripts run.py` |
| pytest | ✅ 9 passed in 333.67s | 修复：conftest.py 增加 `collect_ignore=["test_ai_guards.py"]`（脚本式文件，顶层 sys.exit） |
| JS 语法检查 | ✅ 通过 | 新建 `scripts/check_js.py`（esprima + ES2020 兼容层：`?.`→`.`、`??`→`||`，词法安全）；esprima 已装入 .venv（dev 工具，不入 requirements） |
| 服务器冒烟 | ✅ | `GET /api/health` 无令牌 200（公开端点）；`/api/portfolio/summary` 无令牌 401 / 带令牌 200；`/api/backtest/history` 无令牌 401 |
| 双引擎一致性基线 | ret=-0.224999, trades=37 | `python run.py backtest --source demo --strategy dual_ma`，结果存 data/backtest_dual_ma.json |

## 环境事实

- git：`.git` 存在但 CLI 不在 PATH/未找到 git.exe → commit 由用户手动执行
- node：环境无 JS 运行时（nodejs.org/npmmirror 均不可达），JS 检查用 esprima 方案
- 端口占用：冒烟用 8017（避开 8000-8010）

## 防回归红线（后续每阶段对照）

1. 双引擎一致性：demo dual_ma 基线 ret=-0.224999 / trades=37 不得漂移（除非有意变更并记录）
2. pytest 9 例全绿
3. 认证：受保护端点无令牌 401
4. JS 语法检查通过

## QA 验收（2026-08-15，Phase 1+2 集成后）

| 项 | 结果 |
|---|---|
| compileall | ✅ |
| pytest | ✅ **79 passed**（基线 9 + 模块 A 23 + B 19 + C 24 + D 4） |
| check_js | ✅ |
| 双引擎一致性 | ✅ ret=-0.224999 / trades=37 零漂移 |
| 冒烟 | ✅ health 200（含 `degraded`/`bus` 新键）、vendor 三文件 200、认证 401/200 |
| 连发回测 ×4 | ✅ 全部落库（dual_ma 37 笔/grid 112/factor_signal 94/price_action 0 笔属 demo 数据性质），日志零跨 loop 错误 |
| DRL 训练 ×2 | ✅ worker 拓扑稳定、策略注册成功（AiStrategy 持久化）、模型元数据键齐全（state_dim/state_window/factor_*） |
| 引擎启停 | ✅ simulated 模式 running 正常、K 线流正常 |
| 环境提示 | ⚠️ 代理 127.0.0.1:7890 不可达（回退自动探测）；AI key 模型不匹配（deepseek-chat 404）——用户环境配置，非代码问题 |
| 浏览器手测 | ⏳ 无浏览器环境，留用户（断接口不卡死/切页图表/OKX 24h 口径） |

### 集成修复记录（2026-08-15，审查问题全部闭环）

- P1×3：optimizer apply=False 返回 AI 建议参数（AI 优化恢复生效）；前端 mktChange null 守卫（OKX 切换不再整页冻结）；前端 equity/trades 裸 await 守卫（轮询不再中断）
- P2×14：回灌幂等（先 pop 防双计）+ payload 补 fee/strategy_name；任务创建逐个收集防孤儿任务；_compute_sr_pa 脏数据兜底；validator bool 穿透拦截；client close() GC 迭代加固；close_client 锁外 await；_run_on_loop 异常噪音 + 超时日志区分；ai.py 超时降级对称；evaluate_agent state_window 优先级；safe_analyze 转发 recent_trades；normSym 支持 OKX；换币点击补 loadTickers
- 已知遗留（非本期）：P2-9 扩容重试持久化、P2-3 并发上限、evaluate 奖励塑形系数元数据（模型 v2）、Adam 熵正则过冲观察
