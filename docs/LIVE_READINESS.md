# crypto_ai_trader 实盘就绪度评估（LIVE_READINESS）

> 定位：本文档回答「**脱离纸面（paper/simulated）之前，还缺什么**」，且每条判断都挂仓库内可核验的证据（代码、测试、探针、基线数值）。
> 适用范围：crypto_ai_trader（2026-09 仓库状态）。修改实盘相关代码后，本文档 §5 的 checklist 应随能力变化更新。
> 关联文档：`docs/REDEV_GUIDE.md`（当前基线/红线）、`docs/BASELINE.md`（历史基线）、`.optim/report_latency2.md`（本地链路时延）、`.optim/log.md`（实盘路径可靠性专项日志）。

---

## 0. 结论速览

| 维度 | 当前状态 | 一句话证据 |
|---|---|---|
| 回测/纸面工程化 | ✅ 强 | 579 tests 全绿、双引擎一致、防前视、成本模型、基准对比（`docs/REDEV_GUIDE.md`） |
| 实盘订单状态机本地演练 | ✅ 较强 | `SimulatedExchange` + 故障注入端到端（`.optim/log.md` round3-7） |
| 本地信号→下单性能 | ✅ 好 | 本机纸面 p95=1.151ms（`.optim/report_latency2.md`） |
| 策略正期望（edge） | ❌ 未证明 | 回测基线 total_return=-0.206239（`data/backtest_dual_ma.json`） |
| 真实交易所 API 验证 | 🚧 代码已支持，实测待凭据 | `exchange/manager.py` 有 testnet 开关 + `.optim/p1_testnet_smoke.py`；尚未用真实 testnet 凭据跑过 |
| 资金安全通道（幂等/紧急平仓） | ✅ 已补 | `clientOrderId`+create 重试/intent 对账认领；`POST /api/trading/close-all` + 前端按钮 |
| 运维/恢复 | ❌ 不足 | `trading_auto_start=false`、约 145 个文件未提交（`docs/REDEV_GUIDE.md`） |

**一句话总结**：纸面到实盘之间最短的路不是加策略，而是补三样东西——① 真实数据上的正期望证明；② 交易所沙箱（testnet）里的真实下单验证；③ 防重复下单 + 紧急离场的资金安全通道。

---

## 1. 「脱离纸面」的判定标准

本项目「脱离纸面」定义为：**用真实资金，在真实交易所的 live 模式下无人值守运行**。达到该状态前，以下 E1–E8 全部通过：

| 编号 | 验收标准 | 判定方式 |
|---|---|---|
| E1 | 策略在真实历史数据上、扣除成本后具有正期望（OOS 证据） | walk-forward / CSCV-PBO 报告：OOS ret>0、DSR 通过、交易样本充足 |
| E2 | 在交易所 sandbox/testnet 上完整验证下单生命周期 | testnet 下单→轮询→撤单→对账→部分成交→触发单，全部有断言 |
| E3 | 下单幂等：create 响应丢失不产生重复单 | 普通订单带 clientOrderId；create 异常路径可重试/可对账 |
| E4 | 人工紧急离场：UI/API 一键清仓 | `close-all` 类接口存在且已测试；停引擎≠清仓被明确提示 |
| E5 | 纸面↔实盘执行偏差可量化 | 回测/纸面默认执行模型已知其简化程度；有真实滑点/部分成交统计 |
| E6 | 实盘监控与恢复闭环 | 断链平仓、每日对账、重启状态回放、止损兜底均有证据，且有人工告警 |
| E7 | 目标标的范围与引擎能力匹配 | 单标的：无要求；多标的/组合：live 引擎支持多仓与组合风控 |
| E8 | 代码基线可复现、可回滚 | 工作区已提交基线快照；验证链命令可复现（`docs/REDEV_GUIDE.md` §4） |

---

## 2. 现状盘点

### 2.1 已具备的强项（保留，不要回退）

- **回测正确性防线**：双引擎共用撮合内核 `backtest/_matching.py`、信号次根开盘成交（防前视）、FIFO 成本、ddof=0、期末强平、成本模型（maker/taker/participation/funding/限价部分成交可选）。见 `docs/REDEV_GUIDE.md` §3-A。
- **实盘订单状态机已被本地演练**：`exchange/simulated.py` 让 simulated 模式真实走 `_place_live`（create→轮询→撤单→复核→对账→FIFO→protective stop），并覆盖部分成交/下单异常/撤单失败/触发单回灌。证据：`tests/test_simulated_live_path.py`、`tests/test_live_branch_faults.py`、`tests/test_protective_e2e.py`、`.optim/log.md` round3-7。
- **风控与资金安全基础**：单笔/每日亏损上限、频率限制、连续亏损冷却（可人工确认）、闪崩/量比/24h 跌幅保护、波动率目标仓位、跨重启状态持久化。见 `engine/risk.py`。
- **服务端兜底止损**：`engine/protective_stop.py` 三条硬约束（仅 live、主动卖出前必撤、失败降级告警）。
- **对账与断链**：每日全量对账（`_daily_reconcile_if_due`）、限价单超时撤单+迟到成交回灌（`_reconcile_loop`）、行情断链自动平仓（live 模式）。
- **本地链路性能**：信号→下单 p95 从 1.866ms 优化到 1.151ms（-38.3%），`risk.check` 大样本 2.527ms→0.291ms。见 `.optim/report_latency2.md`。
- **认证/加密**：API Token + Fernet 密钥落盘、日志脱敏、默认 127.0.0.1。

### 2.2 关键缺口（按影响排序）

| 缺口 | 影响 | 证据位置 |
|---|---|---|
| GAP-01 策略无真实数据正期望证据 | 实盘=直接亏钱风险 | `data/backtest_dual_ma.json`、`tests/test_stat_significance.py` |
| GAP-02 无交易所 sandbox/testnet 验证 | 真实订单行为完全未知 | 全仓库 `grep testnet/sandbox`=0 |
| GAP-03 下单无幂等/创建失败不重试 | 网络抖动可能重复下单或漏单 | `engine/order_manager.py` |
| GAP-04 回测/纸面执行模型偏理想 | 纸面收益系统性高于实盘 | `backtest/engine.py`、`backtest/_matching.py`、`exchange/paper.py` |
| GAP-05 无紧急手动清仓/人工 override | 策略失控时无法快速离场 | `web/api/trading.py`、`engine/trading_engine.py` |
| GAP-06 live 引擎单标的 | 多标的/横截面策略无法实盘 | `engine/trading_engine.py`、`web/api/live.py` |
| GAP-07 运维/恢复证据不足 | 断电/重启/崩溃后行为未验证 | `config/settings.py`、`docs/REDEV_GUIDE.md` |

---

## 3. 差距明细与验收标准

### GAP-01：策略在真实数据上无正期望证据（最致命）

**现状证据**：

- 双引擎一致性基线 `data/backtest_dual_ma.json`：`total_return=-0.206239`、`sharpe=-4.2268`、`profit_factor=0.4374`、`final_equity=7937.61`（demo 合成数据、10000 初始资金）。
- `tests/test_stat_significance.py` L193-195 注释：内置默认策略（dual_ma/factor_signal/grid）在近 7 个月真实 1h K 线上 **OOS 收益为负**，且 avg_oos_trades 仅 3~12 笔。

**影响**：纸面账户里赚不赚钱只是虚拟结果；没有真实数据 OOS 正期望，实盘就是给交易所送手续费。

**验收标准（全部满足才算过）**：

1. 用真实历史数据（`data/klines.db` 或 `run.py backfill` 拉取，≥5000 根 1h）跑 `run.py backtest --source exchange --strategy <候选策略> --symbol BTC/USDT --timeframe 1h`，结果落 `data/backtest_<strategy>.json`。
2. 通过 `backtest/overfit.py` 的 walk-forward/CSCV-PBO 检查：`verdict != "严重过拟合"`，且 **OOS ret > 0**、OOS 交易数 ≥ 30（低频策略另行约定样本量）、DSR 通过。
3. 用 `run.py cost-scan`（`backtest/cost_scan.py`）验证成本敏感性：手续费/滑点翻倍后仍为正期望。
4. 在至少 2 个不相关品种（如 BTC/ETH）上方向一致（跨品种证据，见 `docs/QUANT_ADVICE.md` 🔴2）。

**✅ P0 实测结果（2026-09-18，见 `.optim/p0_report.md` + `.optim/p0_grid_pbo.json`）**：

- 数据：binance 1h 真实 K 线，BTC/ETH/SOL 各约 1.5 万根（2025-01 ~ 2026-09），4 折扩展窗口 walk-forward，成本 fee=0.001/slippage=0.0005。
- **没有任何内置策略达到“通过”标准**：
  - dual_ma：三标的 OOS 全负（-0.112 ~ -0.089），237~239 笔；
  - factor_signal / meta_controller：三标的 OOS 全负（-0.14 ~ -0.19），461~639 笔；
  - price_action：SOL 上 OOS=+0.012 但仅 29 笔（证据不足），BTC/ETH 为负；
  - rl_adaptive：无模型文件时 0 交易（默认参数不可用于实盘）；
  - **grid：BTC OOS=+0.0038（4/4 折为正）/ SOL OOS=+1.3e-05（2/4 折）**，但完整 `detect_overfit` 判定为 **“疑似过拟合”**：BTC DSR=0.68、PBO=0.20；SOL DSR=0.37、PBO=0.45；ETH 为 **“严重过拟合”**（OOS 转负）。
- 成本敏感性（BTC 全周期 cost-scan）：grid 是唯一基准为正的策略（+2.18%），5x 成本下仍 +0.53%（75.6% 跌幅）；其余策略基准即负。
- **结论：GAP-01 未通过。实盘上线的第一道门仍关着——现有内置策略没有跨品种、扣除运气后的正期望证据。**

**✅ P2 参数搜索实测（2026-09-18，见 `.optim/p2_search_report.md`）**：

- 协议：BTC 1h 前 10468 根【搜索窗】选参（93 组：dual_ma 25 + grid 30 + price_action 27 + factor_signal 12），后 4486 根【留出窗】验证；幸存者再跑完整 PBO/DSR + ETH/SOL 留出窗 + cost-scan。
- **幸存者 = 0**：搜索窗内 12 个 Top 候选全是 grid 系（OOS +0.003~+0.005），但**留出窗全部转负**（-0.001~-0.005）——早期行情有效、近期失效，是典型的参数/行情体制漂移。
- 结论：**在当前策略空间（1h 技术指标长线/网格/突破）内，没有找到能通过“选参→留出验证”的稳健正期望参数。下一步要么扩搜索空间（更多标的/周期/反转因子），要么走因子挖掘/AI 通道。**

**✅ P3 扩展搜索实测（2026-09-18，见 `.optim/p3_search_expand.md`）**：

- 范围：5m 周期 BTC/ETH/SOL（96 组网格：快周期双均线/密网格/price_action pullback/factor 反转）+ 8 个主流 altcoin 1h（BNB/XRP/DOGE/LINK/LTC/AVAX/ADA/SUI，68 组网格）；协议同上（前 70% 选参 → 后 30% 留出验证）。
- **幸存者 = 2，且均未通过显著性门**：仅 BNB 1h 的 grid（grid_pct=0.01/qty=0.001/max_positions=20|50）两窗 OOS 均为微正（1.4e-05 → 2e-06），但完整 `detect_overfit` 判定 **“严重过拟合”**（PBO=0.75、DSR=0.48）。
- 结论：**把搜索空间扩大到 5m/反转因子/8 个 altcoin 后，依然没有通过“选参→留出验证→PBO/DSR”的稳健正期望策略。当前证据强烈指向：在现有策略执行器（dual_ma/grid/price_action/factor_signal）的参数空间内找不到可实盘的 edge，需要换策略形态（因子挖掘/AI 设计/组合）或接受“无 edge → 暂不上实盘”。**

**🚧 因子通道实测（2026-09-18，P4/P5/P6）**：

- P4 库因子筛选（`.optim/p4_factor_report.md`）：26 个库因子 × BTC/ETH/SOL 1h × h=1/6/24，OOS 后 30% 分窗 + 跨品种同号门。**19 个因子-周期组合通过门，全部为稳定负 IC（反转信号）**：如 trend_regime h24 mean IC=-0.072/ICIR=-0.65、obv_slope h1 IC=-0.054/ICIR=-0.89、rsi_14 h1 IC=-0.036/ICIR=-1.68。
- P5 反转执行器验证（`.optim/p5_reversal_report.md`）：用正确量纲（bb_pos 0-1、rsi_osc ±40 等）把负 IC 因子转成 factor_signal reversal 策略，走 P2 协议 → **搜索窗 0 候选**：负 IC 未能通过当前执行器转成交易级正收益（止损/止盈/阈值实现损耗）。
- P6 DRL 因子挖掘（进行中，`.optim/p6_drl_miner_report.json`）：本地 PPO 自动选因子组合，绕过 LLM。
- ⚠️ AI 通道阻塞（2026-09-18 复测）：代理已通（OpenAI 走代理可达），但软件内配置的 AI key 在 **OpenAI / DeepSeek / SiliconFlow 三处均返回 401「API Key 无效」**——需要更换有效 AI key；本地 DRL/因子挖掘不受影响。

**✅ P6/P7 DRL 因子挖掘实测（2026-09-18）**：

- **P6 组合因子通过 OOS 质量门**（`.optim/p6_drl_miner_report.json`）：PPO 选出 adx_14/ma_dist_10/macd_hist_pct/bb_pos/bias_20/obv_slope（多数负权重=反转暴露），组合因子 OOS **rank_IC=+0.0507、ICIR=1.26、换手0.13、valid=True**——这是全项目第一个通过质量门的因子。
- **P7 交易级转换失败**（`.optim/p7_combo_report.md`）：把该组合转成 factor_signal combo“阈值多空”（buy z>bt / sell z<st，次根开盘、半仓、fee0.001/slip0.0005），8 组阈值在搜索窗与留出窗**全部亏损**（最佳 search=-0.2785 / holdout=-0.0120），因频繁交易被成本拖垮。
- **结论（关键）**：因子的 IC 真实存在，但**当前单标的、长仓阈值执行器无法把 IC 变成净收益**——这是“IC→PnL 落差”。下一步需要换执行形态：跨品种多空组合（利用负 IC 因子做低/多高/空）或成本感知的阈值/持仓设计（降换手），否则因子挖掘的成果落不了地。

**✅ P8 成本感知执行优化（2026-09-18）**：

- 思路：P7 失败根因是频繁交易被成本拖垮 → 用滞回阈值（entry z=1.5 / exit z=-1.0 死区）+ 最短持仓期 min_hold（0/12/48）+ 无止损止盈，72 组参数扫描（`.optim/p8_cost_aware_report.md`）。
- 向量化模拟（次根开盘/半仓/费滑）：**3 个幸存者（entry=1.5/exit=-1.0/min_hold=0|12|48），BTC 搜索+留出两窗为正**（min_hold=48：search +4.0% / holdout +3.6%），但 ETH/SOL 留出为负（-1.7%~-3.1%）。
- 引擎验证（`.optim/p8b_engine_report.md`，FastCombo 预计算 z + 引擎撮合）：**修正 candles_count 1 基错位后，三个配置在 BTC 搜索 OOS（+5.4%~+7.5%）、BTC 留出（+1.2%~+1.6%）、ETH 留出（+0.2%~+1.2%）、SOL 留出（+0.5%~+0.65%）全部为正**——首次出现“引擎级、跨品种同向”的候选。
- 未达标项：**OOS 交易笔数不足**（BTC 留出仅 11~13 笔 < 30）且**成本 5x 转负**（-5.4%~-8.3%）。方向正确但样本/边际不足，未达 GAP-01 通过线。
- P8c 精调（`.optim/p8c_finetune_report.md`）：降阈值提频（entry 1.0/1.2/1.5）+ min_hold 24/48/96 + size 0.5/1.0，36 组加 **5x 成本过滤 → 0 幸存者**（最佳 5x 留出 ret=-3.1%）。**结论：成本感知执行让因子在基础成本下可交易（P8b 三窗口全正），但边际太薄，扛不住 5 倍成本；离 GAP-01“成本 5x 仍正”还差一档。**
- 代码产出：`strategies/factor_signal.py` 新增 `min_hold_bars` 参数（默认0=兼容旧行为，止损不受限），配套 `tests/test_factor_signal_min_hold.py`（4 例）；全量 pytest exit 0。
- **P9 限价 maker（`.optim/p9_report.md`，2026-09-18）**：h=1 组合因子 + 限价单（maker fee=0.0005/slip=0.0003）+ 滞回（entry=1.5/exit=-1.0）+ min_hold=48 + size=0.5，**首次通过 5x 成本过滤**：BTC 搜索 ret=+10.4%（92 笔）、留出 +5.6%（30 笔）、**5x maker 成本留出 +4.3%**。但 ETH/SOL 留出为负（-1.4%/-2.2%）——**跨品种一致性仍未达标**；h=6/24 因子交易过少（5 笔）不可用。
- **P10 引擎 partial 复核（`.optim/p10_engine_partial_report.md`，2026-09-18）**：同一配置进引擎 `limit_order_model="partial"`（未触及挂单/超时取消，最保守口径）：BTC 搜索 OOS **+0.25%**、留出 **+0.41%**、**5x 成本留出 +0.30%（仍正）**；但限价成交率极低（3 折仅 6 笔）、ETH 留出 -0.62%、SOL +1.12%——**正收益可复现但样本极小、跨品种不稳定**。
- **P11-P14 更长历史 + 等权（2026-09-18）**：BTC/ETH/SOL 1h 回填至 30k 根（2023-04 起）。
  - P12（`.optim/p12_report.md`）：**DRL 权重在更长历史上过拟合暴露**（留出 -5.1%、27 邻域 0 正）；**等权多因子反而全正**（留出 +3.0%、5x +2.2%、ETH/SOL 正，但仅 18 笔）。
  - P13（`.optim/p13_report.md`）：等权 + entry1.2/exit-1.0 扫描出 **7 个幸存者**（搜索/留出均正、≥30 笔、5x 正；最优留出 +15.4%、5x +6.7%），邻域稳健 11/27、跨品种 SOL 负。
  - P14（`.optim/p14_report.md`）：最优配置进引擎 partial 撮合 → **留出转负（-2.7%）、5x -2.9%**——向量化“限价必成交”假设再次被保守口径证伪。
- **P16 长历史重挖 + 多持有周期（`.optim/p16_report.md`，2026-09-18）**：在 30k 根（2023-04 起）上按 h=1/4/12 重新 DRL 挖掘，DRL 权重与等权各跑 144 组（entry/exit/min_hold/size）：
  - h=1 OOS rank_IC=+0.017（valid=False）、h=4 IC=-0.018（valid=False）、**h=12 IC=-0.031/ICIR=1.15（valid=True）**；
  - 但**幸存者 0**：h=1 的候选搜索窗虽 +25%~+76%，留出窗仅 **2~4 笔**（成交样本远不足）；h=4/12 因子在测试阈值下**0 成交**（z 分布随周期变化，阈值未迁移）。
- **策略搜索最终结论（P0-P16）**：现有策略执行器参数空间（P0-P3）、反转执行器（P5）、朴素阈值（P7）、DRL/等权因子 + 限价 maker（P9-P14）、长历史多周期重挖（P16）、跨品种多空（P15）**均未通过 GAP-01 全部门槛**；**跨品种一致性、样本量（≥30 笔）、保守撮合下 5x 成本仍正，三者从未同时满足**。继续在现有执行器/因子库里调参的边际收益已接近零，建议：或换数据/资产类别（资金费率、基差、链上、更高频），或先接受“暂无 edge”把精力放在实盘工程化。

### GAP-02：从未在真实交易所 API 上验证订单

**现状证据**：

- 全仓库 `grep -i "testnet|sandbox|set_sandbox|dryRun"` → **0 处匹配**（`exchange/manager.py` 只有 `enableRateLimit`）。
- 实盘相关测试全部使用假对象：`tests/test_live_order_spec.py` 的 `SpecExchange`、`tests/test_simulated_live_path.py` 的 `SimulatedExchange`、`tests/test_reconcile.py` 的模拟 `ExchangeManager`。
- `tests/test_direction_real.py` 只拉真实 K 线验证方向引擎，**不包含任何下单**。

**证据陷阱**：simulated 模式落库的 Trade 也写 `exchange="live"`（`tests/test_simulated_live_path.py` L116），因此 **DB 里有 `live` 行 ≠ 真实下单过**。

**影响**：REST 下单、限频、错误码、精度、时区、触发单参数、代理链路在真实交易所侧全部是未验证状态。

**验收标准**：

1. `exchange/manager.py` 支持 testnet/sandbox 开关（如 ccxt `set_sandbox_mode(True)` 或独立 testnet 域名），且只读/下单均走 testnet。
2. 新增 testnet 冒烟脚本（可手工运行，不必进 CI）：真实 testnet 上下单→轮询→撤单→复核→余额核对，输出订单 id/filled/status 并断言非空。
3. 对 Binance 与 OKX 各跑一遍（两个不同交易所，验证 ccxt 参数兼容性），覆盖：市价单、限价单、触发式止损单（protective stop 的 `triggerPrice` 参数）。
4. 测试中记录并归档真实网络时延（下单→响应 RTT），与本地 p95 分开统计。

**🚧 执行状态（2026-09-18）**：`exchange/manager.py` 已支持 `testnet=True`（ccxt `set_sandbox_mode`，不支持即显式报错）；已提供冒烟脚本 `.optim/p1_testnet_smoke.py` 与运行器 `.optim/p1b_okx_testnet_run.py`（从 DB 读密钥、不打印明文）。
- ✅ **代理已通**：用户开启 VPN 后本机 7890 端口可用，应用自动探测生效（`resolved_proxy=http://127.0.0.1:7890`）；OKX 公共行情实测 120 根 K 线拉取成功、sandbox 客户端与市场规格读取成功。
- ~~❌ 下单鉴权失败（实盘与模拟盘均失败）：`okx 50119 API key doesn't exist`~~ → **已由用户更换新 key 解决**（旧 key 过期）。
- ✅ **实盘鉴权已通（2026-09-18 复测）**：新 key `42c4****6eab`（len=36）实盘只读 `fetch_balance` 成功，账户非零余额 `{'USDT': 1.0}`；BTC/USDT 实时价 78097、规格 min_amount=1e-05 BTC（最小名义 ≈ **0.78 USDT**）。
- ⚠️ **沙箱环境不匹配**：同一把实盘 key 走 `testnet=True` 返回 `50101 APIKey does not match current environment`（预期：实盘 key 不能用于模拟盘）。**沙箱验证需要单独在 OKX「模拟交易」环境创建 demo key。**
- ⚠️ **实盘下单余额过小**：账户仅 1.0 USDT，接近最小名义 0.78 USDT（加手续费缓冲后基本不可用）——真实下单演练需要先入金或改用模拟盘。
- 待办：①（推荐）创建 OKX 模拟盘 demo key → 运行 `.venv/Scripts/python .optim/p1b_okx_testnet_run.py --place-order` 完成下单→撤单→复核；②或用实盘 key 在最小额下单演练（需用户明确同意 + 入金）。

### GAP-03：下单无幂等、创建失败不重试

**现状证据**（`engine/order_manager.py`）：

- L474-477：`create_order(symbol, otype, side, qty, limit_price)` **未传 clientOrderId**（普通订单；protective stop 才用 `ps_` 前缀）。
- L478 注释：**“简化：市价单直接视为成交”**——真实市价单可能部分成交/排队/拒单。
- L555-557：`create_order` 抛异常 → 宽泛 except 返回 None，**不重试、不登记注册表**（`.optim/report_latency2.md` exp6 只修了轮询 fetch_order 异常；`probe_live_branches.py` 实测 create 失败路径为“0 落库、0 注册表、余额不变”）。

**影响**：网络响应丢失时（订单已在交易所侧生效但本地没收到），下一根 K 线同向信号可能**重复下单**；create 瞬时失败则**信号丢失**，策略状态与交易所状态分叉。

**验收标准**：

1. 普通订单（市价/限价）发送时带稳定 `clientOrderId`（如 `tr_<uuid>`），重试时复用同一 id 保证幂等。
2. `create_order` 网络异常（ccxt.NetworkError/TimeoutError）时：有限次数重试（复用 clientOrderId）→ 仍失败则登记对账注册表，而不是静默丢单。
3. 新增测试：模拟“create 已生效但响应丢失”（SimulatedExchange 记录订单后抛 NetworkError），断言重试/对账后不重复开仓。
4. 市价单“视为成交”简化需显式注释并记录已知偏差；若交易所支持，改为按 `fetch_order` 确认。

**✅ 执行状态（2026-09-18，已闭环）**：`engine/order_manager.py` 普通订单带 `tr_<uuid>` clientOrderId；create 网络异常最多重试 2 次（复用同一 id）；持续失败登记 `intent:` 注册表，对账按 clientOrderId 认领真实订单；`exchange/simulated.py` 透传 clientOrderId 供本地演练。测试：`tests/test_order_idempotency.py`（4 例）。市价单“视为成交”简化保留并注释，待 testnet 实测后再按 `fetch_order` 确认。

### GAP-04：回测/纸面执行模型偏理想

**现状证据**：

- `backtest/engine.py` L51/57-60：`limit_order_model="none"`（默认不做部分成交）、`participation_rate=0.0`（默认不限流动性）、`funding_rate=0.0`（默认无资金费率）、`intrabar_stops=True`。
- `backtest/_matching.py` L239：市价单 = 次根开盘价 ± 固定 slippage（0.0005）。
- `exchange/paper.py` L42 与 `exchange/simulated.py` L120-123：纸面/模拟成交 = `last_price ± slippage`，无 tick、无深度、无排队。

**影响**：纸面/回测的胜率与收益通常系统性好于实盘（滑点、部分成交、流动性冲击、资金费率、延迟都未建模）。没有“纸面 vs 实盘偏差”的实测统计，就无法判断纸面结果有多少可信度。

**验收标准**：

1. 回测默认参数升级为更保守口径（或至少文档明示默认口径的乐观方向）。
2. 用 testnet 成交记录统计真实滑点分布（均值/P95/最大），与 `paper_slippage=0.0005` 对比；偏差显著时调整默认值。
3. 限价单策略必须开启 `limit_order_model="partial"` 回测，并在报告中标注未成交/部分成交比例。
4. 新增“纸面 vs 回测同参数同行情”一致性测试（已有 `test_fill_consistency.py` 基础上扩展）。

**🚧 执行状态（2026-09-18）**：双引擎已在默认乐观口径（limit_order_model=none / participation=0 / funding=0）下输出 WARNING 并指向本文档（`backtest/engine.py`、`backtest/fast_engine.py`）。**待办：testnet 滑点统计回填 `paper_slippage`；限价单策略默认开 `partial` 需先评估基线漂移（有意变更后更新 `docs/REDEV_GUIDE.md` §1）。**

### GAP-05：无紧急手动清仓/人工 override

**现状证据**：

- `engine/trading_engine.py` 仅 `_force_close_stale_position()`（断链自动平仓专用，L857）。
- `web/api/trading.py` 只有 start/stop/status/strategies/params；**无手动下单、撤单、清仓接口**。
- 全仓库 `grep "close_all|flatten|manual_close|紧急|一键平仓"` 无业务实现。

**影响**：live 模式下“停引擎”不会平仓；策略失控/误配置时没有快速离场通道，只能等断链或止损触发。

**验收标准**：

1. 新增 `POST /api/trading/close-all`（或等价接口）：遍历全部持仓，按 live 订单链路逐个市价/限价平仓，返回每笔结果。
2. 前端实盘页有“一键清仓”按钮，二次确认后调用；风控冷却不拦截平仓单（与现有 `closing` 豁免语义一致）。
3. 新增测试：simulated 模式下 close-all 清空持仓、protective stop 全部 release、Trade 落库为 sell。
4. 停引擎接口响应中明确提示“停引擎≠清仓”，避免用户误判。

**✅ 执行状态（2026-09-18，已闭环）**：`engine/trading_engine.close_all_positions()`（三种模式统一走 `_execute_signal`，绕开风控新开仓限制）+ `POST /api/trading/close-all` + 前端仪表盘/实盘页“一键清仓”按钮（二次确认）。停引擎未运行时 close-all 返回 400 提示“停引擎≠清仓”。测试：`tests/test_close_all.py`（2 例）。

### GAP-06：live 引擎单标的

**现状证据**：

- `engine/trading_engine.py` L249：`MarketDataHub(self.exchange_id, [self.symbol], ...)`，引擎只维护单一 symbol 的行情/持仓/风控。
- `web/api/live.py` 的 `LiveConfigIn` 只有单个 `symbol`。
- 多标的能力停留在研究层：`backtest/portfolio.py`、`factors/cross_section.py`、`run.py portfolio`。

**影响**：若目标是实盘跑多标的/横截面因子，需要先扩展 live 引擎：多 symbol 行情订阅、组合持仓、总敞口风控、单标的止损与组合止损的层级。

**验收标准**（仅当路线图包含组合策略时）：

1. live 引擎支持 symbol 列表（KV 配置或 API 传入），hub 同时订阅多个标的。
2. 组合层风控：总仓位/总每日亏损/单标的占比有独立检查。
3. `run.py portfolio` 的组合回测结果可作为多标的策略上线前验证输入。

**🚧 执行状态（2026-09-18）**：
- **P15 跨品种多空研究证据（`.optim/p15_cross_section_report.md`）**：10 个主流币 1h 共同窗口 14176 根，P6 等权多因子做截面多空（low/high × k=1/2，含费用与5x费用），**OOS 净收益全负**（best high_k1 = -5.9e-05/bar，胜率50.8%），累计 -27%~-87%。
- 结论：**多标的多空组合目前也没有可用的因子 edge**；GAP-06 建多标的引擎的价值需等到有组合级 alpha 证据后再投入，否则是给负期望系统加复杂度。建议先解决代理→testnet 实测与基线提交，再回头找组合级因子。

### GAP-07：运维与恢复证据不足

**现状证据**：

- `config/settings.py` L112：`trading_auto_start=False`（默认断电重启后不会自动恢复实盘）。
- `docs/REDEV_GUIDE.md` L41：工作区约 **145 个文件未提交**，无基线快照，任何回退/对比都失去参照。
- `README.md` L67 写“实盘在 AI 设置页填密钥”，实际入口是 `/api/live/mode` + `/api/exchanges`——文档与实现不一致，实盘流程未按正式交付物维护。

**影响**：无人值守实盘的崩溃/断电/重启行为未演练；没有干净基线则无法审计“哪个改动导致实盘事故”。

**验收标准**：

1. 提交当前工作区作为基线快照（commit 全部 145+ 未提交文件），并在 `docs/REDEV_GUIDE.md` §1 更新 commit 号。
2. 配置 `trading_auto_start=true` 后做一次断电恢复演练：live/simulated 模式重启后持仓、FIFO 队列、protective stop、风控状态全部恢复（现有 `_restore_live_position_state` + `restore_state` 已具备，缺的是演练记录）。
3. 修复 README 与实际的 live 配置入口不一致。
4. 建立运行日志分级：交易事件（下单/成交/撤单/对账差异/熔断）与普通 HTTP 访问日志分离（当前 `data/svc_8001.log` 全是轮询 GET，交易事件混在业务日志，可观测性差）。

**✅ 执行状态（2026-09-18，已闭环）**：
- ① **基线快照已提交 `7be56a2`**（23 文件；含 LIVE_READINESS、GAP-03/05 代码与测试）。
- ② **断电恢复演练已完成**：`tests/test_auto_start_restart_drill.py` 用真实 lifespan + `trading_auto_start=true` 跑两段启停，验证重启后自动拉起引擎、FIFO 成本队列从 Trade 回放、风控冷却跨重启仍生效。
- ③ README 实盘入口描述已修正（指向实盘启动页/交易所凭据入口 + testnet 提示）。
- ④ **交易事件日志分级已完成**：`core/logging_config.py` 新增 `data/logs/trading.log`（只收 `engine.*` / `core.bus` 的下单/成交/撤单/对账/熔断事件），与 HTTP 访问日志分离；`tests/test_logging_split.py` 锁住行为（HTTP 日志不得进 trading.log、trader.log 仍保留全量、重复 setup 不叠加 handler）。

---

## 4. 证据索引

| 文件 | 证明了什么 |
|---|---|
| `data/backtest_dual_ma.json` | 双引擎一致性基线：ret=-0.206239、Sharpe=-4.2268、PF=0.4374 |
| `tests/test_stat_significance.py` | 默认策略在近 7 个月真实 1h OOS 收益为负、交易仅 3~12 笔 |
| `tests/test_simulated_live_path.py` | simulated 走 `_place_live` 全状态机；Trade 落 `exchange="live"` 口径 |
| `tests/test_live_order_spec.py` | 实盘 symbol 解析、精度截断、minNotional、protective 挂单参数 |
| `tests/test_live_branch_faults.py` | 部分成交/create 异常/撤单失败/引擎余额四分支演练 |
| `tests/test_combo_live_e2e.py` | combo 部署策略在 simulated 撮合下的买入→卖出闭环 |
| `tests/test_order_idempotency.py` | GAP-03：clientOrderId 幂等 + create 重试 + intent 登记/对账认领（4 例） |
| `tests/test_close_all.py` | GAP-05：一键清仓清空持仓、落 sell Trade、未运行拒绝（2 例） |
| `.optim/p1_testnet_smoke.py` | GAP-02：testnet 冒烟脚本（连通性/余额/市场规格/限价挂撤单，需用户凭据） |
| `.optim/report_latency2.md` | 本地信号→下单 p95 1.151ms（-38.3%）；风险窗口查询 -88% |
| `.optim/log.md` | exp6/exp7 实盘路径可靠性专项：轮询异常兜底、simulated 架构改造 |
| `.optim/ARCHIVE_SUMMARY.md` | 实盘路径可靠性专项总结；下一步建议“实机验证唯一无法脚本化” |
| `.optim/p0_report.md` | P0 实测：6 内置策略在 BTC/ETH/SOL 真实 1h 上的 walk-forward OOS 与成本敏感性 |
| `.optim/p0_grid_pbo.json` | grid 完整 PBO/DSR：BTC/SOL “疑似过拟合”、ETH “严重过拟合”，无一通过 |
| `.optim/p2_search_report.md` | P2 选参→留出验证：93 组参数 0 幸存者；Top 候选（grid）留出窗全部转负 |
| `.optim/p3_search_expand.md` | P3 扩展搜索：5m+反转因子+8 altcoin；幸存者 2 个（BNB grid）均“严重过拟合” |
| `.optim/p4_factor_report.md` | P4 库因子筛选：19 个因子-周期组合通过门，全部为稳定负 IC（反转信号） |
| `.optim/p5_reversal_report.md` | P5 反转执行器验证：正确量纲阈值下搜索窗 0 候选 |
| `.optim/p6_drl_miner_report.json` | P6 DRL 组合因子：OOS rank_IC=+0.0507/ICIR=1.26，质量门 valid |
| `.optim/p7_combo_report.md` | P7 组合因子阈值策略：8 组阈值全部亏损（IC→PnL 落差） |
| `.optim/p8_cost_aware_report.md` | P8 成本感知执行：3 个幸存者（entry=1.5/exit=-1.0/min_hold=0|12|48），BTC 两窗正、ETH/SOL 留出负 |
| `.optim/p8b_engine_report.md` | P8b 引擎验证（FastCombo 预计算 z）：修正偏移后三窗口全正；OOS 交易 11~13 笔（样本不足）、成本 5x 转负 |
| `.optim/p8c_finetune_report.md` | P8c 精调：36 组 + 5x 成本过滤 → 0 幸存者（最佳 5x 留出 -3.1%，边际太薄） |
| `.optim/p9_report.md` | P9 限价 maker：h=1 配置首次过 5x 成本（5x 留出 +4.3%）；跨品种 ETH/SOL 转负 |
| `.optim/p10_engine_partial_report.md` | P10 引擎 partial 复核：5x 成本仍正(+0.30%)但仅 6 笔、ETH 负/SOL 正 |
| `.optim/p12_report.md` | P12 更长历史：DRL 权重过拟合（邻域 0/27）；等权全正但仅18笔 |
| `.optim/p13_report.md` | P13 等权提频：7 个幸存者（≥30笔、5x正；最优留出+15.4%）；邻域 11/27、SOL 负 |
| `.optim/p14_report.md` | P14 引擎 partial：等权最优配置留出转负(-2.7%)，保守口径证伪 |
| `.optim/p15_cross_section_report.md` | P15 跨品种多空：10 币 1h 截面多空 OOS 净收益全负（无组合级 edge） |
| `.optim/p16_report.md` | P16 长历史(h=1/4/12)重挖：h12 因子过 IC 门但 0 成交，144 组全无幸存者 |
| `core/logging_config.py` | GAP-07④：交易事件独立 `trading.log`（engine.*/core.bus），与 HTTP 访问日志分离 |
| `tests/test_logging_split.py` | 2 例：交易事件隔离 + trader.log 不丢信息 + 重复 setup 不叠加 handler |
| `tests/test_auto_start_restart_drill.py` | GAP-07②：真实 lifespan + trading_auto_start 断电恢复演练（自启/FIFO/风控冷却） |
| `strategies/factor_signal.py` | 新增 min_hold_bars 参数（默认0=兼容旧行为）；`tests/test_factor_signal_min_hold.py` 4 例 |
| `engine/order_manager.py` | live 下单全流程；普通订单带 clientOrderId、create 重试、intent 登记、对账认领（GAP-03 已闭环） |
| `engine/trading_engine.py` | live/simulated 启动装配、每日对账、断链平仓、`close_all_positions` 一键清仓、单 symbol 引擎 |
| `exchange/manager.py` | ccxt REST 封装；`testnet=True` 沙箱开关（ccxt set_sandbox_mode） |
| `engine/risk.py` | 风控规则、冷却/人工确认、闪崩/量比/24h 保护、状态持久化 |
| `backtest/_matching.py` | 撮合单一事实来源：次根开盘±滑点、限价 partial 模型、funding |
| `backtest/engine.py` | 回测默认配置：limit_order_model=none、participation=0、funding=0 |
| `exchange/paper.py` / `exchange/simulated.py` | 纸面/模拟成交 = last_price ± slippage（无深度/tick） |
| `exchange/manager.py` | ccxt REST 封装；无 testnet/sandbox 支持 |
| `web/api/trading.py` | 只有 start/stop/策略/参数，无手动清仓接口 |
| `web/api/live.py` | 模式切换（paper/simulated/live）、单 symbol 实盘配置 |
| `config/settings.py` | trading_auto_start=false、max_stale_seconds、protective stop 参数 |
| `docs/REDEV_GUIDE.md` | 579 tests 基线、红线清单、145 未提交文件警告、验证链 |

---

## 5. 推荐路线图

按“先证明能赚钱 → 再证明能真实下单 → 再补安全 → 最后无人值守”的顺序：

| 阶段 | 内容 | 通过条件 |
|---|---|---|
| **P0 补证据** | 用真实数据验证现有 6 个内置策略的 OOS 表现；跑 cost-scan 成本敏感性 | GAP-01 验收 1-4 全过 |
| **P1 沙箱验证** | exchange/manager 加 testnet/sandbox；Binance + OKX testnet 下单/撤单/对账/触发单脚本 | GAP-02 验收 1-4 全过 |
| **P2 资金安全** | clientOrderId 幂等 + create 重试/登记；close-all API + 前端按钮 | GAP-03、GAP-05 验收全过 |
| **P3 执行模型** | 默认回测口径保守化；testnet 滑点统计回填 paper_slippage；限价 partial 默认开启 | GAP-04 验收全过 |
| **P4 运维** | commit 基线快照；trading_auto_start 演练；README 修正；日志分级 | GAP-07 验收全过 |
| **P5（可选）多标的** | live 引擎多 symbol + 组合风控 | GAP-06 验收全过 |

每阶段完成必须满足 `docs/REDEV_GUIDE.md` §4 验证链（compileall / pytest / check_js / 双引擎一致性），并更新本文档 checklist。

---

## 6. 常见证据陷阱（复评时必读）

1. **DB 里 `exchange="live"` ≠ 真实下单**：simulated 模式也落这个口径（`tests/test_simulated_live_path.py` L116）。
2. **本机 p95=1.151ms ≠ 实盘端到端时延**：该值不含交易所网络 RTT、限频、排队。
3. **demo 数据回测为正 ≠ 真实 edge**：合成数据与真实行情的分布/成本完全不同。
4. **579 tests 全绿 ≠ 实盘就绪**：测试覆盖的是“本地状态机正确”，不是“真实交易所行为正确”。
5. **protective stop 已挂 ≠ 已成交**：触发单是服务端兜底，仍需对账确认成交与 FIFO 记账。

---

## 7. 更新约定

- 每次实盘能力有实质变化（新增 testnet 验证、幂等、close-all、多标的等），在 §2.2/§3 对应 GAP 后追加一行“状态：✅ 已闭环 / 🚧 进行中 / ❌ 未开始”，并更新 §5 checklist。
- 本文档由 `docs/REDEV_GUIDE.md` 文档地图引用为「实盘就绪度评估」；与 REDEV_GUIDE 冲突时以 REDEV_GUIDE 为准（当前基线/红线）。
