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
- ✅ **AI 通道已可用（2026-09-18 更正）**：软件内保存的是**自定义兼容接口**（`provider=custom`、`base_url=https://api.agnes-ai.cn/v1`、`model=agnes-2.5-flash`），实测 `POST /api/ai/test` 返回「连接成功，模型回复: 正常」、`/models` 返回 agnes-2.0/2.5/3.0 系列。
  - 之前的「AI key 401」是**误判**：我用 `config.settings` 里的默认 `ai_base_url`（api.openai.com）去测，而没走 DB 里保存的自定义地址。
  - 🔴 **同时修掉一个真实前端 bug**：AI 设置页会把**已保存的自定义模型覆盖成默认 gpt**——两处元凶：① 提供商 watch 在程序化加载阶段回填默认模型（`gpt-4o-mini`）；② 「获取模型列表」时若保存的模型不在服务商返回列表里，直接替换成列表第一个（Node 复刻验证：场景 A 旧逻辑 `agnes-2.5-flash → gpt-4o-mini`，修复后保持 `agnes-2.5-flash`）。已加守卫 `aiSettingsLoaded` + 非破坏性回填，并加回归测试。

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
- **P18-P20 换数据源：资金费率（funding rate）实测（2026-09-18）**：
  - P18（`.optim/p18_funding_fetch.py`）：从 Binance USDT-M 永续拉取 BTC/ETH/SOL 资金费率全历史，各 **3800 条（2023-04-01 ~ 2026-09-18，8h 一根）**，与 30k 根 1h K 线全段对齐；数据存 `.optim/funding/*.csv`。（修掉一个分页方向 bug：`since` 翻页应取 `batch[-1]` 往后，误用 `batch[0]` 只拿到第一页 1000 条。）
  - P19（`.optim/p19_funding_factor_report.md`）：费率水平/90天z/变化/极端分位 4 因子 × 3 标的 × h=1/6/24，36 项检查**仅 2 项过质量门且方向相反**（ETH f_level h24 IC=-0.063 vs SOL +0.045）。
  - P20（`.optim/p20_funding_8h_report.md`）：排除 ffill 台阶稀释，改在 **8h 结算时点**采样、标签取 8h/24h 收益，8 个组合 **0 个通过**（无不满足跨品种同号）。
  - **关键观察**：ETH 与 SOL 的符号**系统性相反**（ETH：费率越高→未来跌，符合经典反转逻辑；SOL：费率越高→未来涨，样本期大涨特征），说明该信号不是稳定的市场结构；全部组合 ICIR 多为负或近零（IC 时序方向不稳）。
  - **结论**：资金费率在所测样本/口径下**不构成跨品种稳健的 alpha**，不足以支撑实盘；若继续这条线，应转向**资金费率套利（赚费率本身，而非预测价格）**或引入**基差/持仓量**等组合信息，而不是单独用费率预测方向。
- **P21-P22 资金费率套利（delta 中性：现货多 + 永续空，赚费率本身）实测（2026-09-18）**：
  - P21（`.optim/p21_funding_arb_report.md`，抓取永续 8h K 线存 `.optim/perp/*.csv`）：全样本（2023-04~2026-09，3800 个结算点）下，最优参数 **entry=1e-4（0.01%/8h）· exit=0** 时：
    - BTC 年化 **+2.54%**、回撤 **-0.38%**、夏普 7.36、开仓占比 21%、**6 笔交易**；**5x 成本后仍 +1.17%** ✅
    - ETH 年化 **+3.04%**、回撤 -0.37%、夏普 8.64、**5x 后仍 +1.68%** ✅
    - 这是 P0–P21 以来**首次同时满足「正期望 + 抗 5x 成本」**的结果；机理：只在高费率开仓（避低费率期），交易次数从 515 降到 6，成本被摊薄。
  - P22 稳健性检验（`.optim/p22_funding_robust_report.md`）**推翻了它的可用性**：
    - **费率逐年衰减**：BTC 8h 费率均值 0.0104% → 0.0070% → **0.0037%**（三年跌 65%，市场成熟/套利资金涌入）；
    - **近 18 个月（2025-04 至今）失效**：费率均值 0.0037% 低于 entry 阈值 0.01% → **策略不触发开仓，年化 0.00%、0 笔交易**；SOL 更糟（费率转负 -0.023%）；
    - **5x 成本下只有第一年为正**：BTC/ETH 分年 2/3 正、但 5x 仅 1/3 正；邻域（基础+5x 均正）BTC/ETH 6/10、**SOL 0/10**。
  - **结论**：资金费率套利是**真实但已被抹平的 edge**——2023-2024 年确能年化 6~7%（5x 成本后 +5.6%），但到 2025-2026 年费率已低于成本线。无风险收益必然被套利至接近资金成本，**当前不具备可操作性**。
  - **对软件的意义**：这条线走通的价值不在收益，而在**验证了「新数据源 → 因子/策略 → 稳健性检验」这条流水线可用**（P18 抓取 → P19/P20 因子筛查 → P21 策略建模 → P22 稳健性证伪，全程可复现、有报告）。
- **P23 基差（basis = 永续−现货）因子实测（2026-09-18）**：本地 1h 现货重采样 8h + P21 抓取的永续 8h，3750 个 8h 点，8 组因子×周期检查 **0 通过**：
  - 方向一致（多数组合三标同号，均为正）但 **|rank_ic| 仅 0.004~0.042（门槛 0.03）**，ICIR 普遍为负（SOL -0.24~-0.45，方向不稳定）；
  - **机制诊断**：基差与资金费率相关性 **0.57~0.68** → **基差基本是费率的影子**，不是独立新信息，与 P19/P20 结论自然一致。
  - ⚠️ **过程中抓到并修复一次前视泄漏**：初版现货取 1h 的 08:00 close（覆盖 08:00~09:00）去比永续 8h 的 08:00 close（覆盖 08:00~16:00），时间语义错位 → "基差"混入未来 7 小时涨跌，**rank_ic 假高到 0.91**；现货改为重采样 8h 后基差回到真实量级（均值 -0.029%、std 0.039%、正基差占比 16%，修复前是 ±1.9% 的鬼数字）。
  - **经验固化：|rank_ic| > 0.2 一律先当泄漏排查，不当发现**（P15 跨品种多空已犯过一次，本次为第二次，已在脚本写入注释警告）。
- **P24-P25 链上数据（blockchain.info，与价格/合约情绪正交的维度）实测（2026-09-18）**：
  - P24（`.optim/p24_onchain_fetch.py`）：抓取 6 个 BTC 链上指标（活跃地址 / 交易数 / 链上转账额 / 价格 / 算力 / 手续费），日频、历史至 2009（各 5800~6500 点），存 `.optim/onchain/*.csv`。
  - P25（`.optim/p25_onchain_report.md`）：主窗口 2018 起 3160 天，7 因子 × h=1/3/7 天 = 21 项检查，**5 项过质量门**：`tx_z` h7 **rank_ic=-0.080**（P0–P25 以来最大 IC）、fee_z h3 -0.048、hash_z h1 +0.043、addr_mom30 h1 -0.039、nvt_z h7 -0.035。
  - ❌ **但分年稳定性检验彻底否决**：7 个因子**没有任何一个各年同号**。`tx_z` 逐年 IC 从 2010 到 2026 正负交替（+0.03/+0.16/-0.10/+0.19/-0.05/…/-0.18/-0.06/-0.18/+0.01），整体 -0.080 只是 17 年正负相抵后的残值，非可交易信号；全部因子 ICIR 仅 ±0.11~0.19（本身就指示方向不稳）。
  - **结论**：链上因子**不通过**，否决得比费率/基差更彻底（分年检验直接否掉）。
  - **但留下一条有价值线索**：`tx_z` 的 |IC|=0.08 为 P0–P25 最大，说明**链上数据确实携带信息、只是方向随市场周期翻转**（牛市链上活跃=看涨，熊市=看跌）。可能的改进方向是**按周期状态分域（regime-conditional）**而非全样本单一方向——留待后续（P26 候选）。
- **P26-P27 regime-conditional 分域（针对"方向不稳定"根因的正面攻击）实测（2026-09-18）**：
  - P26（`.optim/p26_regime_report.md`）：用 MA200 趋势 + 30 日波动率分位把市场分成 4 个域（bull/bear × highvol/lowvol），域内检验 7 个链上因子 × h=1/3/7 天。**4 项通过全部判据**（域内 |IC|≥0.03、|ICIR|≥0.1、**域内分年同号**），最强项 `bear_highvol` 域内 `addr_px_z` h=7d **IC=+0.146、ICIR=+0.97**（P0–P26 最大且最稳）；`addr_px_z` 在 bull_lowvol / bull_highvol / bear_highvol **三个域均同号为正**。
  - P27 稳健性检验（`.optim/p27_addrpx_robust.md`）**判定 P26 为选择偏差，非真实 alpha**：
    - **时间外推失败**：全样本（不分域）IS 2018-2022 的 IC 为 **-0.037/-0.053/-0.062（h=5/7/10，全负）**，OOS 2023-2026 变为 **+0.019/+0.013/+0.002（全正）** → **方向翻转**，且 OOS |IC| 仅 0.01~0.02，**低于 0.03 门槛**；
    - **域内"分年同号"经不起推敲**：互补域（bull_highvol 与 bear_highvol）**同时**满足分年同号——因为同一年份的样本会随行情被分到不同域，域切换吸收掉了年份效应；
    - `bear_highvol` 的强度集中在 2021(+0.273)/2022(+0.183)，近两年仅 0.049/0.032（贴近门槛）；
    - 邻域检验看似 100/108 组"IS 与 OOS 同号"，但那是**域筛选本身**造成的（域内 IS 为正 → 域内 OOS 也为正），不能作为稳定性证据。
  - **结论**：**分域并没有解决"方向不稳定"，只是把选择偏差包装成了稳定性**。与 P13→P14（向量化 vs 引擎）、P21→P22（套利衰减）同一模式：**看起来很美、严格检验后不成立**。
  - **方法论价值**：P27 的「IS/OOS 时间外推 + 分年拆解 + 邻域敏感性」这套检验有效识别出了 P26 的假信号，应固化为新信号的必检项。
- **P28 期权隐含波动率（Deribit DVOL）实测（2026-09-18）**：新维度（前瞻性风险定价），DVOL 全历史 1997 点（2021-03 起，分页抓取），与本地日线对齐后 BTC/ETH 各 1243 天；6 因子 × h=1/3/7 天 = 36 项检查，门槛为**四关**（质量门 + 跨品种同号 + 分年同号≥4年 + **时间外推**）：
  - 形式通过 1 项：`BTC iv_level h=3d` IC=+0.039、**ICIR=+1.92**、分年同号（2023:+0.031/2024:+0.018/2025:+0.032/2026:+0.093）、时间外推 IS +0.031 → **OOS +0.081**（同号且 OOS 更强）。
  - ❌ **但判定为不通过**：① 仅 1/36 项通过（多重检验下不足为奇）；② **ETH 时间外推失败**（h=7d IS +0.174 → **OOS -0.007**）→ 跨品种稳健性不成立；③ 同一因子在 BTC 上**符号随周期翻转**（h=1d **-0.018**、h=3d +0.039、h=7d +0.061），非稳定经济关系；④ 换手率 0.0（DVOL 日频变化极慢），有效 IC 样本仅 373。
  - **结论**：DVOL 不通过。**同时验证了 P27 固化的时间外推检验有效**——它当场拦下 ETH 的假信号（IS +0.174 看似极强，OOS -0.007）。
- **P29-P31 LLM 提假设闭环（agnes-2.5-flash）实测（2026-09-18）**：把「AI 通道」用于**假设生成**（LLM 提经济逻辑 + 可执行表达式，脚本自动验证），这是 P0–P28 唯一没试过的**方法**。
  - 闭环搭建（`.optim/p29a_datasets.py` 统一数据集 → `.optim/p29b_llm_loop.py` P29 → `.optim/p31_llm_loop_v2.py` P31）：LLM 输出 JSON 假设 → `factors.mining.FactorExecutor` 沙箱执行 → 四关检验（质量门 + 分年同号 + **时间外推** + 样本）→ 把结果与"已提过的表达式"回灌做下一轮。
  - 结果（累计 **98 个假设**）：
    - P29（18 个）：四关通过 2 项，但**是同一表达式重复提交**（`rank(volume) - rank(ma(volume, 20))`），且 ICIR 仅 -0.16；
    - P30 严格复核该式：**BTC 5 折第 3 折变正号**（方向不稳）、**与量比相关 0.81**（只是量比换皮）、**交易级年化 -16.7%/回撤 -58%/胜率 19%** → 否决；
    - P31（40 + 40 个）：四关通过 **0**、严格复核通过 **0**；其中**30/40 因"幻觉列名"无法执行**（LLM 自造 `funding_BTC`、`dvol_BTC`、`close[20]_ETH`、`tx_ETH` 等不存在的列，尽管 prompt 已给出完整列清单）。
  - **结论**：**LLM 提假设在当前 prompt 设计下命中率为 0**，且 75% 的假设因自造列名不可执行。瓶颈不在"想不到逻辑"，而在**LLM 无法稳定输出合规、可执行的因子定义**；更根本地，**无数据反馈的凭空假设与人类拍脑袋等价**——而 P0–P28 已证明这类假设命中率极低。
  - **副产物**：把「新信号必检项」全部工程化进闭环（四关 + 跨品种 + 5 折 + 交易级），后续任何新因子都可用同一脚本一键证伪。
- **P33 5m 长历史复核（2026-09-19）**：针对 P3 原先仅约 32 天 5m 样本的证据缺口，补拉 Binance 现货 5m **每标的 220,000 根**（2024-01-01~2026-02-02，约 2.1 年），复用 P3 的 96 组网格 + 前70%搜索/后30%留出 + walk-forward + 成本扫描：
  - BTC：搜索候选 12 个，但完整留出窗口净收益**全部为负**（例如最佳候选 -1.08%，年化 -1.71%；全量最佳也不超过正收益）。
  - ETH：搜索候选 14 个，留出候选 4 个；完整留出净收益仅 **+0.0069%~+0.0137%**，年化 **+0.0109%~+0.0219%**，41 笔交易，属于噪声/摩擦范围，远非可上线 alpha。
  - SOL：搜索候选 22 个，完整留出幸存者 **0 个**。
  - **三标的无共同通过项**；BTC 全负、SOL 0 幸存者、ETH 仅微小正值且未通过 5x 成本/引擎级证据门。故 P33 把 P3 的“样本不足”补成了可信的否定：**5m 技术指标网格也没有实盘 edge**。报告：`.optim/p33_search_5m_report.md` / `.json`。
- **P34 订单流（Taker 主动买卖）实测（2026-09-19）**：P0–P33 唯一未试的免费正交维度（真实成交方向）。利用 Binance K 线自带 taker_buy_quote 字段可持续翻页（不受 aggTrades 数天限制），拉 1h BTC/ETH/SOL 自 2024-01 起（各约 2.3 万根），因子 taker_ratio / taker_z90 / taker_chg × h=1/6/24：
  - **27 项检查 0 通过**：IC 方向跨品种不统一（BTC h24 +0.015 / ETH -0.002 / SOL +0.029）；ICIR 几乎全负（-0.12~-0.56，IC 时序方向不稳）；SOL h24 IS -0.018 → OOS +0.033 **翻转**；最强 SOL taker_ratio h24 = +0.029 仍低于 0.03 门槛。
  - **结论：订单流维度同样无 stable alpha**。至此 P0–P34 已覆盖技术指标(1h/5m长历史)、资金费率(方向+套利)、基差、链上、分域、DVOL、LLM 假设、订单流——**没有任何维度通过严格实盘验证门槛**。报告：`.optim/p34_taker_report.md` / `.json`。
  - **工程侧结论**：免费/公开数据维度已系统性挖尽；剩余候选（做市/订单簿真钱实验、付费链上实体净流）均需额外资金或授权，属另一类投入。

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
- ⚠️ **实盘下单余额过小**：账户仅 1.0 USDT，接近最小名义 0.78 USDT（加手续费缓冲后基本不可用）。
- ✅ **实盘最小额演练已完成（2026-09-18，用户明确授权；现货无强平风险）**，脚本 `.optim/p17_okx_live_drill.py`，报告 `.optim/p17_okx_live_drill.md`：
  - **订单生命周期全部打通**（真实 OKX）：限价挂单 RTT **409ms** → 查询 `status=open`、**clientOrderId 原样回传** → 撤单 233ms → 复核 `status=canceled`；最小额买入成交（0.00001 BTC @ 78052.3，RTT 218ms）；**兜底止损挂单成功**（触发价 77271.78 = 入场 −1%，SL=TP/2，RTT 499ms）。
  - 🔴 **实盘暴露 bug #1（已修复）**：OKX `clOrdId` 只允许 1–32 位**字母数字**，而 GAP-03 新增的幂等 ID 是 `tr_<hex>`、止损单是 `ps_<...>`（含下划线）→ OKX 直接拒单 `51000 Parameter clOrdId error`，**会导致 OKX 实盘全部下单失败**。修复：改为 `tr`+24 位 hex、`ps`+时间戳（纯字母数字，≤32）；测试同步加"必须 `isalnum()` 且 ≤32"断言。
  - 🔴 **实盘暴露 bug #2（已修复）**：**OKX 条件单（兜底止损）不出现在 ccxt 统一 `fetch_open_orders` 里**（实测挂单成功但 `fetch_open_orders` 返回 0 条，须查 `orders-algo-pending`）→ 重启收养 `ProtectiveStopManager.adopt()` 与对账 `_reconcile_once()` 会误判"止损不存在"，导致**重复挂单/裸奔窗口**。修复：`ExchangeManager.fetch_algo_open_orders()`（OKX trigger/conditional/oco，其他交易所安全返回 `[]`）+ `adopt()` 合并算法单；`tests/test_exchange_manager_algo.py`、`tests/test_protective_stop.py::test_adopt_sees_okx_algo_orders` 锁住。
  - ⚠️→✅ **运营发现（已加固）**：按最小额买入后，手续费以 base 扣除 → 持仓 0.00000999 BTC **低于 min_amount 0.00001**，市价平仓被拒 `51020`（**粉尘仓卖不掉**）。本次残留 dust ≈0.78 USDT 保留在账户（可用 OKX「小额资产兑换」清理）。代码侧加固：① `_fit_live_qty` 卖出前做粉尘检查（`qty < min_amount` → ERROR 日志并跳过，不再每根K线打无效下单）；② `close_all_positions` 清仓后**复核持仓是否真的减少**，未减少则返回 `ok=False` + `manual_close_all_not_filled` 事件（绝不谎报已离场）。测试：`tests/test_live_order_spec.py`（粉尘跳过/正常卖出）、`tests/test_close_all.py::test_close_all_reports_when_sell_not_filled`。
- 待办：若要在**沙箱**复跑，需创建 OKX 模拟盘 demo key（实盘 key 走 sandbox 返回 50101）。

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

**✅ 执行状态（2026-09-18，已闭环）**：`engine/order_manager.py` 普通订单带 `tr`+24位hex clientOrderId；create 网络异常最多重试 2 次（复用同一 id）；持续失败登记 `intent:` 注册表，对账按 clientOrderId 认领真实订单；`exchange/simulated.py` 透传 clientOrderId 供本地演练。测试：`tests/test_order_idempotency.py`（4 例）。
- 🔴 **实盘修正（OKX 2026-09-18）**：幂等 ID 必须**纯字母数字**（原 `tr_`/`ps_` 含下划线被 OKX 拒 `51000 Parameter clOrdId error`，会导致 OKX 实盘全部下单失败）→ 已改为 `tr`+hex / `ps`+时间戳，并加 `isalnum()且≤32` 断言（`tests/test_order_idempotency.py`、`tests/test_live_order_spec.py`、`tests/test_live_branch_faults.py`）。
- 🔴 **粉尘仓守卫**：卖出量本身低于交易所 `min_amount` 时跳过并报错（OKX 实测 51020：手续费扣 base 后持仓 0.00000999 < 0.00001 无法平仓），避免每根 K 线重复无效下单。

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
- 🔴 **实盘加固（2026-09-18）**：清仓后**复核持仓是否真的减少**——若市价卖单被跳过（粉尘仓低于 `min_amount`）或被拒，返回 `ok=False` + 明确提示，并发布 `manual_close_all_not_filled` 事件；**绝不报 ok=True 让用户误以为已离场**（实盘 OKX 粉尘仓 51020 教训）。

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

**✅ 执行状态（2026-09-18，多标的引擎最小可用版已完成）**：
- **引擎支持 symbol 列表（验收 1 ✅）**：`live_trading_config.symbols`（逗号分隔或 list，缺省=单标的，完全向后兼容）；`TradingEngine` 增加 `self.symbols` + `self.strategies`（每 symbol 独立策略实例），hub 用 `self.symbols` 列表多订阅（hub 本就按 `(symbol,timeframe)` 键控，零改动）；`_on_candle` 按 symbol 分发到各自策略实例、`_pending_signals` 按 symbol 存（旧单值 `_pending_signal` 以 property 兼容，单标的路径与测试零改动）；`select_strategy`/`update_strategy_params` 同步应用到所有实例。
- **组合层风控（验收 2 ✅）**：`_check_portfolio_risk()` 每 15s 巡检（复用 portfolio.snapshot 结果，不重复打交易所），总仓位 >95% 与单标的占比 >60% 产生 CRITICAL/WARNING 告警（monitor 出口）。
- **P15 结论仍成立**：多标的多空组合无 alpha → 该引擎是**能力储备**，不是上线依据；工程上验证「多 symbol 订阅/独立策略/组合风控」可行且单标的完全兼容。
- 测试：`tests/test_multi_symbol_engine.py`（3 例：独立策略路由/单标的兼容/组合风控告警）+ `tests/test_live_config_symbols.py`（3 例：逗号字符串规范化/双形态往返/空=单标的）；全量 pytest exit 0。
- **API 与前端入口已补（2026-09-18）**：`web/api/live.py` 的 `LiveConfigIn.symbols`（接受逗号分隔字符串或 list，保存规范化为 list）+ GET/PUT 往返 + 前端实盘配置表单（端到端验证：PUT list/字符串 → GET 回读 → 引擎 `_load_live_config` 加载出 symbols 列表与独立策略实例）。
- **前端入口已收敛为单一「交易对（可多选）」（2026-09-21）**：原先「交易对」（单值 `symbol`）与「多标的（可选）」（`symbols` 列表）两个字段功能重叠——引擎 `_load_live_config` 会把主标的插到列表首位，两个字段给不同值时语义含糊。现已删除单值输入，改为**沿用「持续进化 · 训练标的池」同构的多选二级菜单**（14 个预设 + 手输，至少保留 1 项）。前端落库时 `symbols` = 池列表、`symbol` = 池首项（主标的），**后端与 KV 契约零改动**，历史单标的配置在加载时回退进池（1 项 = 原单标的语义）。
- 待办：验收 3（`run.py portfolio` 回测作为上线前验证输入）——等有组合级 alpha 证据后再接。

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
- ⑤ **统一告警中心已完成（2026-09-18，GAP-07 追加项）**：新增 `engine/monitor.py`——把散落的异常汇聚成分级告警，解决"没人看日志 = 没有告警"。
  - **巡检项**：行情断流（hub 无新K线）、**持仓无兜底止损**（实盘最危险的静默状态）、风控冷却中、事件总线 handler 连续失败；另有下单响应丢失（order_manager intent 登记时触发）。
  - **出口**：`data/logs/alerts.log`（结构化 JSON，运维可采集）+ SYSTEM 事件（前端可见）+ `GET /api/trading/alerts` 与 `status.alerts`。
  - **工程性质**：同 (kind,key) 在 5 分钟去重窗口内只报一次（抑制次数累计可见，防刷屏）；巡检自身异常一律不抛出（不影响引擎周期循环）。
  - **验证**：`tests/test_monitor_alerts.py`（5 例：去重/落盘/四类巡检/不抛异常）；**真实运行验证**——启动引擎（simulated, ETH/USDT 5m）后行情未接入，自动产生 `{"level":"CRITICAL","kind":"market_stale","message":"行情断流：ETH/USDT 5m 已 ∞ 秒无新K线"}` 并同时出现在 `alerts.log` 与 `/api/trading/alerts`。
- ⑥ **异常场景故障注入演练已完成（2026-09-18，`tests/test_fault_injection_drill.py`，4 例全绿）**：验证「异常发生 → 告警产生 → 系统安全降级/自愈」全链路（全部本地 simulated + 假行情，不依赖网络，没开 VPN 也能跑）：
  - **S1 行情断连**：hub 断流 → `market_stale` CRITICAL 告警 + `status.market_stale=True`（引擎 portfolio_loop 每 15s 自动巡检也能触发）；
  - **S2 下单响应丢失**：create_order 连续 3 次超时（引擎重试 2 次上限）→ `order_create_lost` 告警 + `intent:` 登记待对账认领；恢复后重下单正常成交；
  - **S3 对账差异**：本地注册表有、交易所 open 列表未见且订单仍 open → **新增 `order_reconcile_missing` CRITICAL 告警**（此前只写日志不告警，本次补线）+ 注册表保留下轮复查；
  - **S4 迟到成交回灌**：撤单竞态下实际已成交 → **新增 `order_reconcile_late_fill` WARNING 告警** + Trade 回灌 + 注册表清理。
- ⑦ **运维看门狗（2026-09-19）**：解决"REST 进程活着但引擎主循环假死（阻塞交易所请求/死锁）无人察觉"的盲区：
  - **引擎心跳**：`_portfolio_loop` 每 15s 打点 `_heartbeat_ts`，`status()` 暴露 `heartbeat_age_sec` + `engine_loop_alive`（心跳停滞 >60s 且引擎 running → 假死；未运行不误报）；
  - **运维总览** `GET /api/ops/overview`（`web/api/ops.py`）：一次拿全引擎活性/告警计数/行情新鲜度/日志与 DB 大小，聚合 `health` 判定（CRITICAL 告警 OR 断流 OR 假死 → false）；
  - **独立探针** `scripts/watchdog.py`：退出码 0=健康 / 1=服务不可达 / 2=引擎假死或断流或 CRITICAL（供 supervisor/systemd/cron 轮询）；
  - 验证：`tests/test_ops_monitor.py`（5 例：心跳新鲜/老化假死/停止不误报/overview 聚合/CRITICAL→health=false）+ 真实运行 watchdog 返回 exit 0 + 假死场景 loop_alive=False 识别正确。
- ⑧ **数据库备份 + SQLite 运维（2026-09-21，P0-1）**：`scripts/db_backup.py`——解决"库损坏/误删后无备份可恢复"的最坏情况：
  - **在线备份 API**（`source.backup()`）而非文件复制：WAL 模式下文件复制会丢未 checkpoint 增量，恢复出来是残缺库；
  - 备份 trader.db + klines.db → `data/backups/`，备份后立即 `PRAGMA integrity_check` 验证，坏备份自动删除；
  - **恢复演练 `--verify`**：最新备份恢复到临时库，integrity + 关键表行数与源库对比（实证可恢复：trader kv=25/equity_snapshots=10802、klines=418398 完全一致）；
  - 保留策略 `--keep-days`（默认 7 天）自动清理；`--vacuum` 备份前整理碎片（实测 trader 3344→3264KB、klines 34072→31060KB）；
  - 测试 `tests/test_db_backup.py`（4 例）含 **WAL 未 checkpoint 脏数据一致性验证**（连接保持打开时备份仍含增量行）。
- ⑨ **进程守护（2026-09-21，P0-2）**：`scripts/service_supervisor.py`——把看门狗从"只报警"升级为"能拉起"：
  - 职责分工：`watchdog.py` = 只读探针（外部 cron/supervisor 用）；`service_supervisor.py` = 常驻守护（启动子进程 + 周期探活 + 失败自动重启）；
  - 连续探活失败 ≥ max_fail → 杀子进程并重启；子进程自行崩溃退出 → 自动拉起；事件写 `data/logs/supervisor.log`（JSON 行）；
  - 复用 watchdog.check_once 的 `/api/ops/overview` 健康判定（含引擎心跳假死检测），不重复实现；
  - 验证：`tests/test_service_supervisor.py`（4 例：子进程启停/日志 JSON/不可达返回码/重启逻辑）；**真实进程级冒烟**——守护立即退出的子进程，事件序列 `started → probe_fail → restart_due_probe → started` 自动重启闭环。
- ⑩ **日志轮转（2026-09-22，P0-3）**：`core/logging_config.py` 新增 `uvicorn_log_config()`——server.log 从"stdout 直写"改为 RotatingFileHandler（5MB×5）：
  - 此前 trader.log/trading.log 已有轮转，但 uvicorn 的 access/error 日志默认只进 stdout（被 supervisor 重定向成 server.log 直写、无轮转、还可能双写）；
  - `run.py` 的 uvicorn.run 传 `log_config=uvicorn_log_config()`；supervisor 的 stdout 改 DEVNULL（uvicorn 日志已由 file handler 接管，避免双写同一文件）；
  - `disable_existing_loggers=False`：不重置应用 root handlers（trader.log/trading.log 保持 setup_logging 配置）；
  - 验证：`tests/test_log_rotation.py`（3 例：配置结构/RotatingFileHandler/不动 root handler/run.py 接入）；**真实启动**——web 服务 200 后 server.log 含 `uvicorn.access` 行且由 file handler 写入。

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
| `.optim/p17_okx_live_drill.md` | P17 OKX 实盘最小额演练：订单生命周期打通（clientOrderId 回传/撤单/复核）+ 实盘暴露 2 个 bug |
| `exchange/manager.py`（fetch_algo_open_orders） | 修复 bug#2：OKX 条件单可见性（统一 open orders 不含算法单） |
| `tests/test_exchange_manager_algo.py` | 算法单解析 + 非 OKX 安全降级 |
| `tests/test_protective_stop.py::test_adopt_sees_okx_algo_orders` | 重启收养必须能看到 OKX 算法单（防重复挂单） |
| `web/static/index.html`（AI 设置） | 修掉"已保存自定义模型被覆盖成 gpt"：加载守卫 `aiSettingsLoaded` + 非破坏性回填 |
| `tests/test_ai_settings_persist.py` | 3 例：后端回显 KV 值/保存读回/前端不再破坏性覆盖 |
| `.optim/funding/*.csv` | P18 资金费率数据：BTC/ETH/SOL 各 3800 条（8h，2023-04~2026-09） |
| `.optim/p19_funding_factor_report.md` | P19 资金费率因子（1h ffill）：36 项仅 2 项过门且方向相反 |
| `.optim/p20_funding_8h_report.md` | P20 8h 结算时点评估：8 个组合 0 通过（ETH/SOL 符号系统性相反） |
| `.optim/p21_funding_arb_report.md` | P21 资金费率套利：BTC/ETH 全样本 5x 成本仍正（+1.17%/+1.68%），回撤 -0.38% |
| `.optim/p22_funding_robust_report.md` | P22 稳健性检验：费率三年衰减 65%，近 18 个月失效（0 交易），分年 5x 仅 1/3 正 |
| `.optim/p23_basis_report.md` | P23 基差因子：8 组 0 通过（IC 仅 0.004~0.042）；与费率相关 0.57~0.68 说明只是费率影子 |
| `.optim/onchain/*.csv` | P24 链上数据：6 指标日频，历史至 2009（各 5800~6500 点） |
| `.optim/p25_onchain_report.md` | P25 链上因子：5 项过质量门（tx_z h7 IC=-0.080 为最大），但分年稳定性 0/7 否决 |
| `.optim/p26_regime_report.md` | P26 分域检验：4 项通过（bear_highvol addr_px_z IC=+0.146/ICIR=+0.97） |
| `.optim/p27_addrpx_robust.md` | P27 证伪 P26：IS IC 全负 → OOS 全正（方向翻转、OOS |IC|仅 0.01~0.02）= 选择偏差 |
| `.optim/p28_dvol_report.md` | P28 DVOL 期权波动率：36 项仅 1 项形式过四关（BTC iv_level h3 ICIR=1.92），但 ETH 时间外推失败 → 不通过 |
| `.optim/p29_llm_hyp_report.md` | P29 LLM 提假设（18 个）：四关通过 2 项（同一表达式重复） |
| `.optim/p30_voldiv_report.md` | P30 严格复核该式：BTC 5折第3折变号、与量比相关 0.81、交易级年化 -16.7% → 否决 |
| `.optim/p31_llm_loop_v2_report.md` | P31 闭环 v2（80 个）：0 通过；30/40 因幻觉列名不可执行 |
| `.optim/kline5m/*.csv` | P33 数据：BTC/ETH/SOL 5m 各 220k 根（2024-01~2026-02） |
| `.optim/p33_search_5m_report.md` | P33 5m 长历史复核：BTC 留出全负、ETH 噪声微利、SOL 0 幸存者 → 无 edge |
| `.optim/kline1h_taker/*.csv` | P34 数据：BTC/ETH/SOL 1h 订单流（taker_ratio）2024-01 起 |
| `.optim/p34_taker_report.md` | P34 订单流因子：27 项 0 通过（IC 跨品种不统一、IS/OOS 翻转）→ 无 edge |
| `core/logging_config.py` | GAP-07④：交易事件独立 `trading.log`（engine.*/core.bus），与 HTTP 访问日志分离 |
| `tests/test_logging_split.py` | 2 例：交易事件隔离 + trader.log 不丢信息 + 重复 setup 不叠加 handler |
| `tests/test_auto_start_restart_drill.py` | GAP-07②：真实 lifespan + trading_auto_start 断电恢复演练（自启/FIFO/风控冷却） |
| `engine/monitor.py` | GAP-07⑤：统一告警中心（断链/无保护持仓/风控冷却/总线降级 → alerts.log + 事件 + API） |
| `tests/test_monitor_alerts.py` | 5 例：去重/结构化落盘/四类巡检/巡检不抛异常 |
| `tests/test_fault_injection_drill.py` | 4 例：故障注入演练（断流/下单丢失/对账差异/迟到成交 → 告警 + 安全降级） |
| `GET /api/trading/alerts` | 告警摘要接口；`status.alerts` 同步暴露（真实运行已验证 CRITICAL 断链告警） |
| `engine/trading_engine.py`（多标的） | GAP-06：symbols 列表 + 每 symbol 独立策略实例 + 组合风控（总仓位/单标的占比告警） |
| `tests/test_multi_symbol_engine.py` | 3 例：独立策略路由/单标的向后兼容/组合风控告警 |
| `web/api/ops.py` | 运维总览 `/api/ops/overview`：引擎活性+告警+行情+日志/DB 大小，聚合 health |
| `scripts/watchdog.py` | 独立看门狗探针（exit 0/1/2），供 supervisor/systemd 轮询 |
| `scripts/db_backup.py` | P0-1 数据库备份：在线备份(含 WAL 增量)+integrity 验证+恢复演练+保留策略+VACUUM |
| `tests/test_db_backup.py` | 4 例：WAL 一致性/恢复演练行数对比/保留策略/CLI 备份 |
| `scripts/service_supervisor.py` | P0-2 进程守护：拉起+探活+失败/崩溃自动重启（watchdog 只报警→能拉起） |
| `core/logging_config.py`（uvicorn_log_config） | P0-3 日志轮转：server.log 走 RotatingFileHandler（5MB×5），不再 stdout 直写 |
| `tests/test_log_rotation.py` | 3 例：配置结构/不动 root handler/run.py 接入 log_config |
| `tests/test_service_supervisor.py` | 4 例：子进程启停/日志JSON/不可达返回码/重启逻辑 |
| `tests/test_ops_monitor.py` | 5 例：心跳新鲜/老化假死/停止不误报/overview 聚合/CRITICAL→health=false |
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
