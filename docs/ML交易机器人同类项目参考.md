# ML 交易机器人同类项目参考 —— 对照「持续进化停留在纸面」问题

> 版本：v1.0（2026-09-19）
> 方法：web_search 检索公开项目/论文/文章，摘要对照本仓库诊断结论（`docs/持续进化板块停留在纸面-诊断报告.md`）。
> 声明：以下为外部公开信息（README/摘要/论文标题），作为**线索与参照**，未对每个项目逐行审代码；引用请以原链接为准。

---

## 0. 对照起点：本程序的问题清单（诊断报告摘要）

1. 引擎在跑（三条循环 alive），但三管线全部 stalled：factor_miner 连续 28 轮回退、strategy_drl 连续 32 轮被拒、meta_controller 501 轮里 489 轮被拒。
2. strategy_drl 的级联因子注入因"窗口尾时间戳不一致"21 次被拒、0 次成功 → 模型样本外不交易。
3. 部署热加载对当前策略 `dual_ma` 一律跳过（32 次）→ 进化产物不接管运行实例。
4. 级联因子与 Pine 导出互斥：注入因子 → 有信号但无 Pine；不注入 → 有 Pine 但模型不交易。
5. 根因：真实数据 OOS 正期望（GAP-01）未证明；接受门槛虽严，但"没有 edge"才是实盘门过不去的原因。

---

## 1. 同类项目/框架速览

| 项目 / 文献 | 类型 | 与本程序问题相关的点 |
|---|---|---|
| [FinRL（arXiv:2111.09395）](https://ar5iv.labs.arxiv.org/html/2111.09395) | 开源 RL 交易框架 | 标准化的 RL 训练/评估协议（DQN/PPO/SAC 等），强调"环境-代理-评估"分离；可对照你的三管线（因子→策略→元）做可复现评估 |
| [TensorTrade](https://github.com/HoldemGK/tensortrade) | RL 交易框架 | 交易环境/奖励/执行组件化，奖励函数可插拔——对应 meta_controller 的"空仓懒解"奖励设计问题 |
| [AlphaForge](https://github.com/srgangaram-swe/AlphaForge) | 量化 ML 研究平台 | leakage-safe walk-forward + purged CV + **DSR / PBO 过拟合统计** + 成本化 OOS 回测 + regime 识别——直接对标你的 OOS 硬门/跨标安检，且比单点 `oos_ret>0` 更严格 |
| [Dukesan-ai/crypto-ml-trading-framework](https://github.com/Dukesan-ai/crypto-ml-trading-framework) | 真金实盘 ML 系统 | 真实资本运行 + **CPCV/DSR** + meta-labeling + 微结构特征——对应 GAP-01（真实数据正期望）与"部署到实盘"闭环 |
| [purgedcv](https://pypi.org/project/purgedcv/0.1.5/) | CPCV 工具库 | 时序/金融 ML 的 purged & combinatorial cross-validation 实现，可对照你的 OOS 最小样本数/分段评估是否足够防泄漏 |
| [Backtest overfitting in the machine learning era（ScienceDirect）](https://www.sciencedirect.com/science/article/abs/pii/S0950705124011110) | 学术评测 | 合成可控环境下比较各种 OOS 测试方法，结论指向"不是有 OOS 就行，方法选择本身会骗人" |
| [How Much Sharpe is Illusory? Quantifying Backtest Overfitting in Crypto Factor Strategies](https://istina.ficp.ac.ru/publications/article/824574961/) | 学术（crypto 因子） | 用 CSCV/PBO 量化 crypto 因子策略的回测过拟合——直接对应 factor_miner 的 |rank_ic| 安检与组合因子发布 |
| [MetaTrader: Learning to Generalize RL Trading Policies Beyond Offline Data（AAAI）](https://ojs.aaai.org/index.php/AAAI/article/view/40027/43988) | 论文 | 离线数据训练 RL 策略的**泛化**问题——对应 meta_controller/strategy_drl 在历史窗口外失效 |
| [Freqtrade / FreqAI（deepalpha freqai-plugin）](https://github.com/stefanoviana/deepalpha/blob/main/freqai-plugin/README.md) | 开源实盘机器人 + ML 插件 | ML 策略内嵌进实盘机器人并文档化过拟合/特征对齐注意事项——对应"产品化闭环"与数据对齐 |
| [KuCoin 行业文章：回测漂亮、实盘就废](https://www.kucoin.com/zh-hant/news/insight/BTC/6aa2ab3accf4ec00077c9c97) | 中文行业文章 | "绝大多数 AI 交易死在缺乏样本外验证"——几乎是本程序的写照（纸面↔实盘落差） |
| [Wisdom Trading: AI Overfitting in Trading Systems](https://wisdomtrading.com/insights/ai-overfitting-trading-systems/) | 实践文章 | 交易系统里 AI 过拟合的工程化警示与缓解 |
| [Citadel Global: THE MARKET IS NOT AN ALGORITHMIC PROBLEM](https://www.citadelglobal.co.za/markets-arent-algorithmic/) | 观点文 | 对"策略正期望"的清醒认知：市场不是纯算法问题，防止把工程努力误当 edge |
| [PopAgent（LLM 交易种群）](https://github.com/HowardLiYH/PopAgent) | 种群/自适应方法选择 | Thompson Sampling + 种群级学习，自适应选方法——对应你的 champion/challenger 与 meta 控制器"选子策略"问题 |
| [Ergodicity Library（进化种群训练器）](https://zenodo.org/records/16234469/files/Ergodicity%20Library%20-%20An%20Introduction.pdf) | 进化计算库 | 种群进化训练/保持多样性——对应持续进化的 PBT 思想与"陈旧锚点压死新模型" |
| [Harvard：OOS 报告与 lookahead bias](https://dash.harvard.edu/bitstreams/910c2858-97ee-4dc2-8a2a-a0fb35930df8/download) | 学术资料 | 样本外报告/前瞻偏差规范——对应级联因子窗口错位这类"隐性对齐问题" |
| [QuantStart: VibeQuant（研究→实盘自动化）](https://b-lab.team/en/content/e0c6e112-5419-4e9c-9dda-417b7a885988) | 端到端流程 | 从研究自动化到实盘落地的工作流——对应 LIVE_READINESS 的"纸面到实盘最短路径" |

---

## 2. 逐问题对照（重点）

### 2.1 OOS/过拟合门禁：门设对了，但"方法精度"和"edge 本身"差在哪

- **你做对的**：OOS 硬门（`_strategy_deploy_gate`）、回退保护（`_should_rollback`）、因子 `rank_ic>0` 方向门、跨标传导 OOS、demo 数据永不部署。这些和主流实践方向一致。
- **同类怎么做**：AlphaForge 用 **purged CV + DSR + PBO + 成本化 OOS**；`How Much Sharpe is Illusory` 用 **CSCV/PBO** 量化 crypto 因子过拟合；ScienceDirect 论文在合成环境比较 OOS 方法；purgedcv 提供现成 CPCV 实现。
- **对照差距**：你的 `oos_min_bars`/`oos_segments` 是简化版；实测接受率 factor_miner 50%、strategy_drl 1%、meta_controller 0%（保留 501 轮内），说明**门禁本身不是 bug，而是"真实数据上没有可重复的 edge"**。同行结论：单点 `oos_ret>0` 容易骗人，要 DSR（多次实验被运气解释的概率）与 PBO（参数搜索过拟合占比）才有说服力——`docs/LIVE_READINESS.md` GAP-01 的验收标准其实已经写了这一层，代码侧可对齐 AlphaForge/purgedcv。

### 2.2 纸面 → 实盘（部署/热加载）：同类项目怎么"真上"

- **你的现状**：统一部署入口 + 策略注册 + 运行时热加载回调都有，但当前策略 `dual_ma` 时 32 次 `reload skipped`——进化产物只进策略库，不接管运行实例。
- **同类怎么做**：Dukesan 框架直接真金实盘跑（CPCV/DSR + 实盘监控）；FreqAI 把 ML 策略内嵌进实盘机器人；QuantStart VibeQuant 做"研究→实盘"自动化流水线。
- **对照差距**：你需要一条显式闭环：testnet 沙盒验证（LIVE_READINESS E2）→ 小资金实盘 → 部署切换/回滚/监控（E3-E6）。当前"部署成功但 reload skipped"应视为**产品语义缺失**而非小毛病：进化模型从未被当作运行策略加载过。

### 2.3 数据/窗口对齐（级联因子注入断裂）：同类怎么防

- **你的现状**：`_load_cascade_factor_values` 要求 npy 的 tail_ts 与当前 df 尾戳**严格相等**；滚动窗口右移后必然不等 → 今日 21 次拒绝注入、0 次成功 → strategy_drl 变"无因子裸训"。
- **同类怎么做**：AlphaForge 强调 leakage-safe walk-forward；Harvard 资料规范 lookahead bias 报告；FreqAI 对特征/标签对齐有工程化处理。
- **对照差距**：严格相等校验在"因子先落盘、策略后拉数"的异步流水线里**设计上就不可靠**。同行做法通常是**同轮同 df 直接传递**，或按"窗口身份（起点+长度）"校验而不是"尾戳完全相等"；再退一步也要在拒绝时显式告警（你现在只在日志里 WARNING，面板不展示）。

### 2.4 奖励设计 / 空仓懒解（meta_controller 489/501 被拒）

- **你的现状**：meta_controller 长期 fitness=0.0（空仓=0 的懒最优解），加 `reward_signal_align` 后才出现过一次接受（v286）；最近 489 轮 OOS 稳定 -0.21。
- **同类怎么做**：MetaTrader（AAAI）专门研究 RL 策略离线泛化；TensorTrade 把奖励组件化便于改造；PopAgent 用种群+Thompson 自适应"选方法"，避免单策略懒解。
- **对照差距**：除了奖励塑形，可加**动作约束**（最低持仓比例/空仓惩罚/最低交易次数）从环境侧堵死懒解；并参考论文做法，把"训练段收益"与"OOS 泛化"分开评估（你已做，但 meta 的 OOS 负收益说明泛化问题仍未解决）。

### 2.5 持续进化 / 种群与"陈旧锚点"

- **你的现状**：champion/challenger K=2（PBT top-k 思想）、连续拒绝自动降频、因子挖掘有 from-scratch 重置；但 factor_miner 今天被 v307（0.2373）连续压 28 轮，strategy_drl/meta 没有等价重置机制。
- **同类怎么做**：PopAgent（种群+自适应选择）、Ergodicity Library（进化种群训练/多样性）、PBT 类方法在 RL 中用于对抗"环境漂移/基线陈旧"。
- **对照差距**：把 factor_miner 的"连续拒绝 → from-scratch 重训"推广到 strategy_drl/meta；对"历史锚点不可比/量级离谱"（demo 锚点 1267 事故）继续用不可比闸+归档重置（你已做）。

### 2.6 产品化：Pine 导出与级联互斥

- **你的现状**：注入级联因子后模型无法导出 Pine（预计算因子列不可逐根复算）；不注入则能导出但模型不交易。
- **同类怎么做**：FinRL 提供可部署 agent 接口；FreqAI 在实盘机器人内原生支持 ML（不需要导出成 Pine 脚本）。
- **对照差距**：二选一——(a) 支持"表达式因子"训练让模型可导出 Pine；(b) 明确把"仅软件内运行"作为一等产品形态，UI 标注而非让用户误以为可以上图。

---

## 3. 同行实践提炼的 5 条「纸面 → 落地」清单

1. **门禁要统计严谨**：`oos_ret>0` 只是门槛，要补 DSR / PBO / 成本翻倍扫描（AlphaForge、How Much Sharpe is Illusory、purgedcv、GAP-01 验收标准）。
2. **部署要闭环**：testnet 沙盒 → 小资金实盘 → 自动切换/回滚/监控（Dukesan、FreqAI、QuantStart）；"注册了但 reload skipped"不算部署。
3. **数据对齐要防窗口错位**：同轮直传或按窗口身份校验，而不是尾戳严格相等（AlphaForge、Harvard lookahead 规范、FreqAI）。
4. **奖励设计要防懒解**：空仓惩罚/最低动作约束 + 训练段/OOS 分离评估（TensorTrade、MetaTrader/AAAI）。
5. **连续进化要防陈旧锚点**：种群 + 多样性 + from-scratch 重置，推广到所有管线（PopAgent、Ergodicity、PBT）。

---

## 4. 参考链接清单

- FinRL: https://ar5iv.labs.arxiv.org/html/2111.09395
- TensorTrade: https://github.com/HoldemGK/tensortrade
- AlphaForge: https://github.com/srgangaram-swe/AlphaForge
- Dukesan crypto-ml-trading-framework: https://github.com/Dukesan-ai/crypto-ml-trading-framework
- purgedcv (PyPI): https://pypi.org/project/purgedcv/0.1.5/
- Backtest overfitting in the ML era: https://www.sciencedirect.com/science/article/abs/pii/S0950705124011110
- How Much Sharpe is Illusory: https://istina.ficp.ac.ru/publications/article/824574961/
- MetaTrader (AAAI): https://ojs.aaai.org/index.php/AAAI/article/view/40027/43988
- FreqAI plugin: https://github.com/stefanoviana/deepalpha/blob/main/freqai-plugin/README.md
- KuCoin 行业文章: https://www.kucoin.com/zh-hant/news/insight/BTC/6aa2ab3accf4ec00077c9c97
- Wisdom Trading AI Overfitting: https://wisdomtrading.com/insights/ai-overfitting-trading-systems/
- Citadel Global: https://www.citadelglobal.co.za/markets-arent-algorithmic/
- PopAgent: https://github.com/HowardLiYH/PopAgent
- Ergodicity Library: https://zenodo.org/records/16234469/files/Ergodicity%20Library%20-%20An%20Introduction.pdf
- Harvard lookahead bias / OOS: https://dash.harvard.edu/bitstreams/910c2858-97ee-4dc2-8a2a-a0fb35930df8/download
- QuantStart VibeQuant: https://b-lab.team/en/content/e0c6e112-5419-4e9c-9dda-417b7a885988
