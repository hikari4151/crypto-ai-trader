# Module B: 因子系统 — 实施计划

> 依赖架构契约：docs/plans/architecture.md §2.4（IC 衰变元数据）、§2.8（跨品种验证参数）
> 独占文件：`factors/mining.py`、`factors/model_factor.py`、`factors/library.py`、`factors/analysis.py`、`web/api/factors.py`、`indicators/technical.py`
> 任务：🔴2 跨品种验证 + 🟠6 IC 衰变自动下线 + 🟡10 因子扩展

**重要前置发现（已核实代码）**：
1. `factors/library.py` **已存在** `bias_20`（20期乖离率，:159-160）、`vol_20`（20期波动率，:54-55）。QUANT_ADVICE 🟡10 的 5 个"新因子"中有 2 个（`volatility_20`≈`vol_20`、`bias_20`=同名）实际已存在。**新因子命名必须避开**，或明确标注"复用现有同名因子"。
2. QUANT_ADVICE 🟡10 中 `momentum_1w = close/close.shift(7)-1` 对应现有库的 `mom_5`/`mom_10`（5/10 期动量，% 口径）——1 周=7 根 1d 或 168 根 1h，与现有 5/10/20 期不完全重合，可新增 `mom_7`。
3. 🟠6 涉及"策略隔离"（`factor_signal` 跳过 `alive=False` 因子），但 `strategies/factor_signal.py` **不在本模块独占文件**。采用防御式读取避免越界（详见 T4），并在架构文档 §8 记录。

---

## T1. 🔴2 mining.py factor_quality_gate 跨品种验证

**Files**：`factors/mining.py`（:552-580 `evaluate_mined_factors`）、`factors/analysis.py`（:227-271 `factor_quality_gate`）

**Interfaces**：
- 消费：`factor_quality_gate` 返回 dict（valid/reason/rank_ic/icicir/...）
- 产出：`evaluate_mined_factors(..., cross_validate_data: Optional[dict[str, pd.DataFrame]] = None)`：

```python
def evaluate_mined_factors(candidates: list[dict], df: pd.DataFrame,
                           h: int = 1, gates: Optional[dict] = None,
                           cross_validate_data: Optional[dict] = None) -> list[dict]:
    # 每个候选因子主品种 gate 判定后追加跨品种验证
    # cross_validate_data: {"ETH/USDT": df_eth, ...}（同周期、≥200 根K线）
```

**跨品种判定逻辑**（逐因子）：
```python
sv_cross = []
for sym, cdf in (cross_validate_data or {}).items():
    if len(cdf) < 200:
        continue
    s_c = fx.eval(cdf).astype(float)          # 复用 FactorExecutor
    g = factor_quality_gate(s_c, cdf["close"], h=h, gates=gates)
    if g["samples"] >= 20:
        sv_cross.append({"symbol": sym, "rank_ic": g["rank_ic"], "icir": g["icir"]})
# 判定：跨品种 IC 均值 < 0.01 或方向与主品种不一致 → failed
main_ic = gate["rank_ic"]
aligned = [x for x in sv_cross if np.sign(x["rank_ic"]) == np.sign(main_ic) or main_ic == 0.0]
cross_ok = (len(sv_cross) > 0 and
            np.mean([x["rank_ic"] for x in sv_cross if aligned]) >= 0.01 if aligned else False)
```

> 语义：有跨品种数据时，只有**所有期刊方向与主品种一致且平均 |rank_ic| ≥ 0.01** 才通过；无跨品种数据（cross_validate_data 为空/None）时输出 `cross_validation: "not_configured"`（⚠ 未配置，不阻断主品种 valid）。

**返回字段约束**：
- `evaluate_mined_factors` 每项追加：`cross_validation: "passed" | "failed" | "not_configured"`、`cross_detail: [{symbol, rank_ic, icir}]`
- **不改变现有键**（name/title/category/expression/logic/valid/rank_ic/ic/icir/...），`valid` 仍由主品种 gate 决定（跨品种只标记，不覆盖 valid——QUANT_ADVICE 语义：挖掘阶段标记、前端展示状态，不硬性拒绝）

**防回归红线**：
- `factor_quality_gate` 函数本体零改动；`evaluate_mined_factors` 不传新参数时行为**逐位不变**（现有调用方 web/api/factors.py:279、tests/ 不受影响）
- `evolve_factors`（:359-481）零改动（虽可受益，保持范围）

**验证**：
- 新增测试：主品种信号强但跨品种方向相反 → `cross_validation="failed"`；无跨品种 → `"not_configured"`；手动构造两个方向一致的 df → `"passed"`

---

## T2. 🔴2 web/api/factors.py 挖掘端点跨品种参数

**Files**：`web/api/factors.py`（`FactorMineIn` :35-45、`mine_factors` :224-310）

**Interfaces**：
- 产出：`FactorMineIn.cross_symbols: str = ""`（逗号分隔交易对，如 `"ETH/USDT,BNB/USDT"`）
- 产出：`_load_cross_data(body, engine)` → `Optional[dict[str, pd.DataFrame]]`

**实现**：
1. `FactorMineIn` 加字段（:45 后）：
   ```python
   cross_symbols: str = ""   # 逗号分隔：跨品种验证数据源，如 "ETH/USDT,BNB/USDT"
   ```
2. `mine_factors` 内，`_load_df(body)` 之后加载跨品种数据（复用 `backtest.data_loader.load_from_exchange`，与 `_load_df` 同源；demo/csv 源不支持跨品种——仅 exchange 源支持）：
   ```python
   cross_data = None
   if body.cross_symbols.strip() and body.data_source == "exchange":
       cross_data = {}
       for sym in [s.strip() for s in body.cross_symbols.split(",") if s.strip()]:
           try:
               df_c = await load_from_exchange(body.exchange, sym, body.timeframe, limit=body.limit)
               if len(df_c) >= 200:
                   cross_data[sym] = df_c
           except Exception as e:
               log.warning("[factors] 跨品种 %s 拉取失败: %s", sym, e)
   ```
3. `_mine_pipeline`（:278-286）调用 `evaluate_mined_factors(candidates, df, h=body.horizon, cross_validate_data=cross_data)`
4. 返回（:308-310）追加 `cross_validate_symbols: list(cross_data or {}).keys()`、`cross_validate_status`（每个 candidate 已带 `cross_validation`）
5. **自动注册时保留跨品种标记**：`custom[r["name"]]`（:294-305）追加 `"cross_validation": r.get("cross_validation", "not_configured")`

**防回归红线**：
- `_load_df`、`compute_factor_matrix`、`evolve_factors` 调用零改动；不传 cross_symbols 时行为与现状逐位一致
- exchange 源失败不阻断主流程（仅 warning + 标记 not_configured）

**验证**：
- `POST /api/factors/mine` 带 `cross_symbols="ETH/USDT"`（exchange 源）→ 返回含 `cross_validation` 字段；不带 → `"not_configured"`
- demo 源传 cross_symbols → 忽略（不改行为），响应仍正常

---

## T3. 🔴2 model_factor.py 模型元数据带跨品种 IC

**Files**：`factors/model_factor.py`（`fit_model_factor` :122-179、meta 构造 :171-176）

**Interfaces**：
- 产出：`fit_model_factor(..., cross_validate_data: Optional[dict] = None)`，`meta` 追加 `cross_val_ic`：

```python
meta = {
    ...,
    "cross_val_ic": cross_val_ic,   # {"ETH/USDT": {"rank_ic": ..., "icir": ...}, ...} 或 {}
}
```

**实现**：
1. `fit_model_factor` 加参数 `cross_validate_data: Optional[dict[str, pd.DataFrame]] = None`
2. 训练/质检完成后，对每个跨品种 df：用最后一窗模型预测 → `factor_quality_gate` → 记录 `rank_ic`/`icir`
   ```python
   cross_val_ic = {}
   for sym, cdf in (cross_validate_data or {}).items():
       if len(cdf) < 200:
           continue
       try:
           from drl.env import _precompute_features
           feats_c = np.asarray(_precompute_features(cdf), dtype=float)
           X_c = (feats_c - mu) / sd            # 复用训练段 mu/sd（防统计泄漏）
           pred_c = model.forward(X_c)[-1].reshape(-1)
           s_c = pd.Series(pred_c, index=cdf.index)
           g = factor_quality_gate(s_c, cdf["close"], h=h)
           cross_val_ic[sym] = {"rank_ic": g["rank_ic"], "icir": g["icir"]}
       except Exception as e:
           log.warning("[factor] 模型因子跨品种 %s 验证失败: %s", sym, e)
   ```

**防回归红线**：
- `fit_model_factor` 不传新参数时输出与现状逐位一致（cross_val_ic 为空 dict）
- `_model_eval_cfg`/`fit_model_factor` 现有签名向前兼容；`_segment_report`/`_ic_decay` 零改动

**验证**：
- 测试：demo 数据训练 + 手工构造另一段 df（close 反序）→ `meta["cross_val_ic"]` 有键且方向可观察；无跨品种 → `{}`

---

## T4. 🟠6 library.py 因子 alive 标记 + IC 衰变检测

**Files**：`factors/library.py`（`_LIB` :12、`_BY_KEY` :187、`list_factors` :198-199、`meta()`）、`factors/base.py`（`Factor` dataclass :35-43）

**Interfaces**（契约见 architecture.md §2.4）：
- 产出：`Factor.alive: bool = True`；`Factor.ic_decay: dict = field(default_factory=...)`
- 产出：`_adjust_alive_from_ic(key, rolling_ic_last: float, icir: float) -> None`（供滚动 IC 检测调用）
- 产出：`periodic_ic_refresh(data_by_symbol: dict[str, pd.DataFrame], horizon: int = 1) -> dict`（计算结果 + alive 变化列表）

**实现**：
1. `factors/base.py` `Factor` dataclass 新增两字段（:43 default_params 后）：
   ```python
   alive: bool = True
   ic_decay: dict = field(default_factory=lambda: {
       "rolling_ic": [], "rolling_icir": 0.0, "last_updated": None,
       "auto_offline": False, "auto_online": False,
   })
   ```

   > 注意：`Factor` 在 `factors/base.py` 内，虽不在 B 独占清单顶端，但该文件本就是因子系统基础类型且被 B 全量依赖，PM 已批准并入。若严格坚持清单，可将 `alive` 的实现改为 `library.py` 维护一个旁路 dict `_ALIVE: dict[str, dict]` 而不动 `base.py`——**推荐主方案**（改 base.py 语义最清晰）。

2. `factors/library.py` `_f(...)`（:15-17）：新增注册即带默认 alive/ic_decay。
3. 新增滚动 IC 检测逻辑（放 `library.py` 底部，或 `factors/analysis.py` 免新增文件的既有函数）：

```python
def periodic_ic_refresh(data_by_symbol: dict[str, pd.DataFrame], horizon: int = 1) -> dict:
    """用各品种最新行情计算所有已注册因子的滚动 IC，按衰变规则自动上线/下线。
    返回 {"updated": [key,...], "offlined": [key,...], "onlined": [key,...]}
    """
    from .analysis import factor_ic
    offlined, onlined = [], []
    for key, f in _BY_KEY.items():
        roll = f.ic_decay["rolling_ic"]
        valid_ics = []
        for sym, df in data_by_symbol.items():
            try:
                s = f.series(df).astype(float)
                if s.notna().sum() < 30:
                    continue
                r = factor_ic(s, df["close"], h=horizon, method="rank")
                valid_ics.append(r["rank_ic"])
            except Exception:
                continue
        if not valid_ics:
            continue
        roll.append(float(np.mean(valid_ics)))
        if len(roll) > 30:
            roll.pop(0)
        mean_ic = float(np.mean(roll))
        # ir 用最近 30 期滚动 IC 序列的 mean/std
        icir = float(np.mean(roll) / (np.std(roll) + 1e-12)) if len(roll) > 5 else 0.0
        f.ic_decay.update({"rolling_ic": roll, "rolling_icir": round(icir, 4),
                           "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")})
        if f.alive and mean_ic < -0.01 and icir < -0.1 and _consecutive_negative(roll) >= 3:
            f.alive = False
            f.ic_decay["auto_offline"] = True
            offlined.append(key)
            log.warning("[factor] 因子 %s IC 衰变（mean_ic=%.4f icir=%.4f）→ 自动下线", key, mean_ic, icir)
        elif not f.alive and mean_ic > 0.01 and _consecutive_positive(roll) >= 3:
            f.alive = True
            f.ic_decay["auto_online"] = True
            onlined.append(key)
            log.info("[factor] 因子 %s IC 恢复（mean_ic=%.4f）→ 自动上线", key, mean_ic)
    return {"updated": [k for k in _BY_KEY], "offlined": offlined, "onlined": onlined}
```

> `_consecutive_negative/positive(roll)`：统计序列末尾连续同符号的数量（≥3 才动作），实现为库内私有函数。

**防回归红线**：
- `Factor.meta()` 键不变（新增 `alive`/`ic_decay` 两个**新增键**，现有 key/name/category/category_label/description/default_params 不动）
- `list_factors`/`get_factor`/`factor_keys`/`factors_by_category` 签名不变（`list_factors()` 返回 `Factor` 对象，alive 直接从对象读）
- **不删除**任何因子（只标记 alive=False）；`compute_factor_matrix`/`factor_signal` 消费侧见 T5

**验证**：
- 测试：构造故意反向因子（close.shift(1)-close），用上涨 demo 数据 `periodic_ic_refresh` 调 3 次 → 自动下线（alive=False）；换用下跌数据 → 自动上线

---

## T5. 🟠6 策略隔离 + 端点状态展示 + 周期触发

**Files（本模块）**：`web/api/factors.py`（`/factors/library` :78-82、`/factors/custom` :85-89）
**Files（越界协调）**：`strategies/factor_signal.py`（:146-175 `_factor_value`）、`engine/trading_engine.py`（`_ai_scheduler`）

**实现（本模块内）**：
1. `/factors/library`（:78-82）返回加 `alive`/`ic_decay`（`f.meta()` 已含，零改动即可透出——确认 meta 含新键后无需改代码，仅验证）
2. `/factors/custom`（:85-89）返回的 KV 因子加 `cross_validation`（T2 已写入）、若需要 `alive` 状态补充写回

**策略隔离（越界协调，防回归红线 + 不直接改 factor_signal.py）**：
- 由于 `strategies/factor_signal.py` 不在本模块，方案为**在 `library.py` 暴露 `factor_is_live(key) -> bool`** 辅助函数，供 factor_signal 消费；是否让 factor_signal 调用由 PM 协调（架构文档 §8 记录）。本模块交付辅助函数 + 测试。
  ```python
  def factor_is_live(key: str) -> bool:
      f = _BY_KEY.get(key)
      return f.alive if f else False
  ```

**周期触发（越界协调）**：QUANT_ADVICE 建议"引擎运行时每天/每 100 根K线异步计算"，本模块交付**幂等、轻量的 `periodic_ic_refresh` 纯函数 + 单测**；挂到 `_ai_scheduler` 的周期由模块 C 协调（C 独占 trading_engine.py），架构文档 §3 已写明。

---

## T6. 🟡10 因子扩展 5 个新因子

**Files**：`factors/library.py`

**重要：新因子命名（规避已存在同名/功能近似）**

| QUANT_ADVICE 要求 | 命名 | 说明 |
|---|---|---|
| 动量：`momentum_1w = close/close.shift(7)-1` | `mom_7` | 1 周=7 根日线；现有库 5/10/20，7 期不重合。**注意库内为 % 口径**（×100） |
| 波动率：`volatility_20 = close.pct_change().rolling(20).std()` | `vol_20` | **已存在**（:54-55 `_realized_vol`·% 口径）——复用，不新增 |
| 成交量异常：`vol_spike = volume/volume.rolling(20).mean()-1` | `vol_spike_20` | 现有 `vol_ratio` 是 5 期均量；20 期是独立因子 |
| 价格位置：`price_position = (close-low20min)/(high20max-low20min)` | `price_pos_20` | 现有 `bb_pos` 是布林带位置；此为 Donchian 位置 |
| 乖离率：`bias_20` | `bias_20` | **已存在**（:159-160）——复用，不新增 |

**实现**（在 library.py 相应分类段追加注册，% 口径与库内一致）：

```python
# ============ 动量（新增） ============
_f("mom_7", "7期动量", CATEGORY_MOMENTUM, "过去7根K线收益率（%），1周动量。",
   lambda df: _roc(df["close"], 7), {"n": 7})

# ============ 量价（新增） ============
_f("vol_spike_20", "20期成交量突增", CATEGORY_VOLUME, "成交量相对20期均量之比减1，放量突增信号。",
   lambda df: (df["volume"] / (df["volume"].rolling(20).mean() + 1e-12) - 1.0) * 100.0, {"n": 20})
_f("price_pos_20", "20期价格位置", CATEGORY_TREND, "收盘价在20期最高/最低间的百分位（Donchian位置，0-1）。",
   lambda df: (df["close"] - df["low"].rolling(20).min()) /
              (df["high"].rolling(20).max() - df["low"].rolling(20).min() + 1e-12), {"n": 20})
```

> 说明：`vol_20`、`bias_20` 已存在，本任务**不重复注册**；QUANT_ADVICE 的 5 个因子中实际需新增 3 个（`mom_7`/`vol_spike_20`/`price_pos_20`），另 2 个为"确保已注册并被 factor_signal 可用"。

**analysis.py DEFAULT_GATES 对齐**（:193-198）：新因子的默认门槛沿用 `DEFAULT_GATES`（min_abs_ic=0.01、min_abs_icir=0.1、max_turnover=0.5）——**零改动**，因为所有因子共用同一 DEFAULT_GATES。QUANT_ADVICE 要求"每个因子注册默认参数"——`_f` 的 `default_params` 已含（`{"n": 7}`、`{"n": 20}`）。

**防回归红线**：
- 新增因子只 `_f(...)` 追加，**不改动任何既有因子**；`_BY_KEY` 重建于模块导入期（:187），新因子自动计入
- `factors_by_category` 返回顺序按注册顺序（新增在尾部），现有前端 `factorLibrary.slice(0,12)` 展示的前 12 个不受影响
- 新因子计算纯 pandas 向量化，满足 `base.py` 契约（无未来数据）

**验证**：
- `GET /api/factors/library` → 出现 `mom_7`/`vol_spike_20`/`price_pos_20` 且 `vol_20`/`bias_20` 已有
- `python -c "from factors.engine import compute_factor_matrix; import sys; ..."` 或 `POST /api/factors/compute`（demo）→ 新因子 IC 表有值且无 NaN 崩溃
- factor_signal 策略回测（demo）交易数不为 0（现有激素因子已保证，回归确认）

---

## 交付验收清单（模块 B）

- [ ] `python -m compileall -q factors indicators`
- [ ] `python -m pytest -q`（现有全绿 + 新增跨品种/衰变/新因子测试）
- [ ] `GET /api/factors/library` 含新因子与 alive/ic_decay 键
- [ ] `POST /api/factors/mine`（exchange + cross_symbols）返回 cross_validation 字段
- [ ] 手工注册反向因子 → periodic_ic_refresh 3 次 → library 显示下线
- [ ] `run.py backtest --source demo --strategy factor_signal` 回归正常（交易数不为 0）
- [ ] 记录越界协调项（factor_signal 策略隔离、periodic_ic_refresh 挂载）到架构文档 §8