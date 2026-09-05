# Backtest Header and Data Guards Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the second context bar transparent and floating, add the missing 5m backtest option, and prevent exchange-backed backtests from presenting empty data as a valid result.

**Architecture:** Preserve the topmost `.menubar` completely unchanged. Adjust only `.ctxbar` and its chips so the context row has no opaque strip or separator and the visible categories float independently. Keep the existing exchange loader, but enforce a data sufficiency guard at every backtest execution entry before the engine runs, with an explicit user-facing failure status.

**Tech Stack:** Vue 3 CDN template, CSS in `web/static/index.html`, FastAPI, pandas, pytest.

## Global Constraints

- Do not modify `.menubar` markup or styles.
- Preserve the existing 46px menu bar, 44px context row, and 90px body top offset.
- Keep existing exchange/data-source semantics; never silently replace exchange data with demo data.
- Do not run a backtest engine on an empty or insufficient DataFrame.
- Preserve unrelated uncommitted worktree changes.

---

### Task 1: Add regression tests for empty exchange data

**Files:**
- Test: `tests/test_backtest_empty_exchange.py`
- Modify: `web/api/backtest.py`

**Interfaces:**
- Add a small shared validation helper, `web.api.backtest._require_backtest_data(df, source)`, which raises `ValueError` with a source-specific message when data is empty or has fewer than 60 rows.

- [ ] Write tests for empty exchange data and fewer-than-60 exchange rows, asserting the helper refuses both.
- [ ] Run `python -m pytest tests/test_backtest_empty_exchange.py -q` and confirm the tests fail because the helper is absent.
- [ ] Implement `_require_backtest_data` and call it in the async worker after each data source is loaded, before `BacktestConfig`/`run_backtest_fast`.
- [ ] Apply the same guard to `/overfit`, `/grid-scan`, `/compare`, `/cost-scan`, and portfolio data-loading entry points where an empty exchange DataFrame could otherwise be processed.
- [ ] Run the new tests plus existing backtest/data-loader tests.

### Task 2: Add the 5m period option

**Files:**
- Modify: `web/static/index.html`
- Test: `tests/test_backtest_ui_contract.py`

**Interfaces:**
- Keep `bt.timeframe` as the single source of truth and submit `5m` unchanged to the API.

- [ ] Add a contract test that reads the template and asserts the backtest timeframe choices include `5m`.
- [ ] Run the test and confirm it fails against the current array `['15m','30m','1h','4h','1d']`.
- [ ] Add `5m` to the existing timeframe button array without changing other choices or bindings.
- [ ] Run the template contract test and `python scripts/check_js.py web/static/index.html`.

### Task 3: Make the second bar transparent and floating

**Files:**
- Modify: `web/static/index.html`
- Test: `tests/test_backtest_ui_contract.py`

**Interfaces:**
- `.menubar` remains byte-for-byte untouched by this task.
- `.ctxbar` keeps fixed positioning and height but uses transparent background, no border, no inset shadow, and floating `.ctx-chip`/`.ctx-right` surfaces.

- [ ] Add CSS contract assertions that `.menubar` declarations are unchanged and the final `.ctxbar` declaration includes `background:transparent`, `border:0`, and no separating shadow.
- [ ] Run the contract test and confirm it fails against the current final CSS.
- [ ] Update only `.ctxbar`, `.ctx-chip`, and `.ctx-right` final overrides; use subtle translucent chip backgrounds, 1px low-contrast borders, and compact shadows. Keep responsive overflow behavior intact.
- [ ] Run CSS/JS contract checks and inspect the rendered page at desktop and narrow widths if a local server is available.

### Task 4: Full verification

**Files:**
- No additional files unless a test exposes a direct regression.

- [ ] Run targeted backtest/UI tests.
- [ ] Run full `pytest -q`.
- [ ] Run `python -m compileall -q ai backtest core drl engine exchange factors indicators strategies web config scripts run.py`.
- [ ] Run `python scripts/check_js.py web/static/index.html`.
- [ ] Run `git diff --check` on the files touched in this plan.
- [ ] Report that the topmost menu bar was not modified and report any environment-dependent browser check that could not run.
