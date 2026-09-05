# AI Iteration Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure AI iteration candidates are validated before application, multi-segment DRL OOS metrics are statistically correct, and evolve APIs reject unsafe inputs.

**Architecture:** Keep the existing TradingEngine validation gate as the single performance-policy boundary. The AI API selects a candidate but never mutates strategy state directly; DRL reports aggregate per-segment metrics without concatenating independently reset equity curves; evolve API and ModelZoo enforce defense-in-depth name/config validation while preserving D5-D9 behavior and flat model compatibility.

**Tech Stack:** Python 3.13, FastAPI, asyncio, pandas/numpy, pytest.

## Global Constraints

- Preserve existing OOS hard gates, anchors, rollback persistence, startup restoration, and flat model compatibility.
- Do not revert unrelated uncommitted worktree changes.
- Keep valid existing configuration behavior and response compatibility where possible.
- Use UTF-8 for new files; run the repository verification chain before claiming completion.

---

### Task 1: Make auto-optimize validate before applying

**Files:**
- Modify: `engine/trading_engine.py` around `_validate_param_update` and strategy parameter application
- Modify: `web/api/ai.py` `/auto-optimize`
- Test: `tests/test_ai_apply_gates.py`

**Interfaces:**
- Produce `TradingEngine.validate_and_apply_ai_params(name: str, proposed: dict, current: dict | None = None) -> tuple[bool, dict]`.
- The method returns validation details and only invokes the existing locked application path after validation succeeds.

- [ ] Write tests asserting rejected candidates do not call `apply_strategy_params`, accepted candidates call it once, and grid-selected candidates are validated as the final candidate.
- [ ] Run the focused tests and confirm failure against the current direct-apply implementation.
- [ ] Implement the engine helper by snapshotting current params, calling `_validate_param_update`, and applying only on success; preserve the explicit validation-disabled compatibility mode with an explicit result flag.
- [ ] Change `/auto-optimize` to defer all mutation until AI/grid candidate selection finishes, then call the helper once and store `validation`, `candidate_source`, `applied`, and rejection status in the task result.
- [ ] Run `pytest tests/test_ai_apply_gates.py -q` and the existing AI guard/scheduler tests.

### Task 2: Correct multi-segment DRL OOS metrics

**Files:**
- Modify: `drl/agent.py` multi-segment OOS evaluation
- Test: `tests/test_drl_oos_segments.py`

**Interfaces:**
- Preserve existing report keys and single-segment numeric behavior.
- For multiple segments, report `oos_sharpe` and `oos_max_drawdown` as the median of per-segment metrics.

- [ ] Add deterministic tests for per-segment aggregation, single-segment compatibility, and non-divisible OOS lengths.
- [ ] Run the focused tests and confirm they expose concatenation/trim behavior.
- [ ] Track per-segment equity statistics inside the segment loop, aggregate them without concatenating independently reset absolute equities, and make the final segment include the remainder.
- [ ] Run focused DRL/OOS tests and existing evolve gate tests.

### Task 3: Harden evolve API and ModelZoo boundaries

**Files:**
- Modify: `drl/model_zoo.py` path helpers
- Modify: `web/api/evolve.py` versions, rollback, config, and trigger error mapping
- Modify: `drl/evolve_engine.py` config input validation and trigger acquisition if needed
- Test: `tests/test_evolve_api_guards.py`

**Interfaces:**
- Valid model names follow the existing `[A-Za-z0-9_.-]+` rule and must resolve below `models_dir`.
- Rollback versions are positive integers.
- Config accepts only known keys, list-of-symbols values, and strict booleans.

- [ ] Add tests for path traversal/invalid names, non-positive rollback versions, unknown config keys, string symbols, strict false parsing, and not-running trigger details.
- [ ] Run the focused API tests and confirm current unsafe inputs are accepted or misreported.
- [ ] Implement defense-in-depth model-name and resolved-root checks without removing flat file writes; add `Query(..., ge=1)` for rollback.
- [ ] Reject unknown config keys and malformed symbols before `apply_config`; preserve its transactional update/persistence behavior and existing numeric ranges.
- [ ] Return `error or reason` from trigger endpoints and preserve busy=409.
- [ ] Run focused evolve/OOS/anchor/increment tests.

### Task 4: Verification and regression pass

**Files:**
- Modify only tests or implementation files if failures identify a direct regression.
- Update: `docs/REDEV_GUIDE.md` only if a new baseline or red-line decision is required.

- [ ] Run all targeted tests for AI, DRL, evolve, and meta paths.
- [ ] Run full `pytest -q`.
- [ ] Run `python -m compileall -q ai backtest core drl engine exchange factors indicators strategies web config scripts run.py`.
- [ ] Run `python scripts/check_js.py web/static/index.html`.
- [ ] Run the demo dual-engine backtest and compare return/trade count with `docs/REDEV_GUIDE.md`.
- [ ] Report any remaining failure or skipped verification explicitly.
