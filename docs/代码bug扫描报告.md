# 全局代码 Bug 扫描报告（2026-09-23）

扫描方式：`ruff check . --select F,E9`（未定义名/未用导入/未用变量/语法类），
排除 `.venv`/`.optim`（第三方/实验资产）。

## 扫描结果与处置

| 类别 | 数量 | 处置 |
|---|---|---|
| **F821 未定义名** | 3 | **1 个真 bug 已修复** + 2 个字符串注解 forward-ref 误报（加 noqa） |
| **F401 未使用导入** | 69 | ruff 自动修复 70 项（含 F541 等，全部删除） |
| **F841 未使用变量** | 9+1 | 1 个生产死代码删除（factors/analysis.py `out`）；8 个 tests 无害冗余（构造副作用/未用赋值，保留） |

## 真 Bug（已修复）

### 1. `web/main.py:176` — `HTTPException` 未导入（NameError）
- 位置：`vendor_file` 404 分支 `raise HTTPException(status_code=404, ...)`
- 影响：请求 `/static/vendor/<不存在文件>` → `NameError: name 'HTTPException' is not defined`（500 而非预期 404）
- 修复：`from fastapi import FastAPI, HTTPException, Request`

## 死代码（已删除）

### 2. `factors/analysis.py:177` — 遗留 `out = pd.Series(...)` 死赋值
- E3 优化后改用 `out_arr`（numpy 数组）承接计算，`out` 无后续引用——删除。

## 误报/保留（说明）

- **F821 ×2**：`test_optimizations_p0p1p2.py` / `test_stat_significance.py` 的**字符串注解**（`"BacktestConfig"` / `"OverfitReport"`，函数内 import 的 forward ref）——合法语法，ruff 保守误报，加 `# noqa: F821`。
- **F841 ×8（tests）**：`env`（TradingEnv 构造触发奖励塑形日志，副作用必需）、`loop/ret/original` 等测试桩/冗余赋值——无害，保留。

## 清理规模

- ruff 自动修复 **70 项**（未使用导入/无效 f-string），横跨源码与 tests；
- 回归：**123 passed + 4 xfailed**（streaming_factors 已知专项关闭），exit 0——无破坏。

## 建议（非阻塞）

1. tests 的 8 个 F841 可后续逐个清理或统一 noqa（低优先级）；
2. `drl/factor_miner.py:29` 存在无效 `# noqa` 指令（应删或改正）——已由 ruff 警告提示。