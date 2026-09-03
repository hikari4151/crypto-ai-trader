# M6 代码清理 — 实施计划

> 独占文件：无（清理临时文件，统一 except 横跨多个文件）
> 任务：清理临时文件 + 统一宽泛 except 为精确异常

---

## 问题分析

### P1. 根目录临时文件

**当前**：
- `C:\Users\sbxg\Desktop\crypto_ai_trader\_tmp_scan.py`
- `C:\Users\sbxg\Desktop\crypto_ai_trader\_tmp_verify.py`
- `C:\Users\sbxg\Desktop\crypto_ai_trader\_smoke_test.py`
- `C:\Users\sbxg\Desktop\crypto_ai_trader\final_pytest.txt`

### P2. tests/ 目录临时文件

**当前**：
- `tests/_scratch_ic.py` ~ `_scratch_ic6.py`（6 个文件）
- `tests/check_ai_guards_render.py`
- `tests/check_ai_guards_ui.py`
- `tests/check_b_smoke.py`
- `tests/check_datalist.py`
- `tests/check_frontend_all.py`
- `tests/check_param_dropdown.py`
- `tests/check_param_dropdown2.py`

### P3. 宽泛 except 统一

**当前**：项目中大量使用 `except Exception:  # noqa: BLE001` 或 `except Exception as e:  # noqa: BLE001`。需要逐文件评估哪些可以收紧为精确异常类型。

---

## T1. 删除根目录临时文件

**Files**：`_tmp_scan.py`、`_tmp_verify.py`、`_smoke_test.py`、`final_pytest.txt`

```bash
rm "_tmp_scan.py" "_tmp_verify.py" "_smoke_test.py" "final_pytest.txt"
```

---

## T2. 删除 tests/ 临时文件

**Files**：`tests/_scratch_ic*.py`、`tests/check_*.py`

```bash
rm tests/_scratch_ic.py tests/_scratch_ic2.py tests/_scratch_ic3.py tests/_scratch_ic4.py tests/_scratch_ic5.py tests/_scratch_ic6.py tests/check_ai_guards_render.py tests/check_ai_guards_ui.py tests/check_b_smoke.py tests/check_datalist.py tests/check_frontend_all.py tests/check_param_dropdown.py tests/check_param_dropdown2.py
```

---

## T3. 统一宽泛 except

**Files**：逐文件评估 `except Exception` 是否可以收紧。

### 评估原则

| except 类型 | 适用场景 | 建议 |
|------------|---------|------|
| `except Exception` | 顶级循环体（守护进程防崩溃） | 保留（守护模式） |
| `except Exception` | 回调函数（on_progress 等） | 保留（防回调污染主流程） |
| `except Exception` | 网络请求（交易所 API） | 可收紧为 `except (httpx.HTTPError, ccxt.NetworkError, asyncio.TimeoutError)` |
| `except Exception` | DB 操作 | 可收紧为 `except (sqlalchemy.exc.SQLAlchemyError, asyncio.TimeoutError)` |
| `except Exception` | KV 操作 | 可收紧为 `except (KeyError, ValueError, asyncio.TimeoutError)` |
| `except Exception` | 文件操作 | 可收紧为 `except (IOError, OSError, json.JSONDecodeError)` |
| `except Exception` | 数值运算 | 可收紧为 `except (TypeError, ValueError, ZeroDivisionError)` |

### 具体文件评估

**高优先级（可收紧且有明确异常类型）**：

1. `engine/order_manager.py`：
   - 行 171：`except Exception as e:  # noqa: BLE001` — 纸面成交异常 → 可收紧为 `except ValueError`
   - 行 238：`except Exception as e:  # noqa: BLE001` — 撤单后复核 → 可收紧为 `except (ccxt.NetworkError, asyncio.TimeoutError)`
   - 行 271：`except Exception as e:  # noqa: BLE001` — 实盘下单 → 可收紧为 `except (ccxt.NetworkError, ccxt.ExchangeError, asyncio.TimeoutError)`

2. `engine/trading_engine.py`：
   - 行 127：`except Exception as e:  # noqa: BLE001` — 读取配置 → 可收紧为 `except (json.JSONDecodeError, KeyError, ValueError)`
   - 行 156：`except Exception as e:  # noqa: BLE001` — 读取密钥 → 可收紧为 `except Exception`（保留，DB 操作不确定）
   - 行 179：`except Exception as e:  # noqa: BLE001` — 恢复每日盈亏 → 可收紧为 `except (ValueError, KeyError)`
   - 行 197：`except Exception:  # noqa: BLE001` — 启动失败 → 保留（顶层恢复）
   - 行 439：`except Exception as e:  # noqa: BLE001` — 更新 Trade PnL → 可收紧为 `except (sqlalchemy.exc.SQLAlchemyError, asyncio.TimeoutError)`
   - 行 469：`except Exception as e:  # noqa: BLE001` — 快照循环 → 保留（守护模式）
   - 行 676：`except Exception as e:  # noqa: BLE001` — AI 调度 → 保留（守护模式）

3. `engine/risk.py`：
   - 行 94：`except Exception as e:  # noqa: BLE001` — 加载每日盈亏 → 收紧为 `except (ValueError, KeyError)`
   - 行 111：`except Exception as e:  # noqa: BLE001` — 归档盈亏 → 收紧为 `except (ValueError, KeyError)`
   - 行 118：`except Exception as e:  # noqa: BLE001` — 保存每日盈亏 → 收紧为 `except (ValueError, KeyError)`
   - 行 284：`except Exception as e:  # noqa: BLE001` — 风控检查异常 → 保留（fail-safe，必须捕获所有异常）

4. `indicators/vectorized.py`：
   - 行 201：`except Exception:  # noqa: BLE001` — GPU 验证 → 保留（ImportError 有多种）

5. `drl/agent.py`：
   - 行 369：`except Exception as e:  # noqa: BLE001` — 因子表达式计算 → 收紧为 `except (ValueError, TypeError, ImportError)`
   - 行 408：`except Exception as e:  # noqa: BLE001` — 基础模型加载 → 收紧为 `except (FileNotFoundError, json.JSONDecodeError, KeyError)`
   - 行 511：`except Exception as e:  # noqa: BLE001` — 验证失败 → 保留（通用）
   - 行 527：`except Exception as e:  # noqa: BLE001` — 进度回调 → 保留（防回调污染）
   - 行 568：`except Exception as e:  # noqa: BLE001` — OOS 评估 → 保留（通用）

6. `drl/env.py`：
   - 行 428：`except Exception:  # noqa: BLE001` — GPU 验证 → 保留（ImportError 有多种）

7. `factors/analysis.py`：
   - 行 109：`except Exception:  # noqa: BLE001` — IC 分析 → 收紧为 `except (ValueError, TypeError)`
   - 行 139：`except Exception:  # noqa: BLE001` — 分组收益 → 收紧为 `except (ValueError, TypeError)`

8. `backtest/engine.py`：
   - 行 242：`except Exception:  # noqa: BLE001` — _is_ai_design → 保留（通用）

9. `backtest/fast_engine.py`：
   - 行 201：`except Exception as e:  # noqa: BLE001` — 进度回调 → 保留（防回调污染）
   - 行 252：`except Exception:  # noqa: BLE001` — _is_ai_design → 保留（通用）

---

## 文件修改清单

| 文件 | 改动要点 | 与其他模块冲突 |
|------|---------|--------------|
| 根目录 | 删除 4 个临时文件 | 无冲突 |
| `tests/` | 删除 13 个临时文件 | 无冲突 |
| `engine/order_manager.py` | 收紧 3 个 except 子句 | 与 M1 共享该文件，需在 M1 之后执行 |
| `engine/trading_engine.py` | 收紧部分 except 子句（保留守护模式） | 与 M2 共享该文件，需在 M2 之后执行 |
| `engine/risk.py` | 收紧部分 except 子句（保留 fail-safe） | 与 M5 共享该文件，需在 M5 之后执行 |
| `drl/agent.py` | 收紧部分 except 子句 | 与 M4 共享该文件，需在 M4 之后执行 |
| `factors/analysis.py` | 收紧 2 个 except 子句 | 需在 M3 之后执行 |
| `drl/env.py`、`indicators/vectorized.py` | 保留（GPU 验证路径） | 无冲突 |

---

## 验收标准

- [ ] 根目录下无 `_tmp_*.py`、`_smoke_test.py`、`final_pytest.txt`
- [ ] `tests/` 下无 `_scratch_ic*.py`、`check_*.py`（除测试框架文件外）
- [ ] 收紧的 except 子句在对应异常类型下仍能正常捕获和处理
- [ ] 所有修改后的 `except` 子句不会导致未捕获异常崩溃
- [ ] `python -m compileall -q backtest engine indicators drl factors`