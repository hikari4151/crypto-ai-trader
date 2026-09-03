"""全部策略互通改动的端到端验证（对着真实运行中的服务打）。

顺序刻意设计：第 2 步在任何"策略仓库"请求之前查询实盘策略列表，
用以证明冷启动即恢复（修复前必须先访问 /strategy-repo/all 才能看到策略）。
"""
import sys
import time

import httpx

TOKEN = open("data/api_token.txt", encoding="utf-8").read().strip()
C = httpx.Client(base_url="http://127.0.0.1:8000",
                 headers={"X-API-Token": TOKEN}, timeout=300)
fails = []


def check(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'} | {label}{(' :: ' + detail) if detail else ''}")
    if not cond:
        fails.append(label)


# 1) 等服务就绪
for _ in range(30):
    try:
        if C.get("/api/health").json().get("ok"):
            break
    except Exception:  # noqa: BLE001
        pass
    time.sleep(1)
else:
    print("服务未就绪")
    sys.exit(1)

# 2) 冷启动即恢复：未访问过 /strategy-repo/all 前就有全部策略
r = C.get("/api/trading/strategies").json()
names = [s["name"] for s in r["strategies"]]
builtin = [s for s in r["strategies"] if s.get("builtin")]
check("实盘/回测下拉含内置策略", len(builtin) >= 5, f"内置 {len(builtin)} 个")
check("下拉无同名重复项", len(names) == len(set(names)),
      f"重复: {[n for n in names if names.count(n) > 1]}")
check("实盘/回测下拉含全部动态策略", len(names) >= 11, f"共 {len(names)}: {names}")
check("当前策略在列表中（不再静默跳策略）", r["current"] in names,
      f"current={r['current']}")

# 3) 回测此前报「未知策略」的 AI 策略
r = C.post("/api/backtest/run", json={
    "data_source": "demo", "strategy_name": "sr_pullback_reversal_v3",
    "symbol": "BTC/USDT", "timeframe": "1h", "limit": 800}).json()
task_id = r.get("task_id")
err, done = None, False
for _ in range(120):
    p = C.get(f"/api/backtest/progress/{task_id}").json()
    if p.get("error"):
        err = p["error"]
        done = True
        break
    if p.get("pct", 0) >= 100 or not p.get("running") and p.get("n"):
        done = True
        break
    time.sleep(1)
check("回测 AI 策略不再报未知策略", err is None and done, err or f"task={task_id} done={done}")

# 4) 统一策略参数真正下发 + 落库
r = C.post("/api/strategy-repo/params", json={
    "name": "meta_controller",
    "params": {"sub_strategies": "dual_ma,factor_signal,sr_pullback_reversal_v3",
               "mode": "ensemble", "meta_window": 33}})
ok = r.status_code == 200 and r.json().get("ok")
saved = (r.json().get("params") or {}).get("sub_strategies") if ok else r.text[:120]
check("POST /strategy-repo/params 保存成功", ok, str(saved))
check("非法参数被拒绝", C.post("/api/strategy-repo/params",
                              json={"name": "meta_controller", "params": {"no_such": 1}}
                              ).status_code == 400)
check("内置策略拒绝走本端点", C.post("/api/strategy-repo/params",
                                json={"name": "dual_ma", "params": {"fast_period": 5}}
                                ).status_code == 400)

# 5) 全策略列表：内置可见 + meta_controller spec 已更新
d = C.get("/api/strategy-repo/all").json()
kinds = {s["name"]: s.get("kind") for s in d["strategies"]}
check("/all 返回内置策略", kinds.get("dual_ma") == "builtin", str(list(kinds.items())[:3]))
mc_rows = [s for s in d["strategies"] if s["name"] == "meta_controller"]
check("/all 中 meta_controller 唯一（内置被同名动态覆盖）", len(mc_rows) == 1,
      f"{len(mc_rows)} 行")
mc = mc_rows[0]
check("meta_controller spec 参数已持久化到注册表",
      mc["default_params"].get("sub_strategies") == "dual_ma,factor_signal,sr_pullback_reversal_v3",
      str(mc["default_params"].get("sub_strategies")))
check("非默认数值参数也已保存（meta_window=33）",
      mc["default_params"].get("meta_window") == 33,
      str(mc["default_params"].get("meta_window")))
check("/all 数量 == 实盘列表数量（口径统一）",
      d["count"] == len(names), f"/all={d['count']} trading={len(names)}")

# 6) 策略对比覆盖动态策略
r = C.post("/api/backtest/compare", json={
    "data_source": "demo", "symbol": "BTC/USDT", "timeframe": "1h", "limit": 500})
cj = r.json()
got = [x["name"] for x in cj.get("results") or []]
check("/compare 参与对比数 > 内置数（含动态策略）",
      cj.get("compared", 0) > 6, f"compared={cj.get('compared')} results={got}")

# 7) 元控制器按新子策略池训练
r = C.post("/api/evolve/trigger/meta-controller")
check("元策略训练成功", r.status_code == 200 and r.json().get("ok"), r.text[:150])
st = C.get("/api/evolve/status").json()["meta_controller"]
check("训练使用用户选定的子策略池",
      st.get("sub_strategies") == "dual_ma,factor_signal,sr_pullback_reversal_v3",
      str(st.get("sub_strategies")))
check("训练无错误", not st.get("last_error"), str(st.get("last_error")))

# 8) 训练重新注册 spec 后，用户参数不被抹掉
mc_after = [s for s in C.get("/api/strategy-repo/all").json()["strategies"]
            if s["name"] == "meta_controller"][0]
check("训练后用户参数仍保留（meta_window=33）",
      mc_after["default_params"].get("meta_window") == 33,
      str(mc_after["default_params"]))

print("\n" + ("全部通过" if not fails else f"失败 {len(fails)} 项: {fails}"))
sys.exit(1 if fails else 0)
