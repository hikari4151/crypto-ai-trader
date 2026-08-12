import asyncio
import sys

sys.path.insert(0, r"C:\Users\sbxg\Desktop\crypto_ai_trader")
import httpx


async def main() -> None:
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get("http://127.0.0.1:8001/api/trading/strategies")
        data = r.json()
        cur = data["current"]
        st = next(s for s in data["strategies"] if s["name"] == cur)
        print("当前策略:", cur, "| 参数:", st.get("current_params", st["default_params"]))
        newp = dict(st["default_params"])
        newp["fast_period"] = 15
        r2 = await c.put("http://127.0.0.1:8001/api/trading/strategies/params", json={"params": newp})
        print("热更新返回:", r2.json())
        r3 = await c.get("http://127.0.0.1:8001/api/trading/strategies")
        st2 = next(s for s in r3.json()["strategies"] if s["name"] == cur)
        fp = st2["default_params"].get("fast_period")
        print("热更新后 fast_period:", fp)
        print("结果:", "OK 热更新生效" if fp == 15 else "FAIL 热更新丢失")


asyncio.run(main())
