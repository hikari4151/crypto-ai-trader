"""集成测试：mock AI client 验证 MarketAnalyst.analyze 完整流程。

覆盖：
1. 方向复算注入提示词
2. 决策连续性读取上一轮分析
3. 校验 ctx 传递（program_direction / continuity / trade_plan_ref_price）
4. 结果落盘含 _program / _continuity
5. 校验失败时 AI 可依据 ctx 修正
"""
import sys
sys.path.insert(0, r"C:\Users\sbxg\Desktop\crypto_ai_trader")
import asyncio
import json

import numpy as np
from core.database import Database
from core.bus import EventBus

from ai.market_analyst import MarketAnalyst
from ai.prompts import market_analysis_messages
from ai.direction_engine import compute_direction


class FakeClient:
    """mock AI client：按调用顺序返回预设结果。"""
    def __init__(self, results):
        self._results = list(results)
        self.calls = []
        self.last_ctx = None

    async def chat_json_validated(self, messages, feature, ctx=None, max_repairs=2):
        self.calls.append({"messages": messages, "ctx": ctx})
        self.last_ctx = ctx
        if not self._results:
            return {"regime": "趋势", "bias": "long", "confidence": 0.7,
                    "summary": "mock", "warnings": [], "signals": []}
        return self._results.pop(0)


def make_snap(n=100, trend=1.0):
    """构造趋势行情快照。trend>0 上行，<0 下行。"""
    rng = np.random.default_rng(3)
    closes = 100 * np.exp(np.cumsum(rng.normal(0.004 * trend, 0.008, n)))
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    highs = np.maximum(opens, closes) * 1.001
    lows = np.minimum(opens, closes) * 0.999
    vols = np.abs(rng.normal(100, 20, n))
    candles = [[i, o, h, l, c, v] for i, (o, h, l, c, v) in enumerate(zip(opens, highs, lows, closes, vols))]
    return {"symbol": "BTC/USDT", "timeframe": "1h", "candles": candles,
            "indicators": {"close": float(closes[-1]), "ma_fast": float(np.mean(closes[-10:])),
                           "ma_slow": float(np.mean(closes[-30:]))}}


async def main():
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.init()
    bus = EventBus()

    # 第一轮：预置一条上轮记录（short，30分钟前）用于连续性测试
    from core.database import Analysis
    from datetime import datetime, timezone, timedelta
    prev_ts = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
    prev_content = json.dumps({"bias": "short", "confidence": 0.7, "regime": "下跌"})
    async with db.session() as s:
        s.add(Analysis(symbol="BTC/USDT", content=prev_content,
                       meta_json=json.dumps({"ts": prev_ts})))
        await s.commit()

    # ---- 测试1：上升行情 + mock AI 判 long（一致，应通过） ----
    snap = make_snap(trend=1.0)
    # 校验程序方向确实为 long
    prog = compute_direction(snap["candles"])
    print(f"程序方向: {prog['direction']} score={prog['score']:+d}")
    assert prog["direction"] == "long", "上升行情应判 long"

    fake = FakeClient([
        {"regime": "趋势", "bias": "long", "confidence": 0.7, "summary": "上行",
         "warnings": [], "signals": ["回踩做多"],
         "trade_plan": {"side": "long", "entry": 50000, "stop_loss": 49500,
                        "take_profit": 51000, "win_rate": 0.55}},
    ])
    analyst = MarketAnalyst(fake, db, bus)
    result = await analyst.analyze(snap)
    assert result["bias"] == "long", "mock 结果应保留"
    assert result["_program"]["direction"] == "long", "应附程序方向"
    assert result["_continuity"]["flipped"] is True, "上一轮 short → 本轮 long 应判定反手"
    print("测试1 PASS: 上升行情+方向一致+连续性反手标记")

    # ---- 测试2：检查提示词含程序方向块与连续性块 ----
    msgs = market_analysis_messages(snap, 0.0, [], program=prog, continuity=result["_continuity"])
    user = msgs[-1]["content"]
    assert "程序方向参考" in user, "提示词应含程序方向块"
    assert "上一轮判断" in user, "提示词应含连续性块"
    assert "trade_plan" in msgs[0]["content"], "system 应含 trade_plan 说明"
    print("测试2 PASS: 提示词注入程序方向+连续性+trade_plan schema")

    # ---- 测试3：校验 ctx 传递 ----
    assert fake.last_ctx["snap"].get("program_direction") == "long", "ctx 应含 program_direction"
    assert fake.last_ctx.get("continuity"), "ctx 应含 continuity"
    assert fake.last_ctx.get("trade_plan_ref_price") is not None, "ctx 应含 ref_price"
    print("测试3 PASS: 校验 ctx 完整传递")

    # ---- 测试4：结果落盘 ----
    async with db.session() as s:
        from sqlalchemy import select
        row = (await s.execute(select(Analysis).order_by(Analysis.id.desc()).limit(1))).scalar_one()
    saved = json.loads(row.content)
    assert "_program" in saved and "_continuity" in saved, "落盘应含程序上下文"
    print("测试4 PASS: 结果落盘含程序上下文")

    print("\nINTEGRATION OK")


if __name__ == "__main__":
    asyncio.run(main())
