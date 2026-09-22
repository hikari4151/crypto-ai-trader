"""统一告警中心（engine/monitor.py）回归测试。

覆盖实盘运维最怕的几类"出事了不知道"：
  ① 行情断流（hub 长时间无新K线）
  ② 持仓无兜底止损（实盘最危险的静默状态）
  ③ 风控冷却中（需人工关注）
  ④ 事件总线 handler 连续失败（健康降级）
以及告警本身的工程性质：去重、结构化落盘、引擎 status 暴露。
"""
import json
import time

import pytest

from engine.monitor import CRITICAL, WARNING, AlertMonitor


def _monitor(tmp_path, dedup=60.0):
    return AlertMonitor(None, log_dir=str(tmp_path), dedup_sec=dedup)


def test_alert_dedup_and_persist(tmp_path):
    """同 (kind,key) 在去重窗口内只发一次，但每次都计数；告警结构化落盘。"""
    m = _monitor(tmp_path)
    first = m.alert(CRITICAL, "market_stale", "断流", key="BTC/USDT", age_sec=999)
    second = m.alert(CRITICAL, "market_stale", "断流", key="BTC/USDT", age_sec=999)
    assert first is not None and second is None, "重复告警应被去重抑制"
    assert first.suppressed == 1, "被抑制的次数应累计在原记录上"
    other = m.alert(CRITICAL, "market_stale", "断流", key="ETH/USDT")
    assert other is not None, "不同 key 不应被去重"
    lines = (tmp_path / "alerts.log").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2, f"只应落盘 2 条（去重生效），实际 {len(lines)}"
    rec = json.loads(lines[0])
    for k in ("ts", "level", "kind", "key", "message", "detail", "suppressed"):
        assert k in rec, f"告警行缺少字段 {k}"
    assert rec["level"] == CRITICAL and rec["kind"] == "market_stale"


class _Hub:
    def __init__(self, age):
        self._age = age

    def last_kline_age(self, symbol, timeframe):
        return self._age


class _PS:
    def __init__(self, stops):
        self._stops = stops

    def status(self):
        return {"enabled": True, "stops": self._stops}


class _OM:
    def __init__(self, stops):
        self.protective = _PS(stops)


class _Risk:
    def __init__(self, cooldown_until):
        self._cooldown_until = cooldown_until


class _Bus:
    def __init__(self, failures):
        self._handler_failures = failures


class _Engine:
    """最小引擎替身：只暴露 health_check 需要的属性。"""

    def __init__(self, age=1.0, pos=0.0, stops=None, cd=0.0, fails=None,
                 mode="live", symbol="BTC/USDT"):
        self.hub = _Hub(age)
        self._pos = pos
        self.order_manager = _OM(stops or {})
        self.risk = _Risk(cd)
        self.bus = _Bus(fails or {})
        self.trading_mode = mode
        self.symbol = symbol
        self.timeframe = "1h"

    def _stale_threshold(self):
        return 300.0

    def _current_position_qty(self):
        return self._pos


@pytest.mark.asyncio
async def test_health_check_market_stale(tmp_path):
    m = _monitor(tmp_path)
    await m.health_check(_Engine(age=9999.0))
    assert m.summary()["counts"].get("market_stale") == 1
    # inf（从未收到K线）同样告警
    await m.health_check(_Engine(age=float("inf")))
    assert m.summary()["counts"].get("market_stale") == 2


@pytest.mark.asyncio
async def test_health_check_unprotected_position(tmp_path):
    """有仓位但无兜底止损（实盘）→ CRITICAL。"""
    m = _monitor(tmp_path)
    await m.health_check(_Engine(pos=0.5, stops={}, mode="live"))
    assert m.summary()["counts"].get("position_unprotected") == 1, "实盘无保护持仓必须告警"
    # 有止损则不告警
    m2 = _monitor(tmp_path / "b")
    await m2.health_check(_Engine(pos=0.5, stops={"BTC/USDT": {"id": "s1"}}, mode="live"))
    assert "position_unprotected" not in m2.summary()["counts"]
    # 纸面模式不告警（无服务端止损概念）
    m3 = _monitor(tmp_path / "c")
    await m3.health_check(_Engine(pos=0.5, stops={}, mode="paper"))
    assert "position_unprotected" not in m3.summary()["counts"]


@pytest.mark.asyncio
async def test_health_check_risk_cooldown_and_bus(tmp_path):
    m = _monitor(tmp_path)
    eng = _Engine(cd=time.time() + 600,
                  fails={object(): {"count": 7, "last_error": "boom"}})
    await m.health_check(eng)
    c = m.summary()["counts"]
    assert c.get("risk_cooldown") == 1, "风控冷却应播报"
    assert c.get("bus_handler_failing") == 1, "总线 handler 连续失败应告警"


@pytest.mark.asyncio
async def test_health_check_never_raises(tmp_path):
    """巡检自身异常不得抛出（否则会拖垮引擎周期循环）。"""
    m = _monitor(tmp_path)

    class _Broken:
        @property
        def hub(self):
            raise RuntimeError("boom")

        def __getattr__(self, item):
            raise RuntimeError("boom")

    await m.health_check(_Broken())  # 不应抛异常


def test_alert_log_rotation_when_oversized(tmp_path):
    """alerts.log 超 max_bytes → 滚动为 .1，新告警写入新文件（防无限增长）。"""
    m = AlertMonitor(None, log_dir=str(tmp_path), dedup_sec=10.0,
                     max_bytes=200, backup_count=3)
    # 写多条告警直到超过 200B 触发轮转
    for i in range(20):
        m.alert(WARNING, f"k{i}", "x" * 50, key=f"key{i}")
    # 轮转后应有主文件 + 至少一个 .1 备份
    main = tmp_path / "alerts.log"
    assert main.exists() and main.stat().st_size < 2000, "主文件应保持小（已轮转）"
    backups = sorted(tmp_path.glob("alerts.log.*"))
    assert backups, "应存在轮转备份 alerts.log.1"
    # 备份是可解析的 JSON 行（旧记录没丢）
    txt = backups[0].read_text(encoding="utf-8")
    assert txt.strip() and "kind" in txt


def test_alert_log_rotation_max_backup_count(tmp_path):
    """轮转保留份数受 backup_count 限制：超过的旧备份被清理。"""
    m = AlertMonitor(None, log_dir=str(tmp_path), dedup_sec=5.0,
                     max_bytes=150, backup_count=2)
    for i in range(40):
        m.alert(WARNING, f"k{i}", "y" * 60, key=f"key{i}")
    backups = sorted(tmp_path.glob("alerts.log.*"))
    assert len(backups) <= 2, f"备份份数应 ≤2（backup_count），实际 {len(backups)}"