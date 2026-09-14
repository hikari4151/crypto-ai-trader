"""持续进化后台「保活」回归测试。

覆盖此前会「静默停训」的故障形态：训练循环 asyncio 任务已经结束，而引擎对外
仍报 running=true / enabled=true —— status() 虽然算了 tasks_alive，但全仓无人
消费它，也没有任何重建逻辑，只能重启进程恢复。

本文件锁定三件事：
1. 循环任务意外退出 → 保活监督在若干巡检周期内重建它；
2. 循环体抛出**非 Exception 基类**错误 → 降级为一轮失败，任务不死（并带心跳）；
3. 保活状态可观测：status().supervisor 的 loops_alive / degraded / restarts。
"""
import asyncio

import pytest


class _StubDB:
    """保活测试只需 kv 接口（通知去重走 get_notify_config），不碰 SQL。"""

    async def kv_json_get(self, key):
        return None

    async def kv_json_set(self, key, value):
        return None


def _make_engine(tmp_path):
    from core.bus import EventBus
    from drl.evolve_engine import EvolveEngine
    eng = EvolveEngine(_StubDB(), EventBus(), tmp_path / "models")
    # 把巡检/重试压到毫秒级，测试不必等真实间隔
    eng._supervise_interval = 0.05
    eng._restart_min_gap = 0.2
    eng._RETRY_INTERVAL = 0.02
    eng._running = True
    eng._enabled = True
    return eng


def _install_stub_loops(eng, factory):
    """用桩替换三条训练循环（实例属性遮蔽类方法）。"""
    eng._factor_miner_loop = factory
    eng._strategy_drl_loop = factory
    eng._meta_controller_loop = factory


async def _wait_until(pred, timeout: float = 5.0) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if pred():
            return True
        await asyncio.sleep(0.02)
    return pred()


async def _forever():
    await asyncio.Event().wait()


async def test_supervisor_rebuilds_dead_training_loops(tmp_path):
    """三条训练循环一启动就崩 → 保活监督必须把它们都重建起来。"""
    eng = _make_engine(tmp_path)

    async def _crash():
        raise RuntimeError("训练循环崩溃")

    _install_stub_loops(eng, _crash)
    eng._spawn_loops()
    eng._spawn_supervisor()
    try:
        ok = await _wait_until(
            lambda: all(eng._restarts[n] >= 1 for n in eng._loop_names))
        assert ok, f"保活未重建死掉的循环: {eng._restarts}"
        assert eng._supervise_status["restarts"] >= 3
        assert eng._supervise_status["alive"] is True
        assert eng._supervise_status["last_restart_name"] in eng._loop_names
        st = eng.status()
        assert st["supervisor"]["restarts_by_loop"]["evolve_factor_miner"] >= 1
        # 自愈计数同步回落到对应管线的状态里，前端可直接展示
        assert st["factor_miner"]["restarts"] >= 1
    finally:
        await eng.stop()


async def test_supervisor_reaps_finished_task_from_task_list(tmp_path):
    """反复重建不得让 _tasks 无限增长（否则 tasks_alive 会失真）。"""
    eng = _make_engine(tmp_path)

    async def _crash():
        raise RuntimeError("崩")

    _install_stub_loops(eng, _crash)
    eng._spawn_loops()
    eng._spawn_supervisor()
    try:
        ok = await _wait_until(lambda: eng._supervise_status["restarts"] >= 6)
        assert ok, "未发生足够次数的自愈，无法验证回收"
        # 3 条循环 + 1 个监督，已终结的旧任务必须被就地清掉
        assert len(eng._tasks) <= 4, f"_tasks 泄漏: {[t.get_name() for t in eng._tasks]}"
    finally:
        await eng.stop()


async def test_status_flags_degraded_when_a_loop_dies(tmp_path):
    """循环死掉但引擎仍 running —— 必须能被 status() 判为降级。"""
    eng = _make_engine(tmp_path)
    eng._supervise_interval = 5.0  # 拉长巡检，先观察"尚未自愈"的窗口

    _install_stub_loops(eng, _forever)
    eng._spawn_loops()
    try:
        st = eng.status()
        assert all(st["supervisor"]["loops_alive"].values())
        assert st["supervisor"]["degraded"] is False
        assert st["supervisor"]["expected_running"] is True

        # 模拟一条循环静默死亡（异常逸出循环体，此前这就是永久停训）
        victim = eng._find_task("evolve_strategy_drl")
        victim.cancel()
        await asyncio.sleep(0.02)

        st = eng.status()
        assert st["running"] is True, "引擎自身状态不变——这正是旧实现的假象"
        assert st["supervisor"]["loops_alive"]["evolve_strategy_drl"] is False
        assert st["supervisor"]["degraded"] is True, "引擎开着但已停训必须判为降级"
    finally:
        await eng.stop()


async def test_spawn_loops_fills_missing_pipeline_without_duplicating(tmp_path):
    """热切换/巡检路径必须「缺哪条补哪条」，且不重复创建存活的循环。"""
    eng = _make_engine(tmp_path)

    _install_stub_loops(eng, _forever)
    try:
        assert eng._spawn_loops() == 3
        assert eng._spawn_loops() == 0, "已存活的循环不应被重复创建"
        # P4-A2 的历史形态：只死掉元策略一条
        victim = eng._find_task("evolve_meta_controller")
        victim.cancel()
        await asyncio.sleep(0.02)
        assert eng._spawn_loops() == 1, "只应补齐缺失的那一条"
        names = [t.get_name() for t in eng._tasks]
        assert names.count("evolve_meta_controller") == 1
    finally:
        await eng.stop()


async def test_enable_refills_loops_even_when_supervisor_task_present(tmp_path):
    """_tasks 非空不等于循环齐全——监督任务占槽不得让热切换漏建整组循环。"""
    eng = _make_engine(tmp_path)
    _install_stub_loops(eng, _forever)
    try:
        eng._spawn_supervisor()          # 只有监督，没有任何训练循环
        assert eng._tasks, "_tasks 非空"
        assert not any(eng._find_task(n) for n in eng._loop_names)

        eng._enabled = False             # 先置为禁用，再热切换开启
        await eng.set_enabled(True)
        for name in eng._loop_names:
            t = eng._find_task(name)
            assert t is not None and not t.done(), f"{name} 未被热切换补齐"
    finally:
        await eng.stop()


async def test_run_loop_survives_non_exception_base_error(tmp_path):
    """循环体抛出非 Exception 基类错误时不得终结任务（此前会静默停训）。"""

    class _Weird(BaseException):
        pass

    eng = _make_engine(tmp_path)
    eng._RETRY_INTERVAL = 0.4  # 失败退避窗口，给断言留出观测时间
    calls = {"n": 0}
    seen = {"err": ""}

    async def _train():
        calls["n"] += 1
        if calls["n"] == 1:
            raise _Weird("非 Exception 基类错误")
        return True

    status = {"active": False, "last_error": "", "last_success": False,
              "_demo_streak": 0, "loop_beat": 0.0}
    task = asyncio.create_task(
        eng._run_loop(status, _train, 0.01, "测试", start_delay=0))

    async def _sample_until_second_call() -> bool:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 5.0
        while loop.time() < deadline:
            # 下一轮开始时 last_error 会被清空，须在窗口内采样
            if status["last_error"] and not seen["err"]:
                seen["err"] = status["last_error"]
            if calls["n"] >= 2:
                return True
            await asyncio.sleep(0.02)
        return False

    try:
        assert await _sample_until_second_call(), "非 Exception 异常后循环应继续，而不是静默死亡"
        assert "非 Exception" in seen["err"], f"未记录兜底错误: {seen['err']!r}"
        assert status["loop_beat"] > 0, "心跳时间戳未更新"
        assert not task.done(), "任务不应因此结束"
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_stop_cancels_supervisor_and_clears_tasks(tmp_path):
    """stop() 必须连保活监督一起收掉，且清空任务列表（不残留）。"""
    eng = _make_engine(tmp_path)
    _install_stub_loops(eng, _forever)
    eng._spawn_loops()
    eng._spawn_supervisor()
    assert len(eng._tasks) == 4

    await eng.stop()
    assert eng._tasks == []
    assert eng._supervise_status["alive"] is False
    assert eng._running is False
