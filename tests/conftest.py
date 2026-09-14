"""pytest 全局配置：把项目根目录加入 sys.path，使 tests/ 下脚本可直接 import 项目模块。"""
import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 脚本式验证文件（顶层直接执行并 sys.exit），仅手动运行，禁止被 pytest 收集
collect_ignore = ["test_ai_guards.py"]


@pytest.fixture(autouse=True)
def _close_leaked_databases():
    """兜底关闭未显式 close 的 Database 实例（防 aiosqlite worker 线程崩溃）。

    部分既有测试在 asyncio.run() 内建内存库但不 close：asyncio.run 返回即关闭事件
    循环，而 aiosqlite 的 _connection_worker_thread 在连接关闭后仍会回调
    `future.get_loop().call_soon_threadsafe(...)` → RuntimeError('Event loop is closed')
    使 worker 线程崩溃（pytest 报 PytestUnhandledThreadExceptionWarning）。

    这里逐个用新事件循环 dispose 残留引擎，使 worker 队列在主循环关闭前排空。
    只做清理，不改动任何测试逻辑与断言。
    """
    yield
    from core import database as _db_mod
    leaked = [d for d in list(_db_mod._OPEN_DATABASES)]
    for db in leaked:
        try:
            asyncio.run(db.close())
        except Exception:  # noqa: BLE001  # 清理失败不得影响测试结果
            pass