"""动态策略（AI 设计/迭代/DRL 训练/仓库模板）与数据库之间的恢复。

注册表 `_DYNAMIC` 是进程内全局状态，而策略规格持久化在 `ai_strategies` 表。
二者不同步会导致"策略可用性取决于用户先打开哪个页面"——本模块是唯一实现，
启动路径（web lifespan / engine.start / 策略仓库列表）都复用它。
"""
import logging

from . import register_dynamic

log = logging.getLogger(__name__)


async def restore_from_db(db) -> int:
    """把 DB 中的动态策略 spec 恢复进内存注册表（幂等，可反复调用）。

    单条 spec 解析失败只跳过该条，不影响其余策略——曾因一条坏记录使整个
    Web 服务起不来。返回成功恢复的策略数。
    """
    from core.database import AiStrategy
    from sqlalchemy import select

    try:
        async with db.session() as s:
            rows = (await s.execute(select(AiStrategy))).scalars().all()
    except Exception as e:  # noqa: BLE001
        log.warning("[dynamic_store] 读取动态策略失败（不影响内置策略）: %s", e)
        return 0

    import json
    count = 0
    for r in rows:
        try:
            spec = json.loads(r.spec_json or "{}")
            if not isinstance(spec, dict):
                continue
            register_dynamic(r.name, spec)
            count += 1
        except Exception as e:  # noqa: BLE001
            log.warning("[dynamic_store] 策略 %s 恢复失败: %s", getattr(r, "name", "?"), e)
    return count
