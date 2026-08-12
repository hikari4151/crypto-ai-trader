"""策略仓库 API：内置模板浏览 + 全策略管理中心（版本号/删除/改名/应用）。

所有在本软件生成的策略（内置 / AI 设计 / AI 迭代 / DRL 训练 / 仓库模板）
都能在这里看到、改名或删除。内置策略受保护不可删除。
"""
import logging
import re

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from strategies.repository import get_by_key, list_repository
from strategies import (dynamic_names, get_strategy, list_strategies,
                        register_dynamic, remove_dynamic, rename_dynamic)
from web.deps import get_db, get_engine

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/strategy-repo", tags=["strategy-repo"])


class ApplyIn(BaseModel):
    key: str


class RenameIn(BaseModel):
    old_name: str
    new_name: str


class DeleteIn(BaseModel):
    name: str


def _sanitize_name(name: str) -> str:
    name = re.sub(r"[^a-z0-9_]", "_", (name or "").lower().strip())
    return name[:48]


@router.get("")
async def repo_list():
    """返回内置策略仓库模板（含分类与参数）。"""
    return {"categories": [
        {"key": "trend", "label": "趋势跟踪", "icon": "📈"},
        {"key": "breakout", "label": "突破交易", "icon": "⚡"},
        {"key": "mean_reversion", "label": "均值回归", "icon": "🎯"},
        {"key": "grid", "label": "网格震荡", "icon": "🔲"},
    ], "strategies": list_repository()}


@router.get("/all")
async def repo_all(db=Depends(get_db)):
    """全策略列表：内置 + 动态（含版本号/来源/类型），供前端策略仓库展示。

    动态策略从 DB（AiStrategy 表）恢复合并到内存注册表，确保引擎未启动时
    也能看到所有 AI 设计/迭代/DRL 训练的策略。
    """
    from strategies import register_dynamic
    # 从 DB 恢复动态策略（幂等，合并进内存注册表）
    try:
        from core.database import AiStrategy
        from sqlalchemy import select
        async with db.session() as s:
            rows = (await s.execute(select(AiStrategy))).scalars().all()
        for r in rows:
            try:
                import json as _json
                spec = _json.loads(r.spec_json or "{}")
                register_dynamic(r.name, spec)
            except Exception:  # noqa: BLE001
                continue
    except Exception:  # noqa: BLE001
        log.warning("[strategy-repo] 恢复动态策略失败", exc_info=True)

    builtin = [
        {"key": s["name"], "name": s["name"], "title": s["description"].split("：")[0][:20],
         "description": s["description"], "category": _guess_category(s["name"]),
         "default_params": s["default_params"], "source": "builtin",
         "version": "内置", "kind": "builtin", "can_edit": False}
        for s in list_strategies() if s.get("builtin")
    ]
    dynamic = []
    for s in list_strategies():
        if s.get("builtin"):
            continue
        created_by = s.get("created_by", "")
        if created_by == "repository":
            kind, source = "模板", "仓库模板"
        elif created_by == "drl_train":
            kind, source = "RL", "DRL训练"
        elif created_by == "ai_iteration":
            kind, source = "迭代", "AI迭代"
        else:
            kind, source = "AI", "AI设计"
        iter_no = s.get("iter_no")
        dynamic.append({
            "key": s["name"], "name": s["name"],
            "title": s.get("title") or _guess_title(s["name"]),
            "description": s.get("description", ""),
            "category": _guess_category(s["name"]),
            "default_params": s.get("default_params", {}),
            "source": source, "kind": kind,
            "version": s.get("version") or "", "iter_no": iter_no,
            "based_on": s.get("based_on", ""),
            "overfit": s.get("overfit"), "blocked_by_overfit": s.get("blocked_by_overfit", False),
            "can_edit": True,
        })
    return {"strategies": builtin + dynamic, "count": len(builtin) + len(dynamic)}


@router.get("/{key}")
async def repo_get(key: str):
    t = get_by_key(key)
    if not t:
        raise HTTPException(status_code=404, detail=f"策略模板 {key} 不存在")
    return t


@router.post("/apply")
async def repo_apply(body: ApplyIn, engine=Depends(get_engine)):
    """应用策略模板为当前策略（引擎运行中则热切换）。"""
    t = get_by_key(body.key)
    if not t:
        raise HTTPException(status_code=404, detail=f"策略模板 {body.key} 不存在")
    try:
        from strategies import get_strategy
        try:
            get_strategy(t["name"])  # 验证存在
            engine.select_strategy(t["name"])
        except ValueError:
            from strategies.repository import register_repository_strategies
            register_repository_strategies()
            engine.select_strategy(t["name"])
        return {"ok": True, "strategy": t["name"], "params": t["default_params"]}
    except Exception as e:  # noqa: BLE001
        log.exception("[strategy-repo] 应用策略失败")
        raise HTTPException(status_code=400, detail=f"应用策略失败: {e}")


@router.post("/rename")
async def repo_rename(body: RenameIn, db=Depends(get_db), engine=Depends(get_engine)):
    """重命名动态策略（内置策略不可改名）。"""
    old, new = body.old_name.strip(), _sanitize_name(body.new_name)
    if not old or old not in dynamic_names():
        raise HTTPException(status_code=404, detail=f"策略 {old} 不存在或为内置策略")
    if not new:
        raise HTTPException(status_code=400, detail="新名称不能为空")
    try:
        rename_dynamic(old, new)
        # 同步到 DB（AiStrategy 记录改名）
        from core.database import AiStrategy
        from sqlalchemy import select
        async with db.session() as s:
            row = (await s.execute(select(AiStrategy).where(AiStrategy.name == old))).scalar_one_or_none()
            if row:
                try:
                    import json as _json
                    spec = _json.loads(row.spec_json or "{}")
                    spec["name"] = new
                    row.spec_json = _json.dumps(spec, ensure_ascii=False)
                    row.name = new
                except Exception:  # noqa: BLE001
                    pass
                await s.commit()
        # 若当前正用该策略，引擎热切换
        if engine.strategy.name == old:
            engine.select_strategy(new)
        return {"ok": True, "old": old, "new": new}
    except Exception as e:  # noqa: BLE001
        log.exception("[strategy-repo] 改名失败")
        raise HTTPException(status_code=400, detail=f"改名失败: {e}")


@router.post("/delete")
async def repo_delete(body: DeleteIn, db=Depends(get_db), engine=Depends(get_engine)):
    """删除动态策略（内置策略不可删除；当前正在使用的不可删除）。"""
    name = body.name.strip()
    if not name or name not in dynamic_names():
        raise HTTPException(status_code=404, detail=f"策略 {name} 不存在或为内置策略")
    if engine.strategy.name == name:
        raise HTTPException(status_code=400, detail="当前正在使用该策略，请先切换到其他策略再删除")
    try:
        remove_dynamic(name)
        from core.database import AiStrategy
        from sqlalchemy import select, delete
        async with db.session() as s:
            await s.execute(delete(AiStrategy).where(AiStrategy.name == name))
            await s.commit()
        return {"ok": True, "deleted": name}
    except Exception as e:  # noqa: BLE001
        log.exception("[strategy-repo] 删除失败")
        raise HTTPException(status_code=400, detail=f"删除失败: {e}")


def _guess_category(name: str) -> str:
    from strategies.repository import get_by_key
    t = get_by_key(name)
    if t:
        return t["category"]
    if name.startswith(("rl_",)):
        return "drl"
    if "break" in name or "momentum" in name:
        return "trend"
    if "mean" in name or "revers" in name or "rsi" in name:
        return "mean_reversion"
    return "ai"


def _guess_title(name: str) -> str:
    return {"rl_adaptive": "RL自适应", "dual_ma": "双均线", "grid": "网格",
            "price_action": "关键位突破"}.get(name, name[:16])
