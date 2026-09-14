"""策略仓库 API：内置模板浏览 + 全策略管理中心（版本号/删除/改名/应用）。

所有在本软件生成的策略（内置 / AI 设计 / AI 迭代 / DRL 训练 / 仓库模板）
都能在这里看到、改名或删除。内置策略受保护不可删除。
"""
import logging
import re

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from strategies.repository import get_by_key, list_repository
from strategies.pine_utils import PINE_TEMPLATE_EXECUTORS, resolve_pine
from strategies import (dynamic_names, get_dynamic, get_strategy, list_strategies,
                        register_dynamic, remove_dynamic, rename_dynamic)
from web.deps import get_db, get_engine

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/strategy-repo", tags=["strategy-repo"])


class ApplyIn(BaseModel):
    key: str


class RenameIn(BaseModel):
    old_name: str
    new_name: str


class SaveParamsIn(BaseModel):
    name: str
    params: dict


class DeleteIn(BaseModel):
    name: str


class ImportPineIn(BaseModel):
    """从 TradingView 导入 Pine Script 参数（与前端 buildPine 导出的格式互逆）。"""
    pine_code: str
    name: str = ""            # 可选自定义策略名；缺省取 shorttitle 或 pine_import
    ai_check: bool = False    # 可选：调用户配置的 AI 做参数合理性审查（只警告不阻塞）


# Pine 变量名 → price_action 参数（与前端 buildPine 生成的 input.* 一一对应，
# 见 strategies/pine_utils.py）

@router.post("/import-pine")
async def import_pine(body: ImportPineIn, db=Depends(get_db), engine=Depends(get_engine)):
    """📥 从 TradingView Pine Script 导入策略参数（与工坊导出的 Pine 闭环）。

    流程：本地 regex 解析 input.* → price_action schema 校验/clamp → 注册动态策略并落库。
    ai_check=True 时调用用户配置的 AI 做合理性审查（返回 warnings，不阻塞导入）。
    """
    from strategies.pine_utils import parse_pine_params, pine_shorttitle
    code = (body.pine_code or "").strip()
    if not code:
        raise HTTPException(status_code=400, detail="Pine 代码不能为空")
    params = parse_pine_params(code)
    if not params:
        raise HTTPException(status_code=400, detail="未从 Pine 代码中识别出任何参数；"
                            "请粘贴本软件导出格式的 Pine（含 input.* 参数行）")
    # schema 校验与 clamp（复用执行器 update_params，与 AI 设计策略同一条防线）
    from strategies.price_action import PriceActionStrategy
    stub = PriceActionStrategy()
    applied = stub.update_params(params)
    if not applied:
        raise HTTPException(status_code=400, detail="参数校验失败")

    name = _sanitize_name(body.name or pine_shorttitle(code) or "pine_import")
    if not name or name.isspace():
        name = "pine_import"
    if name in ("dual_ma", "grid", "price_action"):
        name = "pine_" + name
    # 保留用户导入的原始代码（缺成交标记时补齐）：此前只回填参数不存代码，
    # 前端于是套通用模板显示——图上的买卖点不是用户那份策略的
    from strategies.pine_utils import ensure_trade_markers
    spec = {
        "name": name, "title": "Pine 导入策略",
        "description": f"从 TradingView Pine Script 导入（{len(applied)} 个参数）",
        "logic": "参数来自 TradingView 调优后的 Pine 代码回填",
        "params": applied, "risk_tips": [], "created_by": "pine_import",
        "pine_code": ensure_trade_markers(code),
    }
    register_dynamic(name, spec)
    from core.database import AiStrategy
    from sqlalchemy import select as _select
    import json as _json
    async with db.session() as s:
        existing = (await s.execute(
            _select(AiStrategy).where(AiStrategy.name == name)
        )).scalar_one_or_none()
        if existing:
            existing.spec_json = _json.dumps(spec, ensure_ascii=False)
        else:
            s.add(AiStrategy(name=name, spec_json=_json.dumps(spec, ensure_ascii=False)))
        await s.commit()
    log.info("[strategy-repo] Pine 导入成功: %s (params=%s)", name, applied)

    warnings: list[str] = []
    if body.ai_check:
        try:
            from ai.prompts import validate_pine_messages
            result = await engine.ai_client.chat_json_validated(
                validate_pine_messages(code[:4000], applied), feature="market_analysis")
            if isinstance(result, dict):
                warnings = [str(w) for w in (result.get("warnings") or [])][:6]
        except Exception as e:  # noqa: BLE001
            warnings = [f"AI 校验未完成（不影响导入）: {e}"]
    return {"ok": True, "strategy": spec, "parsed_params": applied, "warnings": warnings}


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
    """全策略列表：内置 + 动态（含版本号/来源/类型），供「全部策略」页与统一策略选择器使用。

    恢复动态策略与 Web 启动、引擎启动共用 dynamic_store.restore_from_db，
    保证任何页面看到的策略集合一致（曾只有本端点恢复 → 页面访问顺序决定可用性）。
    """
    from strategies.dynamic_store import restore_from_db
    await restore_from_db(db)

    # 内置策略必须展示：前端「内置」筛选 chip、kind==='builtin' 徽标、统一选择器的
    # 子策略候选（dual_ma/price_action/factor_signal/grid）都依赖它们出现在列表里
    builtin = [{
        "key": s["name"], "name": s["name"], "title": _guess_title(s["name"]),
        "description": s.get("description", ""), "category": _guess_category(s["name"]),
        "default_params": s.get("default_params", {}), "param_schema": s.get("param_schema", {}),
        "source": "内置", "kind": "builtin", "executor": s["name"],
        "version": "", "iter_no": None, "based_on": "",
        "overfit": None, "blocked_by_overfit": False,
        # L2：内置策略恒为 active（草稿只可能出现在 AI 产物上），补齐字段让前端
        # 不必对两类条目做分支判断
        "status": "active", "draft": False, "draft_reason": "",
        "overfit_inconclusive": False,
        "can_edit": False,
        "has_pine": s["name"] in PINE_TEMPLATE_EXECUTORS,
        "pine_note": "" if s["name"] in PINE_TEMPLATE_EXECUTORS
                     else f"内置策略 {s['name']} 无等价 Pine 模板",
    } for s in list_strategies() if s.get("builtin")]
    dynamic = []
    for s in list_strategies():
        if s.get("builtin"):
            continue
        created_by = s.get("created_by", "")
        if created_by == "repository":
            kind, source = "模板", "仓库模板"
        elif created_by == "evolve_engine":
            # 持续进化引擎训练产物（rl_evolve / meta_controller）：此前落入
            # "AI设计" 兜底桶，前端「进化」徽标与筛选永远不亮
            kind, source = "进化", "持续进化"
        elif created_by == "drl_train":
            kind, source = "RL", "DRL训练"
        elif created_by == "ai_iteration":
            kind, source = "迭代", "AI迭代"
        else:
            kind, source = "AI", "AI设计"
        iter_no = s.get("iter_no")
        # 完整 spec（含 pine_code / pine_note）只在注册表里，list_strategies 是精简视图
        full = get_dynamic(s["name"]) or {}
        executor = s.get("executor", "price_action")
        dynamic.append({
            "key": s["name"], "name": s["name"],
            "title": s.get("title") or _guess_title(s["name"]),
            "description": s.get("description", ""),
            "category": _guess_category(s["name"]),
            "default_params": s.get("default_params", {}),
            "param_schema": s.get("param_schema", {}),
            "source": source, "kind": kind,
            "executor": executor,
            "version": s.get("version") or "", "iter_no": iter_no,
            "based_on": s.get("based_on", ""),
            "overfit": s.get("overfit"), "blocked_by_overfit": s.get("blocked_by_overfit", False),
            # L2：草稿（未证明）标记——过拟合未过 / 样本外证据不足的产物。
            # 前端据此分区显示与二次确认；后端已禁止它当迭代父代、也不让 AI 接管调参。
            "status": s.get("status", "active"), "draft": bool(s.get("draft")),
            "draft_reason": s.get("draft_reason", ""),
            "overfit_inconclusive": s.get("overfit_inconclusive", False),
            "is_evolve": kind == "进化" or s["name"] == "rl_evolve",
            "can_edit": True,
            # 代码本体不内嵌：一份 DRL 权重脚本约 44KB，几十条列表会撑爆响应，
            # 前端按 has_pine 决定走 GET /pine/{name}，按 pine_note 显示原因
            "has_pine": bool(full.get("pine_code")) or executor in PINE_TEMPLATE_EXECUTORS,
            "pine_note": "" if full.get("pine_code") or executor in PINE_TEMPLATE_EXECUTORS
                         else full.get("pine_note", ""),
        })
    return {"strategies": builtin + dynamic, "count": len(builtin) + len(dynamic)}


@router.get("/pine/{name}")
async def repo_pine(name: str, db=Depends(get_db)):
    """单个策略的真实 Pine 代码（贴到图表必须给出买卖点，见 strategies/pine_utils）。

    曾经前端只有列表数据，于是对任何策略都套 price_action 模板生成代码：
    进化/RL 策略显示的买卖点与其真实逻辑毫无关系。
    """
    from strategies.dynamic_store import restore_from_db
    await restore_from_db(db)
    spec = get_dynamic(name)
    if spec is None:
        builtin = next((s for s in list_strategies()
                        if s.get("builtin") and s["name"] == name), None)
        if builtin is None:
            raise HTTPException(status_code=404, detail=f"策略 {name} 不存在")
        spec = {"name": name, "executor": name,
                "params": builtin.get("default_params") or {}}
    code, note = resolve_pine(spec)
    return {"name": name, "pine_code": code, "pine_note": note}


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
            await engine.select_strategy(t["name"])
        except ValueError:
            from strategies.repository import register_repository_strategies
            register_repository_strategies()
            await engine.select_strategy(t["name"])
        return {"ok": True, "strategy": t["name"], "params": t["default_params"]}
    except Exception as e:  # noqa: BLE001
        log.exception("[strategy-repo] 应用策略失败")
        raise HTTPException(status_code=400, detail=f"应用策略失败: {e}")


@router.post("/params")
async def save_params(body: SaveParamsIn, db=Depends(get_db), engine=Depends(get_engine)):
    """保存动态策略参数：schema 校验 → 写回注册表 spec → 落库 → 当前策略热应用。

    曾因缺少本端点，「统一策略」构建的 sub_strategies/mode 只能热更新当前实例，
    重启或重选策略后即丢失（前端甚至退化到把参数拼好后直接丢弃）。
    """
    name = body.name.strip()
    spec = get_dynamic(name)
    if spec is None:
        raise HTTPException(status_code=400,
                            detail=f"策略 {name} 不是动态策略；内置策略请在实盘页热更新参数")
    if not isinstance(body.params, dict) or not body.params:
        raise HTTPException(status_code=400, detail="params 不能为空")

    # 与 Pine 导入同一防线：交给执行器实例按 param_schema 校验类型/范围/枚举
    try:
        stub = get_strategy(name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    unknown = [k for k in body.params if k not in stub.param_schema]
    if len(unknown) == len(body.params):
        raise HTTPException(status_code=400,
                            detail=f"没有参数在 {name} 的 schema 内，可用参数: {sorted(stub.param_schema)}")
    validated = stub.update_params(body.params)

    merged = dict(spec.get("params") or {})
    merged.update({k: validated[k] for k in validated})
    new_spec = dict(spec)
    new_spec["params"] = merged
    register_dynamic(name, new_spec)

    import json as _json
    from core.database import AiStrategy
    from sqlalchemy import select
    async with db.session() as s:
        row = (await s.execute(
            select(AiStrategy).where(AiStrategy.name == name))).scalar_one_or_none()
        if row:
            row.spec_json = _json.dumps(new_spec, ensure_ascii=False)
        else:
            s.add(AiStrategy(name=name, spec_json=_json.dumps(new_spec, ensure_ascii=False)))
        await s.commit()

    live = engine.strategy.name == name
    if live:
        await engine.apply_strategy_params(name, merged)
    log.info("[strategy-repo] 参数已保存: %s (%s)%s", name, merged, "，已热应用" if live else "")
    return {"ok": True, "name": name, "params": merged, "applied_live": live,
            "ignored_params": unknown}


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
            await engine.select_strategy(new)
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
