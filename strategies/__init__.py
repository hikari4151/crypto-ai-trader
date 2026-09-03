"""策略注册表：内置策略 + AI 动态设计策略。"""
from typing import Optional
from .base import Signal, Strategy
from .dual_ma import DualMAStrategy
from .factor_signal import FactorSignalStrategy
from .grid import GridStrategy
from .price_action import PriceActionStrategy
from .rl_adaptive import RLAdaptiveStrategy
from .meta import MetaController

_REGISTRY: dict[str, type[Strategy]] = {
    DualMAStrategy.name: DualMAStrategy,
    GridStrategy.name: GridStrategy,
    PriceActionStrategy.name: PriceActionStrategy,
    RLAdaptiveStrategy.name: RLAdaptiveStrategy,
    FactorSignalStrategy.name: FactorSignalStrategy,
    MetaController.name: MetaController,
}
_DYNAMIC: dict[str, dict] = {}   # AI 设计策略 -> {name, title, description, logic, params, executor}


def register_dynamic(name: str, spec: dict) -> None:
    """注册动态策略。spec 可含 executor（默认 price_action）指定执行器类，
    param_schema / default_params 用于让前端参数面板正确渲染对应策略的参数。

    向后兼容：旧 DRL 策略记录没有 executor 字段，通过名字前缀 rl_ 或
    params 含 model_path 自动推断为 rl_adaptive 执行器。
    """
    spec = dict(spec)
    spec["name"] = name
    if "executor" not in spec:
        params = spec.get("params") or {}
        if name.startswith("rl_") or "model_path" in params:
            spec["executor"] = "rl_adaptive"
        else:
            spec["executor"] = "price_action"
    # 显式 schema / params 缺失时回退到执行器类自带
    executor = spec["executor"]
    if executor in _REGISTRY:
        spec.setdefault("param_schema", _REGISTRY[executor].param_schema)
        spec.setdefault("params", dict(_REGISTRY[executor].default_params))
    else:
        spec.setdefault("param_schema", {})
        spec.setdefault("params", {})
    _DYNAMIC[name] = spec


def get_strategy(name: str) -> Strategy:
    # 动态策略优先：AI 迭代/设计会覆盖内置同名策略（如迭代 dual_ma 后参数应生效）
    if name in _DYNAMIC:
        executor = _DYNAMIC[name].get("executor", "price_action")
        base = _REGISTRY.get(executor)
        if base is None:
            raise ValueError(f"动态策略 {name} 的 executor 无效: {executor}")
        st = base()
        st.name = name
        st.description = _DYNAMIC[name].get("description", st.description)
        st.update_params(_DYNAMIC[name].get("params", {}))
        return st
    if name in _REGISTRY:
        return _REGISTRY[name]()
    raise ValueError(f"未知策略: {name}，可用: {[s['name'] for s in list_strategies()]}")


def list_strategies() -> list[dict]:
    """列出全部可用策略：内置 + 动态（AI 设计/迭代/DRL/仓库）。

    同名动态策略覆盖内置项（与 get_strategy 的优先级一致），故跳过被覆盖的内置条目
    ——否则前端下拉会出现两个同名 option、表格出现重复 key、策略对比重复计一次。
    """
    out = [
        {"name": cls.name, "description": cls.description,
         "default_params": dict(cls.default_params), "param_schema": cls.param_schema,
         "source": "builtin", "builtin": True}
        for cls in _REGISTRY.values() if cls.name not in _DYNAMIC
    ]
    for name, spec in _DYNAMIC.items():
        out.append({
            "name": name, "description": spec.get("description", ""),
            "default_params": dict(spec.get("params", {})),
            "param_schema": dict(spec.get("param_schema", {})),
            "executor": spec.get("executor", "price_action"),
            "title": spec.get("title", ""),
            "logic": spec.get("logic", ""),
            "risk_tips": spec.get("risk_tips", []),
            "version": spec.get("version", ""),
            "iter_no": spec.get("iter_no"),
            "based_on": spec.get("based_on", ""),
            "created_by": spec.get("created_by", ""),
            "overfit": spec.get("overfit"),
            "blocked_by_overfit": spec.get("blocked_by_overfit", False),
            "model_path": (spec.get("params") or {}).get("model_path", ""),
            "source": "ai",
            "ai_designed": True,
        })
    return out


def get_dynamic(name: str) -> Optional[dict]:
    return _DYNAMIC.get(name)


def rename_dynamic(old_name: str, new_name: str) -> bool:
    """重命名动态策略（内置策略不可改名）。返回是否成功。"""
    if old_name not in _DYNAMIC:
        return False
    if new_name in _REGISTRY or new_name in _DYNAMIC:
        raise ValueError(f"策略名 {new_name} 已被占用")
    spec = _DYNAMIC.pop(old_name)
    spec["name"] = new_name
    _DYNAMIC[new_name] = spec
    return True


def remove_dynamic(name: str) -> bool:
    """删除动态策略（内置策略不可删除）。返回是否成功。"""
    if name in _DYNAMIC:
        _DYNAMIC.pop(name, None)
        return True
    return False


def dynamic_names() -> list[str]:
    return list(_DYNAMIC)