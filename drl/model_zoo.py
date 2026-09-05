"""模型仓库：持久化训练好的智能体，支持版本管理和最佳回退。

保存格式：
  - 版本管理：data/models/<name>/ (best.json.gz, v1.json.gz, meta.json)
  - 兼容格式：data/models/<name>.json (与 web/api/drl.py 的 ACAgent.save() 同格式)

两种格式互通：最佳模型同时保存到两个路径，确保 EvolveEngine 和 DRL API 能互相看到对方的模型。
"""
import gzip
import json
import logging
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from .agent import ACAgent

log = logging.getLogger(__name__)


class ModelZooError(RuntimeError):
    """模型部署快照无法安全恢复。"""


_DEPLOYMENT_META_KEYS = (
    "train_meta", "factor_expression", "factor_mu", "factor_sd",
    "min_trade_zone", "pine_code", "pine_note", "cascade_factor",
    "cascade_factor_path", "cascade_weights_path",
)


# 默认保留的最大版本数（若调用方未传 max_versions 时使用）
_DEFAULT_MAX_VERSIONS = 10
# archive/ 目录每个模型的保留上限（P4-E1：拦截/demo 存档高频，防磁盘无界增长）
_MAX_ARCHIVES = 30


class ModelZoo:
    """模型仓库：持久化智能体，支持版本管理和最佳回退。"""

    def __init__(self, models_dir: Path, max_versions: Optional[int] = None) -> None:
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        # P4-A5：最大版本数改为可配置（此前硬编码模块常量 _MAX_VERSIONS=10，
        # settings.evolve_max_versions 是死配置，用户修改无效）
        self.max_versions = max(1, int(max_versions or _DEFAULT_MAX_VERSIONS))
        # 锚点 _meta 缓存：name -> ((best 文件 mtime_ns, size), meta dict)
        self._best_meta_cache: dict[str, tuple[tuple, dict]] = {}
        # P2：meta.json 缓存（status() 每次轮询对三条管线各读一次 meta.json，
        # 此前无缓存；与 _best_meta_cache 同款按文件身份 (mtime_ns, size) 键控，
        # 文件被重写后键自然变化。唯一写口 _write_meta 按名失效）。
        self._meta_cache: dict[str, tuple[tuple, dict]] = {}

    # ---- 路径管理 ----

    @staticmethod
    def _validate_name(name: str) -> str:
        name = str(name or "")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in (".", ".."):
            raise ValueError(f"非法模型名: {name!r}")
        return name

    def _model_dir(self, name: str) -> Path:
        """模型子目录，如 data/models/factor_miner/"""
        name = self._validate_name(name)
        root = self.models_dir.resolve()
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"非法模型路径: {name!r}")
        path.mkdir(exist_ok=True)
        return path

    def _validate_version(self, version: int) -> int:
        if isinstance(version, bool):
            raise ValueError(f"非法版本: {version!r}")
        try:
            version = int(version)
        except (TypeError, ValueError) as e:
            raise ValueError(f"非法版本: {version!r}") from e
        if version < 1:
            raise ValueError(f"非法版本: {version!r}")
        return version

    def _best_path(self, name: str) -> Path:
        return self._model_dir(name) / "best.json.gz"

    def _version_path(self, name: str, version: int) -> Path:
        return self._model_dir(name) / f"v{self._validate_version(version)}.json.gz"

    def _meta_path(self, name: str) -> Path:
        return self._model_dir(name) / "meta.json"

    def _flat_path(self, name: str) -> Path:
        """兼容格式路径：data/models/<name>.json（与 DRL API 同格式）。"""
        name = self._validate_name(name)
        root = self.models_dir.resolve()
        path = (root / f"{name}.json").resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"非法模型路径: {name!r}")
        return path

    # ---- 保存 ----

    @staticmethod
    def _flat_payload(data: dict) -> dict:
        """Return the deployment payload: no ModelZoo envelope keys, with
        deployment metadata promoted from the snapshot meta to top level."""
        flat = {k: v for k, v in data.items() if k not in ("_meta", "_version")}
        emeta = data.get("_meta") or {}
        for key in _DEPLOYMENT_META_KEYS:
            if key in emeta:
                flat[key] = emeta[key]
        return flat

    @staticmethod
    def _gzip_bytes(data: dict) -> bytes:
        return gzip.compress(
            json.dumps(data, ensure_ascii=False, default=str).encode("utf-8"),
            compresslevel=6,
        )

    @staticmethod
    def _json_bytes(data: dict) -> bytes:
        return json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")

    @staticmethod
    def _atomic_write_bytes(path: Path, payload: bytes) -> None:
        """Write bytes beside the destination and replace it atomically."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_name = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=path.parent, prefix=f".{path.name}.",
                suffix=".tmp", delete=False,
            ) as tmp:
                tmp_name = tmp.name
                tmp.write(payload)
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(tmp_name, path)
        except Exception:
            if tmp_name:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
            raise

    def _read_version_payload(self, name: str, version: int) -> dict:
        path = self._version_path(name, version)
        if not path.exists():
            raise FileNotFoundError(f"版本 v{version} 不存在")
        try:
            data = self._read_json_gz(path)
        except Exception as e:
            raise ModelZooError(f"版本 v{version} 损坏: {e}") from e
        if not isinstance(data, dict):
            raise ModelZooError(f"版本 v{version} 格式无效")
        return data

    def _snapshot_bytes(self, data: dict) -> tuple[bytes, bytes]:
        return self._gzip_bytes(data), self._json_bytes(self._flat_payload(data))

    def save_agent(self, agent: ACAgent, name: str, *,
                   meta: Optional[dict] = None,
                   is_best: bool = True) -> int:
        """保存智能体到磁盘。

        Args:
            agent: 待保存的智能体。
            name: 模型名称，如 "factor_miner"、"strategy_agent"。
            meta: 附加元数据（fitness、时间戳等）。
            is_best: 是否同时设为最佳版本。

        Returns:
            版本号（递增）。
        """
        meta = dict(meta or {})
        meta.setdefault("timestamp", time.time())

        # 读取当前版本号
        current_meta = self._load_meta(name)
        version = current_meta.get("next_version", 1)

        # 保存到版本文件
        data = agent.to_dict()
        data["_meta"] = meta
        data["_version"] = version
        version_path = self._version_path(name, version)

        # 更新元数据（先持久化 next_version 递增，再写版本文件——P2：崩溃窗口
        # 修复。此前先写版本文件后写 meta，崩溃在中间会让 next_version 未递增，
        # 下一轮保存以同号覆盖刚写好的 vN（有效模型丢失）。先落 meta 后崩溃的
        # 最坏结果是 versions 列表多一条无文件的幽灵记录，但绝不覆盖已有版本）
        current_meta["next_version"] = version + 1
        versions = current_meta.setdefault("versions", [])
        versions.append({
            "version": version,
            "timestamp": meta.get("timestamp", time.time()),
            "meta": {k: v for k, v in meta.items() if isinstance(v, (int, float, str, bool))},
        })
        # 限制版本数量（P4-A5：读取实例配置而非硬编码常量）
        # P4-E1：裁剪不许删掉 best 锚点指向的版本——否则 best_version 悬空。
        # 先选出要删除的旧版本，再写入新版本，避免 max_versions=1 删掉刚保存的 best。
        if len(versions) > self.max_versions:
            best_v = current_meta.get("best_version", 0)
            removable = next((v for v in versions if v["version"] != best_v), None)
            if removable is not None:
                versions.remove(removable)
                old_path = self._version_path(name, removable["version"])
                try:
                    if old_path.exists():
                        old_path.unlink()
                except Exception as e:  # noqa: BLE001
                    log.warning("[model_zoo] 版本文件删除失败 %s: %s", old_path.name, e)
            else:
                # 新版本将成为 best 时，保留它；否则最多保留 best 这一份历史版本。
                versions.pop(0)
        # 版本记录先落盘：让 next_version 的递增持久化（防崩溃窗口覆盖）
        self._write_meta(name, current_meta)
        self._write_json_gz(version_path, data)

        if is_best:
            # 保存为最佳版本（压缩格式，含版本管理元数据）
            self._write_json_gz(self._best_path(name), data)
            # 同时保存为兼容格式（与 DRL API 保存的 flat JSON 同格式）
            self._write_flat_json(self._flat_path(name), self._flat_payload(data))
            current_meta["best_version"] = version
            current_meta["best_fitness"] = meta.get("fitness")
            self._write_meta(name, current_meta)

            log.info("[model_zoo] 模型 %s v%d 设为最佳 (fitness=%s)", name, version, meta.get("fitness"))

        log.info("[model_zoo] 模型 %s v%d 已保存", name, version)
        return version

    def save_agent_best(self, agent: ACAgent, name: str, *,
                        meta: Optional[dict] = None) -> int:
        """仅保存为最佳版本（不递增版本号，覆盖旧的最佳文件）。

        Prefer the deployed version payload so compatibility metadata is never
        rebuilt from the bare agent object during a rollback.
        """
        meta = dict(meta or {})
        meta.setdefault("timestamp", time.time())
        current_meta = self._load_meta(name)
        _, best_file_version = self._best_head(name)
        version = best_file_version or current_meta.get("best_version") or 1
        try:
            data = self._read_version_payload(name, version)
        except (FileNotFoundError, ModelZooError):
            data = agent.to_dict()
        prev_meta = dict(data.get("_meta") or self._best_meta(name))
        merged = {**prev_meta, **meta}
        if prev_meta.get("timestamp"):
            merged["timestamp"] = prev_meta["timestamp"]
            merged["rolled_back_at"] = meta["timestamp"]
        data["_meta"] = merged
        data["_version"] = version
        self._write_deployment_files(name, data)
        current_meta["best_version"] = version
        current_meta["best_fitness"] = merged.get("fitness")
        self._write_meta(name, current_meta)
        log.info("[model_zoo] 模型 %s 最佳已更新 (fitness=%s)", name, merged.get("fitness"))
        return version

    def _write_deployment_files(self, name: str, data: dict) -> None:
        """Atomically replace best and flat files from one payload."""
        best_path = self._best_path(name)
        flat_path = self._flat_path(name)
        old_best = best_path.read_bytes() if best_path.exists() else None
        old_flat = flat_path.read_bytes() if flat_path.exists() else None
        try:
            self._write_json_gz(best_path, data)
            self._write_flat_json(flat_path, self._flat_payload(data))
        except Exception as e:
            try:
                self._restore_file(best_path, old_best)
                self._restore_file(flat_path, old_flat)
            except Exception as restore_error:
                log.error("[model_zoo] 部署文件回滚失败(%s): %s", name, restore_error)
            raise ModelZooError(f"部署文件写入失败: {e}") from e

    def _restore_file(self, path: Path, payload: Optional[bytes]) -> None:
        if payload is None:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        else:
            self._atomic_write_bytes(path, payload)

    def restore_deployment(self, name: str, version: int, *,
                           reason: str = "", outcome: str = "rollback") -> dict:
        """Restore a complete version payload as the deployed best snapshot."""
        data = self._read_version_payload(name, version)
        meta = dict(data.get("_meta") or {})
        now = time.time()
        meta.setdefault("timestamp", now)
        meta["rolled_back_at"] = now
        if reason:
            meta["rollback_reason"] = reason
        if outcome:
            meta["rollback_outcome"] = outcome
        data["_meta"] = meta
        data["_version"] = self._validate_version(version)

        meta_path = self._meta_path(name)
        old_meta = meta_path.read_bytes() if meta_path.exists() else None
        current_meta = self._load_meta(name)
        current_meta["best_version"] = data["_version"]
        current_meta["best_fitness"] = meta.get("fitness")
        try:
            self._write_deployment_files(name, data)
            self._write_meta(name, current_meta)
        except Exception as e:
            try:
                if old_meta is None:
                    meta_path.unlink(missing_ok=True)
                else:
                    self._atomic_write_bytes(meta_path, old_meta)
            except Exception as restore_error:
                log.error("[model_zoo] 锚点元数据回滚失败(%s): %s", name, restore_error)
            raise ModelZooError(f"部署快照恢复失败: {e}") from e
        return {"version": data["_version"], "meta": meta,
                "best_path": str(self._best_path(name)),
                "flat_path": str(self._flat_path(name))}

    def save_agent_archive(self, agent: ACAgent, name: str, *,
                           meta: Optional[dict] = None,
                           max_archives: Optional[int] = None) -> int:
        """仅存档不递增版本号（P4-C2）。

        被 OOS 拦截/跨标拦截/demo 兜底的"废模型"此前也走 save_agent(is_best=False)，
        每个都递增版本号占名额——高失败率环境下版本历史很快被低质量模型充斥，
        真正的改进版被挤出（max_versions 限制）。本方法把这类模型存到独立
        archive/ 子目录（带时间戳），不写入 versions 列表、不占名额，
        同时保留排查证据。

        P4-E1：archive/ 目录增加保留上限（默认 _MAX_ARCHIVES），并给文件名加
        亚秒后缀防同秒覆盖——磁盘无界增长与同秒撞名都是此前实测过的风险。
        """
        meta = dict(meta or {})
        meta.setdefault("timestamp", time.time())
        arch_dir = self._model_dir(name) / "archive"
        arch_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        ms = int((time.time() % 1) * 1000)
        data = agent.to_dict()
        data["_meta"] = meta
        data["_archived"] = True
        # 亚秒后缀：同模型 1 秒内两次存档（手动触发+调度重叠）不再互相覆盖
        path = arch_dir / f"v{stamp}_{ms:03d}.json.gz"
        n = 1
        while path.exists() and n < 100:
            path = arch_dir / f"v{stamp}_{ms:03d}_{n}.json.gz"
            n += 1
        self._write_json_gz(path, data)
        # 保留上限：只留最近 max_archives 份（时间戳前缀字典序=时间序）
        cap = max(1, int(max_archives or _MAX_ARCHIVES))
        try:
            existing = sorted(arch_dir.glob("v*.json.gz"))
            for stale in existing[:-cap]:
                stale.unlink(missing_ok=True)
        except Exception as e:  # noqa: BLE001
            log.warning("[model_zoo] 归档裁剪失败（不阻断存档）: %s", e)
        log.info("[model_zoo] 模型 %s 已存档（不占版本名额）: %s (meta=%s)",
                 name, path.name,
                 {k: v for k, v in meta.items() if isinstance(v, (int, float, str, bool))})
        return 0

    # ---- 加载 ----

    def load_agent(self, name: str, version: Optional[int] = None) -> Optional[ACAgent]:
        """加载智能体。

        Args:
            name: 模型名称。
            version: 版本号。None 表示加载最佳版本。

        Returns:
            ACAgent 实例，或 None（不存在时）。
        """
        if version is None:
            # 优先加载最佳版本（压缩格式）
            path = self._best_path(name)
            if path.exists():
                try:
                    data = self._read_json_gz(path)
                    data.pop("_meta", None)
                    data.pop("_version", None)
                    return ACAgent.from_dict(data)
                except Exception as e:
                    log.warning("[model_zoo] 加载压缩最佳模型 %s 失败: %s", name, e)
            # 兜底：加载兼容格式（DRL API 保存的 flat JSON）
            flat = self._flat_path(name)
            if flat.exists():
                try:
                    data = json.loads(flat.read_text(encoding="utf-8"))
                    return ACAgent.from_dict(data)
                except Exception as e:
                    log.warning("[model_zoo] 加载兼容格式模型 %s 失败: %s", name, e)
            return None
        else:
            path = self._version_path(name, version)
            if not path.exists():
                return None
            try:
                data = self._read_json_gz(path)
                data.pop("_meta", None)
                data.pop("_version", None)
                return ACAgent.from_dict(data)
            except Exception as e:
                log.warning("[model_zoo] 加载模型 %s v%d 失败: %s", name, version, e)
                return None

    # ---- 查询 ----

    def best_version(self, name: str) -> int:
        """返回当前最佳版本号。无模型时返回 0。"""
        return self._load_meta(name).get("best_version", 0)

    def best_fitness(self, name: str) -> Optional[float]:
        """返回当前最佳 fitness。"""
        return self._load_meta(name).get("best_fitness")

    def best_meta_value(self, name: str, key: str):
        """返回当前最佳版本 meta 中指定键的值（无则 None）。

        P1-6：回退/部署判据统一用 OOS 收益——读取最佳版本的 oos_ret 与当前轮比较。
        """
        meta = self._load_meta(name)
        best_v = meta.get("best_version")
        if best_v is None:
            return None
        for v in meta.get("versions", []):
            if v.get("version") == best_v:
                return v.get("meta", {}).get(key)
        return None

    def list_versions(self, name: str) -> list[dict]:
        """列出所有版本及其元数据。"""
        return self._load_meta(name).get("versions", [])

    def best_info(self, name: str) -> dict:
        """当前锚点（回退基线）的完整画像：fitness、口径、来源、时间。

        必须从 best.json.gz 自带的 _meta 读，不能查 versions[best_version]：
        save_agent_best（回退路径）只覆盖 best 文件、不递增也不写入 versions，
        用 versions 反查会读到"最近一轮被拦截模型"的 meta。实测 meta_controller
        就是这样错配的——best_fitness=0.016223 却指向 fitness=0.0 的 v289。

        comparable：能否与新一轮训练结果直接比较。demo（合成数据）锚点的 fitness
        与真实交易所数据不同量级（合成数据能跑到 1267，真实数据 0.3 已算不错），
        拿它当基线会让进化循环永远 rollback。data_source 缺失的老锚点按可比处理
        （保留回退保护），只打 source_unknown 标记供前端提示。
        """
        meta = self._load_meta(name)
        best_version = meta.get("best_version") or 0
        anchor_meta, deployed_version = self._best_head(name)
        if not anchor_meta.get("data_source"):
            # 老代码的回退写盘把 data_source 抹掉了（现场 strategy_drl 就是空）。
            # 只在缺口径时回查版本存档，且只补来源类字段：fitness 仍以 best 为准，
            # 否则又会读到"最近一轮被存档模型"的分数。版本优先问 best 文件自己
            # （重置锚点后 meta.json 的 best_version 已归零，在线权重却还在跑）。
            probe = deployed_version or best_version
            for v in meta.get("versions", []):
                if v.get("version") == probe:
                    anchor_meta = {**(v.get("meta") or {}), **anchor_meta}
                    break
        info = {"fitness": meta.get("best_fitness"), "version": best_version,
                "data_source": str(anchor_meta.get("data_source") or ""),
                "timestamp": anchor_meta.get("timestamp"),
                "oos_ret": anchor_meta.get("oos_ret"),
                "rolled_back_at": anchor_meta.get("rolled_back_at"),
                "reset": meta.get("anchor_reset"),
                "comparable": False, "reason": "", "source_unknown": False}
        # 训练身份字段透传：_identity_mismatch 需要读锚点的符号/周期/配置指纹
        for _ik in ("model", "symbol", "timeframe", "state_window",
                    "factor_signature", "config_fingerprint", "window_tail_ts"):
            info[_ik] = anchor_meta.get(_ik)
        if not best_version and info["fitness"] is None:
            # 没有锚点：全新仓库，或 reset_anchor 之后。此时 best 文件里的权重仍然
            # 在线部署着，data_source 要照实返回——前端得知道"现在跑的是 demo 权重"。
            info["version"] = 0
            info["reason"] = "尚无锚点（首份通过检验的模型将建立基线）"
            return info
        if info["fitness"] is None:
            info["fitness"] = anchor_meta.get("fitness")
        if not info["data_source"]:
            info["source_unknown"] = True
        if info["data_source"] == "demo":
            info["reason"] = ("锚点由合成数据(demo)训练得出，fitness 与真实行情不同口径，"
                              "不可比较——继续用它锁死迭代")
            return info
        info["comparable"] = True
        return info

    def reset_anchor(self, name: str, *, reason: str = "") -> dict:
        """清除锚点（归档式，不删权重）：让下一条合格模型重新建立基线。

        只动 meta.json 的 best_version/best_fitness。权重文件（best.json.gz 与
        实盘读的 flat 文件）保持原样——删了会让部署端在下一轮接受前无模型可用。
        重置前的 meta.json 与 best.json.gz 复制到 anchor_reset/<ts>_* 留档可回滚。
        """
        meta = self._load_meta(name)
        old = self.best_info(name)
        cleared = {"fitness": old.get("fitness"), "version": old.get("version"),
                   "data_source": old.get("data_source") or ""}
        stamp = time.strftime("%Y%m%d_%H%M%S")
        arch_dir = self._model_dir(name) / "anchor_reset"
        arch_dir.mkdir(exist_ok=True)
        for src in (self._meta_path(name), self._best_path(name)):
            try:
                if src.exists():
                    shutil.copy2(src, arch_dir / f"{stamp}_{src.name}")
            except Exception as e:  # noqa: BLE001
                log.warning("[model_zoo] 锚点留档失败 %s: %s", src.name, e)
        # P4-E1：anchor_reset/ 只保留最近 _MAX_ARCHIVES 份留档（文件名前缀=时间序，
        # 防止每次重置都把 MB 级 best.json.gz 无限累积）。
        # P2：按"全局保留最近 N 份"裁剪——此前按文件名前缀（日期）分组、每天各
        # 留 30 份，每天重置一次的常规用法下裁剪永不触发，跨天无界增长。
        try:
            existing = sorted(arch_dir.glob("*"))
            for stale in existing[:-_MAX_ARCHIVES]:
                stale.unlink(missing_ok=True)
        except Exception as e:  # noqa: BLE001
            log.warning("[model_zoo] 锚点留档裁剪失败（不阻断重置）: %s", e)
        meta["best_version"] = 0
        meta["best_fitness"] = None
        meta["anchor_reset"] = {"timestamp": time.time(), "stamp": stamp,
                                "reason": reason, **cleared}
        self._write_meta(name, meta)
        log.warning("[model_zoo] 模型 %s 锚点已重置（原 fitness=%s 来源=%s）：%s",
                    name, cleared["fitness"], cleared["data_source"], reason or "未填写原因")
        return {"cleared": cleared, "stamp": stamp, "archive_dir": str(arch_dir)}

    def rollback(self, name: str, version: int) -> bool:
        """回退到指定版本，将其设为最佳。"""
        try:
            self.restore_deployment(name, version, outcome="rollback")
            return True
        except FileNotFoundError:
            return False
        except Exception as e:  # noqa: BLE001
            log.warning("[model_zoo] 回退模型 %s 到 v%d 失败: %s", name, version, e)
            return False

    def has_model(self, name: str) -> bool:
        """检查是否存在指定模型的最佳版本。"""
        return self._best_path(name).exists() or self._flat_path(name).exists()

    # ---- 内部工具 ----

    def _best_meta(self, name: str) -> dict:
        """当前最佳文件自带的 _meta（无则空 dict）——锚点的口径凭证。"""
        return self._best_head(name)[0]

    def _best_head(self, name: str) -> tuple[dict, Optional[int]]:
        """最佳文件的 (_meta, _version)：锚点凭证 + 它是哪个版本的存档。

        版本号必须问 best 文件自己：老代码的回退写盘会在 _meta 里丢掉 data_source，
        而 reset_anchor 之后 meta.json 的 best_version 已归零——此时在线权重还是
        那份模型，它的口径只能靠 _version 反查版本存档。

        best.json.gz 有 1MB+，而 status() 每次刷新都要读三条管线的锚点：
        在主事件循环里同步解压会卡顿界面，故按文件身份（mtime+size）缓存，
        文件被重写后键自然变化，不需要手动失效。
        """
        path = self._best_path(name)
        if not path.exists():
            return {}, None
        try:
            st = path.stat()
            key = (st.st_mtime_ns, st.st_size)
        except OSError:
            key = None
        if key is not None:
            hit = self._best_meta_cache.get(name)
            if hit and hit[0] == key:
                return hit[1]
        try:
            data = self._read_json_gz(path)
        except Exception as e:  # noqa: BLE001
            log.warning("[model_zoo] 读取 %s 锚点元数据失败: %s", name, e)
            return {}, None
        head = (dict(data.get("_meta") or {}), data.get("_version"))
        if key is not None:
            self._best_meta_cache[name] = (key, head)
        return head

    def _load_meta(self, name: str) -> dict:
        """加载模型元数据文件。

        P4-E1：损坏/截断的 meta.json 不再静默吞掉后返回 next_version=1——
        那会让下一次 save_agent 以 version=1 覆盖已有 v1.json.gz，其余版本变孤儿。
        现在：1) 报错并 rename 损坏文件留证；2) 从文件系统重建 next_version
        （扫 vN.json.gz 取 max+1），保证不覆盖已有版本；3) best_version/best_fitness
        置 0/None（损坏 meta 无法恢复锚点画像，宁可让回退门重新建立）。

        P2：按文件身份 (mtime_ns, size) 缓存——status() 每次轮询对三条管线
        各读一次 meta.json，此前同步 read_text 每轮 3 次系统调用打在事件循环上。
        写入方（_write_meta）按名失效；调用方只在随后落盘时才修改返回的 dict
        （置入缓存前会先做磁盘读取校验），不会污染缓存。
        """
        path = self._meta_path(name)
        try:
            st = path.stat() if path.exists() else None
        except OSError:
            st = None
        if st is not None:
            key = (st.st_mtime_ns, st.st_size)
            hit = self._meta_cache.get(name)
            if hit and hit[0] == key:
                return hit[1]
        if st is None:
            return {"next_version": 1, "versions": [], "best_version": 0, "best_fitness": None}
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
            self._meta_cache[name] = (key, meta)
            return meta
        except Exception as e:  # noqa: BLE001
            import time as _t
            stamp = _t.strftime("%Y%m%d_%H%M%S")
            corrupt = path.with_name(f"meta.corrupt-{stamp}.json")
            try:
                path.replace(corrupt)
            except Exception:  # noqa: BLE001
                pass
            # 从磁盘上的版本文件重建 next_version，避免覆盖已有 vN
            max_v = 0
            try:
                for p in self._model_dir(name).glob("v*.json.gz"):
                    try:
                        max_v = max(max_v, int(p.stem[1:]))
                    except ValueError:
                        continue
            except Exception:  # noqa: BLE001
                pass
            log.error("[model_zoo] %s 的 %s 损坏/无法解析（已留证 %s）："
                      "返回重建元数据, next_version=%s（不覆盖已有版本）",
                      name, path.name, corrupt.name, max_v + 1)
            return {"next_version": max_v + 1, "versions": [],
                    "best_version": 0, "best_fitness": None}

    def _write_meta(self, name: str, meta: dict) -> None:
        """写入模型元数据文件（失败抛错，供部署层回滚判断）。"""
        path = self._meta_path(name)
        old = path.read_bytes() if path.exists() else None
        try:
            self._atomic_write_bytes(path, self._json_bytes(meta))
        except Exception as e:
            try:
                self._restore_file(path, old)
            except Exception as restore_error:  # noqa: BLE001
                log.error("[model_zoo] 元数据回滚失败(%s): %s", name, restore_error)
            raise ModelZooError(f"写入元数据失败: {e}") from e
        finally:
            # 锚点任何变更都会写 meta.json：这里是缓存失效的唯一收口
            self._meta_cache.pop(name, None)
            self._best_meta_cache.pop(name, None)

    @staticmethod
    def _write_json_gz(path: Path, data: dict) -> None:
        """写入压缩 JSON 文件（失败抛错）。"""
        tmp = path.with_suffix(".tmp.gz")
        try:
            # P2：compresslevel=9 → 6（best.json.gz 可达 1MB+，6 档与 9 档压缩比
            # 几乎无差但写盘快数倍；保存路径已在 evolve_engine 侧移出事件循环）
            with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as f:
                json.dump(data, f, ensure_ascii=False, default=str)
            os.replace(tmp, path)
        except Exception as e:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise ModelZooError(f"写入 {path.name} 失败: {e}") from e

    @staticmethod
    def _read_json_gz(path: Path) -> dict:
        """读取压缩 JSON 文件。"""
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _write_flat_json(path: Path, data: dict) -> None:
        """写入兼容格式 JSON（与 DRL API 的 ACAgent.save() 同格式，失败抛错）。"""
        try:
            ModelZoo._atomic_write_bytes(path, ModelZoo._json_bytes(data))
        except Exception as e:
            raise ModelZooError(f"写入兼容格式 {path.name} 失败: {e}") from e
