"""内置代理客户端（2026-09-25）：让软件直接支持 hysteria2 等需客户端的节点。

原理：检测本地 hysteria 客户端二进制 → 用节点参数生成 hysteria v2 配置 →
启动 `hysteria client` 进程监听 127.0.0.1 本地端口（SOCKS5）→ 软件出站请求
指向该本地端口（仅本软件生效，不影响系统）。
"""
import json
import logging
import shutil
import subprocess
import threading
from pathlib import Path


log = logging.getLogger("proxy_client")

# 本地客户端二进制目录（用户可手动放入，或程序引导下载）
BIN_DIR = Path(__file__).resolve().parent.parent / "bin"
_HY_BIN_CANDIDATES = ("hysteria", "hysteria-windows-amd64.exe")

_lock = threading.Lock()
_procs: dict[str, subprocess.Popen] = {}
_used_ports: set[int] = set()

_LOCAL_SOCKS_HOST = "127.0.0.1"


def find_hysteria_bin() -> str:
    """查找 hysteria 客户端二进制：PATH → bin/ 目录。"""
    for name in _HY_BIN_CANDIDATES:
        p = shutil.which(name)
        if p:
            return p
        cand = BIN_DIR / name
        if cand.exists():
            return str(cand)
    return ""


def hy2_client_config(node: dict, listen_port: int) -> dict:
    """由 hysteria2 节点生成 hysteria v2 客户端配置（JSON 兼容 dict）。"""
    params = node.get("params") or {}
    cfg: dict = {
        "server": f"{node.get('server')}:{node.get('port')}",
        "auth": params.get("auth") or "",
        "socks5": {"listen": f"{_LOCAL_SOCKS_HOST}:{listen_port}"},
    }
    tls: dict = {}
    if params.get("sni"):
        tls["sni"] = params["sni"]
    if str(params.get("insecure", "")).lower() == "true":
        tls["insecure"] = True
    if params.get("pinSHA256"):
        tls["pinSHA256"] = params["pinSHA256"]
    if tls:
        cfg["tls"] = tls
    return cfg


def _pick_port() -> int:
    port = 7891
    while port in _used_ports and port < 7990:
        port += 1
    return port


def start_local_proxy(node: dict) -> tuple[bool, str, str]:
    """启动本地代理客户端（hysteria2）：返回 (ok, socks_url, error)。"""
    scheme = (node.get("type") or "").lower()
    if scheme not in ("hysteria2", "hysteria"):
        return False, "", f"暂不支持内置客户端协议: {scheme}"
    bin_path = find_hysteria_bin()
    if not bin_path:
        return False, "", ("未检测到 hysteria 客户端。请下载 hysteria 二进制并放入 "
                           f"{BIN_DIR}（如 hysteria-windows-amd64.exe），参考 "
                           "https://github.com/apernet/hysteria/releases")
    with _lock:
        _stop_locked()
        port = _pick_port()
        cfg = hy2_client_config(node, port)
        cfg_file = BIN_DIR / f"hy2_client_{port}.json"
        BIN_DIR.mkdir(parents=True, exist_ok=True)
        cfg_file.write_text(json.dumps(cfg), encoding="utf-8")
        try:
            proc = subprocess.Popen(
                [bin_path, "client", "-c", str(cfg_file)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as e:
            return False, "", f"启动 hysteria 客户端失败: {e}"
        _procs[scheme] = proc
        _used_ports.add(port)
        url = f"socks5://{_LOCAL_SOCKS_HOST}:{port}"
        log.info("[proxy-client] %s 客户端已启动: %s", scheme, url)
        return True, url, ""


def _stop_locked() -> None:
    for p in _procs.values():
        try:
            p.terminate()
        except Exception:  # noqa: BLE001
            pass
    _procs.clear()


def stop_local_proxy() -> None:
    with _lock:
        _stop_locked()


def select_node(node: dict) -> tuple[bool, str, str]:
    """选择节点：可直接代理（socks5/http）→ 返回其 URL；需客户端（hysteria2 等）→ 本地启动。"""
    scheme = (node.get("type") or "").lower()
    url = (node.get("url") or "").strip()
    if url:
        return True, url, ""
    if scheme in ("hysteria2", "hysteria"):
        return start_local_proxy(node)
    return False, "", f"节点类型 {scheme} 暂不支持直接使用（需本地客户端转换）"


# 官方 GitHub Releases 最新版 Windows amd64（下载来源固定，防止任意 URL 注入）
HYSTERIA_DOWNLOAD_URL = ("https://github.com/apernet/hysteria/releases/"
                         "latest/download/hysteria-windows-amd64.exe")
_HY_BIN_NAME = "hysteria-windows-amd64.exe"


def download_hysteria_client(dest_dir: Path | None = None,
                             timeout: int = 180) -> tuple[bool, str, str]:
    """自动下载官方 hysteria 客户端二进制到 bin/（网页一键操作）。返回 (ok, dest, error)。"""
    import requests
    dest_dir = Path(dest_dir or BIN_DIR)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / _HY_BIN_NAME
    try:
        resp = requests.get(HYSTERIA_DOWNLOAD_URL, stream=True, timeout=timeout)
        resp.raise_for_status()
        size = int(resp.headers.get("content-length") or 0)
        if size and not (1_000_000 <= size <= 200_000_000):
            return False, "", f"下载文件大小异常（{size} 字节），已中止"
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)
    except Exception as e:  # noqa: BLE001
        return False, "", f"下载 hysteria 客户端失败: {e}"
    ok_size = dest.stat().st_size if dest.exists() else 0
    if ok_size < 1_000_000:
        return False, "", f"下载不完整（{ok_size} 字节），请重试或手动放入 bin/"
    log.info("[proxy-client] hysteria 客户端已下载: %s (%d 字节)", dest, ok_size)
    return True, str(dest), ""
