"""内置代理客户端（2026-09-25）：让软件直接支持 hysteria2 等需客户端的节点。

原理：检测本地 hysteria 客户端二进制 → 用节点参数生成 hysteria v2 配置 →
启动 `hysteria client` 进程监听 127.0.0.1 本地端口（SOCKS5）→ 软件出站请求
指向该本地端口（仅本软件生效，不影响系统）。
"""
import json
import logging
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path


log = logging.getLogger("proxy_client")

# 本地客户端二进制目录（用户可手动放入，或程序引导下载）
BIN_DIR = Path(__file__).resolve().parent.parent / "bin"
_HY_BIN_CANDIDATES = ("hysteria", "hysteria-windows-amd64.exe")

_lock = threading.Lock()
_procs: dict[str, subprocess.Popen] = {}
_used_ports: set[int] = set()

_LOCAL_SOCKS_HOST = "127.0.0.1"

# 起进程后等端口就绪的秒数：调用方（保存节点 / 启动预热）要立刻探测端口，
# 不等会读到"还没 bind"的空窗。测试置 0 跳过。
_READY_TIMEOUT = 2.0


def _wait_listening(port: int, timeout: float | None = None) -> bool:
    # timeout 运行时取 _READY_TIMEOUT（不能写成默认参数值：那会在函数定义时绑死，
    # 测试与运行期修改都会失效）
    if timeout is None:
        timeout = _READY_TIMEOUT
    if timeout <= 0:
        return True
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((_LOCAL_SOCKS_HOST, int(port)), timeout=0.2):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def _launch(bin_path: str, cfg_file: Path, scheme: str, port: int) -> tuple[bool, str, str]:
    """拉起 hysteria 客户端进程并登记（start_local_proxy / rearm 共用）。

    端口未就绪 = 客户端启动即退（配置不可达/证书失败/端口被占）→ **必须如实报失败**：
    曾只打 warning 仍返回 ok=True，程序重启时日志说"已恢复本地加速客户端"、实际
    hysteria 进程已死/端口无人监听，加速静默失效（2026-09-28 用户实测）。"""
    try:
        proc = subprocess.Popen([bin_path, "client", "-c", str(cfg_file)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError as e:
        return False, "", f"启动 hysteria 客户端失败: {e}"
    _procs[scheme] = proc
    _used_ports.add(int(port))
    # socks5h（远端 DNS）：本机 DNS 被污染时（实测 www.okx.com -> 169.254.0.2），
    # socks5:// 会在本地解析域名再送进隧道，交易所请求必然超时。
    url = f"socks5h://{_LOCAL_SOCKS_HOST}:{int(port)}"
    if not _wait_listening(int(port)):
        # 客户端未就绪：终止进程防僵尸占端口，从登记表摘除，如实返回失败。
        try:
            proc.terminate()
        except Exception:  # noqa: BLE001
            pass
        _procs.pop(scheme, None)
        _used_ports.discard(int(port))
        log.warning("[proxy-client] 客户端 %s 启动即退/端口 %d 未就绪（配置不可达？"
                    "证书失败？端口被占？），查看 %s 对应节点是否可用",
                    scheme, port, cfg_file.name)
        return False, url, (f"客户端未就绪：端口 {port} 在 {_READY_TIMEOUT}s 内无人监听"
                            f"（节点不可达/证书失败/端口被占），查看 {cfg_file.name}")
    log.info("[proxy-client] %s 客户端已启动: %s", scheme, url)
    return True, url, ""


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
    pin = str(params.get("pinSHA256") or "")
    if pin:
        tls["pinSHA256"] = pin
        # 机场节点多为自签证书：hysteria **先做证书链校验、再校验 pin**，只给 pin
        # 会先死在链校验（x509: certificate signed by unknown authority）→ 客户端
        # 秒退、本地 socks5 没人监听、加速静默失效（2026-09-27 实测）。
        # 设 insecure 跳过链校验，把信任判断交给 pin —— 实测错 pin 仍会被拒
        # （no certificate matches the pinned hash），并非关闭验证。
        tls["insecure"] = True
    if str(params.get("insecure", "")).lower() == "true":
        tls["insecure"] = True
    if tls:
        cfg["tls"] = tls
    return cfg


def _pick_port() -> int:
    """选一个空闲端口：跳过本进程已用 + 已被实际监听的端口。

    曾只查 _used_ports（模块级内存集合）：重启后集合是空的，会选中上个会话
    遗留孤儿 hysteria 仍占着的端口 → 新进程 bind 失败即退，但 _wait_listening
    探测到孤儿还在监听 → 返回 ok=True，界面显示已切换、实际流量走的是旧节点的
    孤儿进程（2026-09-28 用户实测"自动开启没做好"的成因之一）。"""
    port = 7891
    while port < 7990:
        if port in _used_ports:
            port += 1
            continue
        # 真实探测：端口已被（孤儿）进程监听则跳过
        try:
            with socket.create_connection((_LOCAL_SOCKS_HOST, int(port)), timeout=0.15):
                port += 1  # 已被占用
                continue
        except OSError:
            return port
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
        return _launch(bin_path, cfg_file, scheme, port)


def rearm_local_proxy() -> tuple[bool, str, str]:
    """启动时自愈：持久化的代理若指向本机 socks5(h)://127.0.0.1:PORT，
    且 bin/ 里留有该端口的 hysteria 配置，就把客户端重新拉起来。

    否则重启后 data/proxy_config.json 仍指着一个没人监听的端口：端口探测落空、
    resolved 为空、流量回退直连，而界面却显示"已配置"——加速静默失效
    （2026-09-27 用户实测：OKX 测试连接失败）。返回 (ok, url, error)。"""
    from urllib.parse import urlsplit

    from config.settings import settings  # 局部导入，避免与 config 顶层循环依赖

    raw = (settings.proxy_url or "").strip()
    if not raw:
        return False, "", ""
    try:
        parts = urlsplit(raw)
        host = parts.hostname or ""
        port = parts.port or 0
    except ValueError:
        return False, "", ""
    if host not in ("127.0.0.1", "localhost") or (parts.scheme or "").lower() not in ("socks5", "socks5h"):
        return False, "", ""  # 远端代理，不该由本地客户端接管
    if not port:
        return False, "", ""
    cfg_file = BIN_DIR / f"hy2_client_{port}.json"
    if not cfg_file.exists():
        return False, "", ""  # 没有配置可复用，交由界面提示重选节点
    bin_path = find_hysteria_bin()
    if not bin_path:
        return False, "", (f"已保存本地加速配置，但未找到 hysteria 二进制，请到「网络加速」"
                           f"重新下载（放入 {BIN_DIR}）")
    with _lock:
        # 端口复用：目标 socks 端口已有人监听（上个会话遗留的孤儿客户端仍在/配置
        # 对应客户端还活着）→ 直接复用，不重复拉起。重复拉起会 bind 失败即退
        # （端口被占），反而造成"新进程死了、日志却说已恢复"（2026-09-28 实测）。
        try:
            with socket.create_connection((_LOCAL_SOCKS_HOST, int(port)), timeout=0.3):
                log.info("[proxy-client] 端口 %s 已有监听，复用现成客户端（无需重启）",
                         _LOCAL_SOCKS_HOST + ":" + str(port))
                return True, f"socks5h://{_LOCAL_SOCKS_HOST}:{int(port)}", ""
        except OSError:
            pass  # 端口无人监听 → 正常拉起
        _stop_locked()
        return _launch(bin_path, cfg_file, "hysteria2", port)


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


# --- CA 信任包（2026-09-27：github.com 下载 SSL 校验失败的根因修复）---------
# 本机加速器/杀软（实测为 Steam++ · Watt Toolkit · BeyondDimension 装的
# 「SteamTools Certificate」根证书）会拦截 github.com 的 TLS 并用自建根证书签名。
# 该根证书在 Windows 根证书库（浏览器/系统默认信任），但 requests 只认 certifi，
# 于是报 `unable to get local issuer certificate`。桌面应用标准做法是与系统
# 信任集对齐：合并 certifi ∪ Windows 根证书库，验证保持开启（绝不 verify=False）。
_CA_BUNDLE_CACHE = Path(tempfile.gettempdir()) / "cat_ca_trust_bundle.pem"
_CA_TTL_SECONDS = 24 * 3600
_ca_lock = threading.Lock()


def _export_windows_roots(dest: Path) -> bool:
    """导出 Windows 根证书库（LocalMachine\\Root + CurrentUser\\Root）为 PEM。"""
    ps = ("Get-ChildItem Cert:\\LocalMachine\\Root, Cert:\\CurrentUser\\Root "
          "-ErrorAction SilentlyContinue | ForEach-Object { "
          "$b=[Convert]::ToBase64String($_.RawData); "
          "'-----BEGIN CERTIFICATE-----'; $b; '-----END CERTIFICATE-----' } | "
          f"Set-Content -Path '{dest}' -Encoding Ascii")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                           capture_output=True, timeout=60)
    except Exception as e:  # noqa: BLE001
        log.warning("[proxy-client] 导出 Windows 根证书库失败: %s", e)
        return False
    return r.returncode == 0 and dest.exists() and dest.stat().st_size > 1000


def ca_bundle_path(refresh: bool = False) -> str | bool:
    """certifi ∪ Windows 根证书库的合并 CA 包（缓存 24h），给 requests 的 verify= 用。

    失败一律回退 certifi；**绝不返回空值**——requests 收到 falsy 的 verify
    会直接关闭证书验证（CERT_NONE），那是把下载暴露给任意中间人。
    """
    import certifi
    try:
        certifi_path = str(certifi.where())
    except Exception:  # noqa: BLE001  # pragma: no cover
        return True  # certifi 是 requests 的依赖，取不到才退化为库默认（仍开启验证）
    if os.name != "nt":
        return certifi_path
    with _ca_lock:
        try:
            stale = (not _CA_BUNDLE_CACHE.exists()
                     or time.time() - _CA_BUNDLE_CACHE.stat().st_mtime >= _CA_TTL_SECONDS)
        except OSError:
            stale = True
        if refresh:
            stale = True
        if not stale:
            return str(_CA_BUNDLE_CACHE)
        try:
            roots = _CA_BUNDLE_CACHE.with_suffix(".roots.pem")
            if _export_windows_roots(roots):
                certifi_text = Path(certifi_path).read_text(errors="ignore")
                merged = certifi_text + "\n" + roots.read_text(errors="ignore")
                if merged.count("BEGIN CERTIFICATE") > 0:
                    _CA_BUNDLE_CACHE.write_text(merged, encoding="utf-8")
                    roots.unlink(missing_ok=True)
                    return str(_CA_BUNDLE_CACHE)
        except Exception as e:  # noqa: BLE001
            log.warning("[proxy-client] 构建合并 CA 包失败，回退 certifi: %s", e)
        return certifi_path


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
        resp = requests.get(HYSTERIA_DOWNLOAD_URL, stream=True, timeout=timeout,
                            verify=ca_bundle_path())
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
