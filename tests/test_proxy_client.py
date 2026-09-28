"""内置代理客户端（hysteria2）测试（2026-09-25）。

程序检测 hysteria 客户端二进制 → 生成配置 → 本地启动 SOCKS5 → 软件走它。
"""

from net import proxy_client


def _settings():
    from config.settings import settings
    return settings


def _hy_node(**over):
    node = {
        "name": "JP-01", "type": "hysteria2", "server": "203.10.98.189", "port": 443,
        "url": "",
        "params": {"auth": "c0f9ee58-ce11-4e63-9115-d3d0887a3626",
                   "sni": "www.bing.com", "insecure": "false",
                   "pinSHA256": "1d7995901a93bada6d17f10289441793e8ec54ee314d9f04f3a9d05daa622331"},
    }
    node.update(over)
    return node


def test_hy2_client_config_generation():
    cfg = proxy_client.hy2_client_config(_hy_node(), listen_port=7891)
    assert cfg["server"] == "203.10.98.189:443"
    assert cfg["auth"] == "c0f9ee58-ce11-4e63-9115-d3d0887a3626"
    assert cfg["socks5"] == {"listen": "127.0.0.1:7891"}
    assert cfg["tls"]["sni"] == "www.bing.com"
    # 2026-09-27：机场节点普遍自签证书，hysteria **先做链校验再校验 pin**，
    # 只给 pin 会让链校验先失败（x509: certificate signed by unknown authority）
    # 导致客户端秒退、加速静默失效。因此带 pin ⇒ 必须同时 insecure，
    # 把信任判断交给 pin（实测：错 pin 仍会被拒 "no certificate matches the pinned hash"）。
    assert cfg["tls"]["insecure"] is True
    assert cfg["tls"]["pinSHA256"] == ("1d7995901a93bada6d17f10289441793e8ec54ee"
                                       "314d9f04f3a9d05daa622331")


def test_hy2_config_insecure_true():
    node = _hy_node()
    node["params"]["insecure"] = "true"
    cfg = proxy_client.hy2_client_config(node, 7891)
    assert cfg["tls"]["insecure"] is True


def test_hy2_config_no_pin_keeps_chain_verification():
    """无 pin 且未显式 insecure → 不注入 insecure，保留正常链校验。"""
    node = _hy_node()
    node["params"].pop("pinSHA256")
    node["params"]["insecure"] = "false"
    cfg = proxy_client.hy2_client_config(node, 7891)
    assert "insecure" not in cfg["tls"]
    assert cfg["tls"]["sni"] == "www.bing.com"


class _FakeProc:
    def terminate(self):
        pass


def test_find_hysteria_bin(monkeypatch, tmp_path):
    monkeypatch.setattr(proxy_client, "BIN_DIR", tmp_path)
    assert proxy_client.find_hysteria_bin() == ""
    fake = tmp_path / "hysteria-windows-amd64.exe"
    fake.write_bytes(b"x")
    assert proxy_client.find_hysteria_bin() == str(fake)


# 2026-09-27：重启自愈——proxy_config.json 指向本机 socks5，但 hysteria 进程不会
# 自己活过来 ⇒ 端口探测落空 ⇒ resolved 为空 ⇒ 加速静默失效（用户实测 OKX 连不上）。
def test_rearm_local_proxy_restores_from_saved_config(monkeypatch, tmp_path):
    monkeypatch.setattr(proxy_client, "BIN_DIR", tmp_path)
    monkeypatch.setattr(proxy_client, "_READY_TIMEOUT", 0)
    monkeypatch.setattr(proxy_client, "_used_ports", set())
    monkeypatch.setattr(proxy_client, "_procs", {})
    fake = tmp_path / "hysteria-windows-amd64.exe"
    fake.write_bytes(b"x")
    monkeypatch.setattr(proxy_client, "find_hysteria_bin", lambda: str(fake))
    (tmp_path / "hy2_client_7891.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(proxy_client.subprocess, "Popen", lambda *a, **k: _FakeProc())
    monkeypatch.setattr(_settings(), "proxy_url", "socks5h://127.0.0.1:7891")
    ok, url, err = proxy_client.rearm_local_proxy()
    assert ok is True, err
    assert url == "socks5h://127.0.0.1:7891"


def test_rearm_skips_when_proxy_is_remote(monkeypatch, tmp_path):
    """远端代理（非本机 socks5）不该被本地客户端接管。"""
    spawned = {"n": 0}
    monkeypatch.setattr(proxy_client, "BIN_DIR", tmp_path)
    monkeypatch.setattr(proxy_client.subprocess, "Popen",
                        lambda *a, **k: spawned.__setitem__("n", spawned["n"] + 1) or _FakeProc())
    monkeypatch.setattr(_settings(), "proxy_url", "socks5://10.0.0.5:7890")
    ok, url, err = proxy_client.rearm_local_proxy()
    assert ok is False and spawned["n"] == 0


def test_rearm_skips_when_no_saved_config(monkeypatch, tmp_path):
    """没有对应端口的配置文件 → 不乱起进程，交由界面提示用户重选节点。"""
    spawned = {"n": 0}
    monkeypatch.setattr(proxy_client, "BIN_DIR", tmp_path)
    monkeypatch.setattr(proxy_client.subprocess, "Popen",
                        lambda *a, **k: spawned.__setitem__("n", spawned["n"] + 1) or _FakeProc())
    monkeypatch.setattr(_settings(), "proxy_url", "socks5h://127.0.0.1:7891")
    ok, url, err = proxy_client.rearm_local_proxy()
    assert ok is False and spawned["n"] == 0


def test_start_local_proxy_success(monkeypatch, tmp_path):
    monkeypatch.setattr(proxy_client, "BIN_DIR", tmp_path)
    monkeypatch.setattr(proxy_client, "_READY_TIMEOUT", 0)
    monkeypatch.setattr(proxy_client, "_used_ports", set())  # 模块级状态，跨测试会串
    monkeypatch.setattr(proxy_client, "_procs", {})
    fake = tmp_path / "hysteria-windows-amd64.exe"
    fake.write_bytes(b"x")
    monkeypatch.setattr(proxy_client, "find_hysteria_bin", lambda: str(fake))

    class _Proc:
        def terminate(self):
            pass

    monkeypatch.setattr(proxy_client.subprocess, "Popen", lambda *a, **k: _Proc())
    # 端口探测 mock 为空闲：_pick_port 会真实探测孤儿进程占位，真实环境 7891 可能
    # 已被占（测试断言写死 7891），必须 mock 空闲才能稳定落在 7891。
    def _free_port(*a, **k):
        raise OSError("free")
    monkeypatch.setattr(proxy_client.socket, "create_connection", _free_port)
    ok, url, err = proxy_client.start_local_proxy(_hy_node())
    assert ok is True
    # 2026-09-27：必须是 socks5h（远端 DNS）。本机 DNS 已被污染
    # （实测 www.okx.com -> 169.254.0.2），socks5:// 会在本地解析成废 IP，
    # 经隧道连过去必然超时；socks5h:// 让节点侧解析，OKX 实测 1.5s 通。
    assert url == "socks5h://127.0.0.1:7891"
    assert err == ""


def test_start_local_proxy_missing_bin(monkeypatch, tmp_path):
    monkeypatch.setattr(proxy_client, "BIN_DIR", tmp_path)
    monkeypatch.setattr(proxy_client, "find_hysteria_bin", lambda: "")
    ok, url, err = proxy_client.start_local_proxy(_hy_node())
    assert ok is False
    assert "hysteria" in err and "github.com/apernet/hysteria/releases" in err


def test_pick_port_skips_occupied_ports(monkeypatch):
    """端口选择必须跳过已被监听的端口（含上个会话遗留的孤儿进程占位）。

    曾只查进程内 _used_ports：重启后集合为空 → 选中孤儿占着的端口 → 新客户端
    bind 失败即退，界面却显示已切换（流量实走旧节点孤儿进程）。"""
    occupied = {7891: True, 7892: False, 7893: False}

    def _fake_conn(*a, **k):
        class _Sock:
            def __enter__(self):
                return self
            def __exit__(self, *exc):
                return False
            def close(self):
                pass
        if occupied.get(a[0][1]):
            return _Sock()
        raise OSError("free")

    monkeypatch.setattr(proxy_client, "_used_ports", set())
    monkeypatch.setattr(proxy_client.socket, "create_connection", _fake_conn)
    assert proxy_client._pick_port() == 7892, "7891 被孤儿占用时必须跳过"


def test_select_node_direct_url():
    ok, url, err = proxy_client.select_node({"type": "socks5", "url": "socks5://1.2.3.4:7890"})
    assert ok is True and url == "socks5://1.2.3.4:7890"


def test_select_node_unsupported():
    ok, url, err = proxy_client.select_node({"type": "vmess", "url": ""})
    assert ok is False and "暂不支持" in err or "需本地客户端" in err or err != ""


# 2026-09-26：自动下载 hysteria 客户端二进制
def test_download_hysteria_client(monkeypatch, tmp_path):
    class _Resp:
        headers = {"content-length": "5000000"}
        def raise_for_status(self):
            pass
        def iter_content(self, chunk_size=1 << 16):
            yield b"x" * (1 << 20)  # 1MB（满足大小校验）

    def _fake_get(url, stream=True, timeout=0, verify=True):
        assert verify is not False, "证书验证不允许被关闭"
        return _Resp()
    monkeypatch.setattr("requests.get", _fake_get)
    ok, dest, err = proxy_client.download_hysteria_client(dest_dir=tmp_path)
    assert ok is True and err == ""
    assert (tmp_path / "hysteria-windows-amd64.exe").exists()
    assert (tmp_path / "hysteria-windows-amd64.exe").stat().st_size == (1 << 20)


def test_download_hysteria_client_http_error(monkeypatch, tmp_path):
    class _Resp:
        def raise_for_status(self):
            raise RuntimeError("404 Not Found")
    monkeypatch.setattr("requests.get", lambda *a, **k: _Resp())
    ok, dest, err = proxy_client.download_hysteria_client(dest_dir=tmp_path)
    assert ok is False
    assert "404" in err or "下载失败" in err


# ---------------------------------------------------------------------------
# 2026-09-27：github.com 下载报 `unable to get local issuer certificate`
# 根因：本机 Steam++（Watt Toolkit/BeyondDimension）等加速器把自建根证书装进
# Windows 根证书库并拦截 github.com 的 TLS，而 requests 只认 certifi（118 张），
# 不含该根证书 → 验证失败。系统信任库（172 张）能验证。
# 修法：下载改用「certifi ∪ Windows 根证书库」合并包，验证**保持开启**。
# ---------------------------------------------------------------------------
def test_download_verifies_with_trust_bundle(monkeypatch, tmp_path):
    """下载必须带 verify=CA 包路径 —— 既不能用默认 certifi，也绝不允许 verify=False。"""
    import certifi
    from pathlib import Path as _Path

    seen = {}

    class _Resp:
        headers = {"content-length": "5000000"}

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size=1 << 16):
            yield b"x" * (1 << 20)

    def _fake_get(url, **kw):
        seen.update(kw)
        return _Resp()

    monkeypatch.setattr("requests.get", _fake_get)
    ok, dest, err = proxy_client.download_hysteria_client(dest_dir=tmp_path)
    assert ok is True, err
    bundle = seen.get("verify")
    assert isinstance(bundle, str) and bundle, f"verify 必须是 CA 包路径，实际 {bundle!r}"
    assert _Path(bundle).exists(), f"CA 包不存在: {bundle}"
    n_bundle = _Path(bundle).read_text(errors="ignore").count("BEGIN CERTIFICATE")
    n_certifi = _Path(certifi.where()).read_text(errors="ignore").count("BEGIN CERTIFICATE")
    assert n_bundle >= n_certifi > 0


def test_ca_bundle_falls_back_to_certifi_when_export_fails(monkeypatch):
    """Windows 根库导出失败（无 PowerShell/权限）→ 回退 certifi，验证不能关。"""
    import certifi
    from pathlib import Path as _Path

    monkeypatch.setattr(proxy_client, "_export_windows_roots", lambda dest: False)
    path = proxy_client.ca_bundle_path(refresh=True)
    assert isinstance(path, str) and path
    assert _Path(path).exists()
    assert path == certifi.where() or "BEGIN CERTIFICATE" in _Path(path).read_text(errors="ignore")


def test_ca_bundle_falls_back_when_powershell_raises(monkeypatch):
    """导出过程抛异常也不允许返回空串/False —— 空串会让 requests 关掉验证。"""
    def _boom(dest):
        raise RuntimeError("powershell exploded")

    monkeypatch.setattr(proxy_client, "_export_windows_roots", _boom)
    path = proxy_client.ca_bundle_path(refresh=True)
    assert path, "绝不能返回空值（requests 会因此跳过证书验证）"


def test_ca_bundle_refreshes_only_when_stale(monkeypatch, tmp_path):
    """合并包缓存 24h：新鲜缓存直接复用，不再每次下载都跑 PowerShell。"""
    calls = {"n": 0}
    cache = tmp_path / "merged.pem"

    def _fake_export(dest):
        calls["n"] += 1
        dest.write_text("-----BEGIN CERTIFICATE-----\nAAA\n-----END CERTIFICATE-----\n",
                        encoding="utf-8")
        return True

    monkeypatch.setattr(proxy_client, "_export_windows_roots", _fake_export)
    monkeypatch.setattr(proxy_client, "_CA_BUNDLE_CACHE", cache)
    monkeypatch.setattr(proxy_client, "_CA_TTL_SECONDS", 86400)
    p1 = proxy_client.ca_bundle_path(refresh=True)
    p2 = proxy_client.ca_bundle_path()
    assert p1 == p2 == str(cache)
    assert calls["n"] == 1  # 第二次命中缓存，未再导出


# ---- 2026-09-28：rearm 假成功根治（用户实测"启动后自动开启网络加速没做好"）----
# 背景：_launch 曾对"客户端进程拉起但端口未就绪/启动即退"只打 warning 仍返回
# ok=True。于是程序重启时 rearm 报"已恢复本地加速客户端"，但 hysteria 进程
# 实际已死/端口无人监听 → 界面显示已配置、实际流量在直连，加速静默失效。
# 这些测试锁死：客户端未真正就绪必须如实返回 ok=False。


def test_launch_reports_failure_when_port_never_listens(monkeypatch, tmp_path):
    """端口在超时内未就绪（客户端启动即退）→ _launch 必须 ok=False，不得假成功。"""
    spawned = []
    class _FakeProc:
        def __init__(self):
            spawned.append(self)
            self._terminated = False
        def terminate(self):
            self._terminated = True
        def poll(self):
            return 0 if self._terminated else None

    fake = tmp_path / "hysteria-windows-amd64.exe"
    fake.write_bytes(b"x")
    cfg = tmp_path / "hy2_client_9999.json"
    cfg.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(proxy_client, "find_hysteria_bin", lambda: str(fake))
    monkeypatch.setattr(proxy_client.subprocess, "Popen",
                        lambda *a, **k: _FakeProc())
    monkeypatch.setattr(proxy_client, "_wait_listening", lambda port, timeout=None: False)

    ok, url, err = proxy_client._launch(str(fake), cfg, "hysteria2", 9999)
    assert ok is False, f"端口未就绪必须报失败，实际 ok={ok} url={url}"
    assert err, "必须带具体错误信息（此前静默假成功：日志说已恢复、实际无加速）"
    assert "9999" in err or "监听" in err or "就绪" in err or "启动" in err
    assert spawned and spawned[0]._terminated, "客户端未就绪时应终止已起的进程，防僵尸占用端口"


def test_rearm_returns_false_when_client_dies_after_spawn(monkeypatch, tmp_path):
    """rearm 链路：客户端拉起后端口无监听 → 如实返回 (False, url, err)，
    供 web/main.py 报"恢复失败"，前端才能显示未生效（不得静默假成功）。"""
    fake = tmp_path / "hysteria-windows-amd64.exe"
    fake.write_bytes(b"x")
    (tmp_path / "hy2_client_7891.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(proxy_client, "BIN_DIR", tmp_path)
    monkeypatch.setattr(proxy_client, "find_hysteria_bin", lambda: str(fake))
    monkeypatch.setattr(proxy_client.subprocess, "Popen",
                        lambda *a, **k: _FakeProc())
    monkeypatch.setattr(proxy_client, "_wait_listening", lambda port, timeout=None: False)
    monkeypatch.setattr(_settings(), "proxy_url", "socks5h://127.0.0.1:7891")
    # 模拟端口空闲（无孤儿在监听）→ 才会真正走 _launch 拉起路径，测到"拉起后死"
    def _free_port(*a, **k):
        raise OSError("port free")
    monkeypatch.setattr(proxy_client.socket, "create_connection", _free_port)

    ok, url, err = proxy_client.rearm_local_proxy()
    assert ok is False, "客户端未就绪 rearm 必须报失败（此前假成功导致加速静默失效）"
    assert err, "必须带具体错误"


def test_rearm_reuses_alive_port_without_respawning(monkeypatch, tmp_path):
    """端口已有人监听（上个会话的孤儿客户端还在）→ rearm 复用，不重复拉起新进程。

    重复拉起会因端口被占而 bind 失败即退，反而制造"新进程死了、日志却说已恢复"
    的假象（2026-09-28 实测：两个孤儿 hysteria 进程同时存活、旧配置顶掉新配置）。"""
    spawned = {"n": 0}
    class _FakeProc:
        def terminate(self):
            pass
        def poll(self):
            return None

    fake = tmp_path / "hysteria-windows-amd64.exe"
    fake.write_bytes(b"x")
    (tmp_path / "hy2_client_7891.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(proxy_client, "BIN_DIR", tmp_path)
    monkeypatch.setattr(proxy_client, "find_hysteria_bin", lambda: str(fake))
    monkeypatch.setattr(proxy_client.subprocess, "Popen",
                        lambda *a, **k: spawned.__setitem__("n", spawned["n"] + 1) or _FakeProc())
    monkeypatch.setattr(proxy_client, "_wait_listening", lambda port, timeout=None: True)
    monkeypatch.setattr(_settings(), "proxy_url", "socks5h://127.0.0.1:7891")
    # 模拟端口 7891 已有孤儿在监听
    def _occupied(*a, **k):
        class _Sock:
            def __enter__(self):
                return self
            def __exit__(self, *exc):
                return False
            def close(self):
                pass
        return _Sock()
    monkeypatch.setattr(proxy_client.socket, "create_connection", _occupied)

    ok, url, err = proxy_client.rearm_local_proxy()
    assert ok is True
    assert spawned["n"] == 0, "端口已监听时应复用，不得重复拉起新进程"
    assert url == "socks5h://127.0.0.1:7891"

