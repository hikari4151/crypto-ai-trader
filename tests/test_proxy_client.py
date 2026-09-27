"""内置代理客户端（hysteria2）测试（2026-09-25）。

程序检测 hysteria 客户端二进制 → 生成配置 → 本地启动 SOCKS5 → 软件走它。
"""

from net import proxy_client


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
    assert "insecure" not in cfg["tls"]  # insecure=false → 不写


def test_hy2_config_insecure_true():
    node = _hy_node()
    node["params"]["insecure"] = "true"
    cfg = proxy_client.hy2_client_config(node, 7891)
    assert cfg["tls"]["insecure"] is True


def test_find_hysteria_bin(monkeypatch, tmp_path):
    monkeypatch.setattr(proxy_client, "BIN_DIR", tmp_path)
    assert proxy_client.find_hysteria_bin() == ""
    fake = tmp_path / "hysteria-windows-amd64.exe"
    fake.write_bytes(b"x")
    assert proxy_client.find_hysteria_bin() == str(fake)


def test_start_local_proxy_success(monkeypatch, tmp_path):
    monkeypatch.setattr(proxy_client, "BIN_DIR", tmp_path)
    fake = tmp_path / "hysteria-windows-amd64.exe"
    fake.write_bytes(b"x")
    monkeypatch.setattr(proxy_client, "find_hysteria_bin", lambda: str(fake))

    class _Proc:
        def terminate(self):
            pass

    monkeypatch.setattr(proxy_client.subprocess, "Popen", lambda *a, **k: _Proc())
    ok, url, err = proxy_client.start_local_proxy(_hy_node())
    assert ok is True
    assert url == "socks5://127.0.0.1:7891"
    assert err == ""


def test_start_local_proxy_missing_bin(monkeypatch, tmp_path):
    monkeypatch.setattr(proxy_client, "BIN_DIR", tmp_path)
    monkeypatch.setattr(proxy_client, "find_hysteria_bin", lambda: "")
    ok, url, err = proxy_client.start_local_proxy(_hy_node())
    assert ok is False
    assert "hysteria" in err and "github.com/apernet/hysteria/releases" in err


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

