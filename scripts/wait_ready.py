"""start.bat 服务器就绪探测辅助：退出码 0=就绪(HTTP 200)，1=超时。

修复背景：start.bat 原用 curl 探测 http://localhost:PORT/api/health，
但服务器默认绑定 127.0.0.1（config/settings.py web_host），Windows 下
localhost 常解析为 IPv6 ::1，curl 连不上 → 探测永远失败 → 等待循环
空转到 30s 超时后仍"anyway"打开浏览器，用户先看到"网页无法连接"。

本脚本改用 127.0.0.1 直连 + urllib 轮询，与服务器绑定一致；且不依赖
外部 curl / timeout 命令（timeout /t 在非交互 bat 环境下会报错空转）。
"""
import socket
import sys
import time
import urllib.error
import urllib.request

HOST = "127.0.0.1"  # 与 config/settings.py web_host 默认一致
TIMEOUT_SEC = 60    # 就绪等待上限
INTERVAL_SEC = 0.5  # 轮询间隔
PROBE_TIMEOUT = 2   # 单次连接超时


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    timeout_sec = float(sys.argv[2]) if len(sys.argv) > 2 else TIMEOUT_SEC
    url = "http://{}:{}/api/health".format(HOST, port)
    deadline = time.time() + timeout_sec
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            with urllib.request.urlopen(url, timeout=PROBE_TIMEOUT) as resp:
                if resp.status == 200:
                    print("ready: HTTP 200 after {} probes".format(attempt))
                    return 0
        except (urllib.error.URLError, urllib.error.HTTPError,
                OSError, socket.timeout):
            pass
        time.sleep(INTERVAL_SEC)
    print("not-ready: no HTTP 200 within {}s".format(timeout_sec), file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
