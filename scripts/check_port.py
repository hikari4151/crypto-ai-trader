"""start.bat 端口检测辅助：退出码 0=端口被占用，1=端口空闲。

独立脚本避免在 cmd for 括号块内嵌复杂 -c 命令（括号冲突导致块解析错乱）。
"""
import socket
import sys


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    s = socket.socket()
    try:
        s.settimeout(0.4)
        r = s.connect_ex(("127.0.0.1", port))
        return 0 if r == 0 else 1
    finally:
        s.close()


if __name__ == "__main__":
    sys.exit(main())
