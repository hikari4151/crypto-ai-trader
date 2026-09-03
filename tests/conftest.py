"""pytest 全局配置：把项目根目录加入 sys.path，使 tests/ 下脚本可直接 import 项目模块。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 脚本式验证文件（顶层直接执行并 sys.exit），仅手动运行，禁止被 pytest 收集
collect_ignore = ["test_ai_guards.py"]
