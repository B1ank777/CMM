"""将 src/ 目录注册到 sys.path，使得子包可以不带 src. 前缀直接导入。"""
import sys
from pathlib import Path

_src_dir = str(Path(__file__).resolve().parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
