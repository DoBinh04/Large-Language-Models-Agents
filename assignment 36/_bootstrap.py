"""Thêm "assignment 33" vào sys.path để dùng lại search_web và load_dotenv."""
import sys
from pathlib import Path

PHASE3 = Path(__file__).resolve().parent.parent
for sub in ("assignment 33",):
    p = str(PHASE3 / sub)
    if p not in sys.path:
        sys.path.insert(0, p)
