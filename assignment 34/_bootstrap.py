"""Cho phép import code của các assignment khác trong C:\\phase3 (tên thư mục có dấu cách
nên không import như package được). Import file này TRƯỚC khi import react_agent."""
import sys
from pathlib import Path

PHASE3 = Path(__file__).resolve().parent.parent
for sub in ("assignment 33",):
    p = str(PHASE3 / sub)
    if p not in sys.path:
        sys.path.insert(0, p)
