"""Thêm RAG pipeline (assignment 32) và ReAct agent (assignment 33) vào sys.path."""
import sys
from pathlib import Path

PHASE3 = Path(__file__).resolve().parent.parent
RAG_DIR = PHASE3 / "assignment 32"
AGENT_DIR = PHASE3 / "assignment 33"
for p in (RAG_DIR, AGENT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
