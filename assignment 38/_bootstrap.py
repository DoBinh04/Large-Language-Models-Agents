"""Thêm RAG pipeline của assignment 32 vào sys.path."""
import sys
from pathlib import Path

PHASE3 = Path(__file__).resolve().parent.parent
RAG_DIR = PHASE3 / "assignment 32"
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))
