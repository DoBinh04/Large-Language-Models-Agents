"""Cấu hình chung cho assignment 3.9. Sửa ở đây thay vì sửa trong code."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"
RESULTS_DIR = ROOT / "results"
TRACE_DIR = RESULTS_DIR / "traces"
TRAIN_DATA_DIR = ROOT / "training" / "data"
CHECKPOINT_DIR = ROOT / "training" / "checkpoints"


def load_env(path: Path = ROOT / ".env") -> None:
    """Đọc .env đơn giản (KEY=VALUE), không cần python-dotenv."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()

# ---- Model API (Anthropic) ---------------------------------------------------
# Baseline của đề là gpt-oss-120b @ medium reasoning. Bài này dùng Anthropic key, nên
# baseline = Claude Opus 5.5 @ effort "medium" (effort = mức reasoning của Claude).
BASE_MODEL = os.environ.get("BASE_MODEL", "claude-opus-5-5")
BASE_EFFORT = os.environ.get("BASE_EFFORT", "medium")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-opus-5-5")
JUDGE_EFFORT = os.environ.get("JUDGE_EFFORT", "low")

# Server-side fallback khi model từ chối (stop_reason = "refusal"). Chỉ Opus/Sonnet/Fable hỗ trợ.
USE_FALLBACKS = os.environ.get("USE_FALLBACKS", "1") == "1"
FALLBACK_MODELS = ("claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1")

# Giá USD / 1M token (Claude API first-party). Cache write = 1.25x input (TTL 5 phút).
# Kiểm tra lại https://www.anthropic.com/pricing trước khi nộp báo cáo.
PRICING = {
    #                     input  output  cache_read
    "claude-opus-5-5":   (4.00, 20.00, 0.20),
    "claude-opus-5":     (5.00, 25.00, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20),
    "claude-haiku-5-5":  (0.10, 0.50, 0.01),
    "claude-fable-5-1":  (10.00, 50.00, 0.25),
}
CACHE_WRITE_MULT = 1.25

# Small LLM chạy trên GPU tự host: quy đổi chi phí theo giờ GPU x thời gian xử lý.
GPU_COST_PER_HOUR = float(os.environ.get("GPU_COST_PER_HOUR", "0.35"))   # ~ T4 / L4 thuê theo giờ

# ---- Small LLM ---------------------------------------------------------------
SMALL_BASE_MODEL = os.environ.get("SMALL_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
SFT_ADAPTER = CHECKPOINT_DIR / "sft"
DPO_ADAPTER = CHECKPOINT_DIR / "dpo"

# ---- RAG ---------------------------------------------------------------------
EMBED_MODEL = os.environ.get("EMBED_MODEL", "intfloat/multilingual-e5-small")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
CHUNK_CHARS = 700
CHUNK_OVERLAP = 120
RRF_K = 60
TOP_K_RETRIEVE = 12      # số ứng viên mỗi nhánh (BM25 / dense) trước khi fuse
TOP_K_FINAL = 4          # số chunk trả về sau rerank

# Query rewriting trước khi search (support/query_rewriter.py): off | rules | llm
QUERY_REWRITE = os.environ.get("QUERY_REWRITE", "rules")
REWRITE_MODEL = os.environ.get("REWRITE_MODEL", "claude-haiku-5-5")
REWRITE_EFFORT = os.environ.get("REWRITE_EFFORT", "low")

# ---- Serving (FastAPI / Streamlit) -------------------------------------------
API_ROUTER = os.environ.get("API_ROUTER", "api")            # api | rule | small | sft | dpo
API_TOOLS = os.environ.get("API_TOOLS", "local")            # local | mcp
SESSION_TTL_S = int(os.environ.get("SESSION_TTL_S", "3600"))
MAX_SESSIONS = int(os.environ.get("MAX_SESSIONS", "500"))
API_URL = os.environ.get("API_URL", "http://localhost:8000")  # Streamlit gọi FastAPI ở đây

# ---- Agent loop --------------------------------------------------------------
MAX_TOOL_TURNS = 6
AGENT_MAX_TOKENS = 8000
ROUTER_MAX_TOKENS = 4000

AGENTS = ("product", "order", "policy")
