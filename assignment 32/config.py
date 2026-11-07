"""Cấu hình chung cho toàn bộ assignment. Sửa ở đây thay vì sửa trong code."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# ---- Tài liệu & bộ câu hỏi ---------------------------------------------------
DOCUMENT_PATH = DATA_DIR / "sample_document.md"   # đổi sang file .pdf / .txt của bạn
QUESTIONS_PATH = DATA_DIR / "questions.json"

# ---- Embedding model (Bước 2) -------------------------------------------------
# Mỗi preset: tên model + prefix cho query/passage (một số model cần prefix).
EMBEDDING_PRESETS = {
    # Mặc định: tiếng Anh, 384 chiều, max_seq_length = 512.
    # Lưu ý: fixed_512 có thể vượt giới hạn này vì model thêm special tokens.
    "bge-small": {
        "model_name": "BAAI/bge-small-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "passage_prefix": "",
    },
    # Model phổ biến nhất nhưng max_seq_length = 256 -> chunk 512 bị CẮT khi embed
    "minilm": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "query_prefix": "",
        "passage_prefix": "",
    },
    # Dùng khi tài liệu là tiếng Việt / đa ngôn ngữ
    "e5-multilingual": {
        "model_name": "intfloat/multilingual-e5-base",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
}
EMBEDDING_PRESET = "bge-small"

# Tokenizer dùng để đếm token khi chunk. None = dùng tokenizer của embedding model
# (để "256 tokens" đúng bằng 256 token mà model thực sự nhìn thấy).
# "simple" = tách từ bằng regex (không cần tải model, dùng cho test offline).
CHUNK_TOKENIZER = None

# ---- Chunking strategies (Bước 1) --------------------------------------------
STRATEGIES = {
    "fixed_256": {"type": "fixed", "chunk_size": 256, "overlap": 32},
    "fixed_512": {"type": "fixed", "chunk_size": 512, "overlap": 64},
    "recursive": {"type": "recursive", "chunk_size": 300, "overlap": 0},
}
DEFAULT_STRATEGY = "recursive"

# ---- Retrieval (Bước 4) & đánh giá ------------------------------------------
TOP_K = 5
# Một evidence được coi là "đã retrieve" nếu >= tỉ lệ này số ký tự của nó nằm
# trong top-k chunks. Kèm theo đó luôn báo thêm strict recall (ngưỡng 1.0).
EVIDENCE_COVERAGE_THRESHOLD = 0.5

# ---- LLM (Bước 6) -------------------------------------------------------------
# Có thể ghi đè bằng biến môi trường ANTHROPIC_MODEL trong file .env
# Có thể ghi đè bằng ANTHROPIC_MODEL. ID mặc định là model Haiku API đang active.
DEFAULT_LLM_MODEL = "claude-haiku-4-5-20251001"
MAX_OUTPUT_TOKENS = 512
TEMPERATURE = 0.0
