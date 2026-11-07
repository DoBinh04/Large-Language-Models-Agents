"""Chạy RAG pipeline (3.2) có tracing trên bộ câu hỏi, rồi xem dashboard.

    python run_rag.py                         # 10 câu của 3.2 + 3 câu ngoài phạm vi tài liệu
    python run_rag.py --strategy fixed_256
    python run_rag.py --embedder hashing      # không tải model embedding (nhẹ RAM, chỉ để thử)
    python run_rag.py --dry-run               # không gọi Claude (tokens ước tính, cost = 0)
    python run_rag.py -q "When was the Pathfinder first light?"
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import _bootstrap
import config                                   # config của assignment 32
from rag.embedder import build_embedder
from rag.llm import load_dotenv
from rag.loader import load_document
from rag.tokenizer import get_tokenizer

from observability import get_tracer
from observability.instrument_rag import TracedClaudeLLM, TracedDryRunLLM, TracedRAGPipeline

ROOT = Path(__file__).resolve().parent

# Câu hỏi mà tài liệu KHÔNG trả lời được -> kỳ vọng "I don't know" -> phải hiện trong "top failed queries"
OUT_OF_SCOPE = [
    "Who is the current president of France?",
    "What is the ticket price for the night-time public observing session?",
    "How many exoplanets has HERON discovered?",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-q", "--question", action="append", help="câu hỏi (lặp lại được)")
    p.add_argument("--strategy", default=config.DEFAULT_STRATEGY, choices=list(config.STRATEGIES))
    p.add_argument("--embedder", choices=["st", "hashing"], default="st")
    p.add_argument("--k", type=int, default=config.TOP_K)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-langfuse", action="store_true")
    p.add_argument("--session", default=None, help="Langfuse session id để gom các trace của lần chạy này")
    args = p.parse_args()

    load_dotenv(ROOT / ".env")
    load_dotenv(_bootstrap.RAG_DIR / ".env")
    tracer = get_tracer(ROOT / "logs", use_langfuse=not args.no_langfuse)
    print(f"Langfuse: {'BẬT' if tracer.langfuse.enabled else 'TẮT (chỉ log JSONL)'} | log: {tracer.log_file}")

    preset = config.EMBEDDING_PRESETS[config.EMBEDDING_PRESET]
    tok_name = config.CHUNK_TOKENIZER or ("simple" if args.embedder == "hashing" else preset["model_name"])
    if args.dry_run or not os.environ.get("ANTHROPIC_API_KEY"):
        llm = TracedDryRunLLM(tracer)
    else:
        llm = TracedClaudeLLM(tracer, os.environ.get("ANTHROPIC_MODEL", config.DEFAULT_LLM_MODEL), config.MAX_OUTPUT_TOKENS)

    pipe = TracedRAGPipeline(tracer, build_embedder(args.embedder, preset), get_tokenizer(tok_name), llm,
                             args.k, strategy=args.strategy)
    chunks = pipe.index(load_document(config.DOCUMENT_PATH), args.strategy, config.STRATEGIES[args.strategy])
    print(f"Indexed {len(chunks)} chunks ({args.strategy}) | LLM: {llm.model}\n")

    if args.question:
        questions = [(None, q) for q in args.question]
    else:
        qs = json.loads(Path(config.QUESTIONS_PATH).read_text(encoding="utf-8"))
        questions = [(q["id"], q["question"]) for q in qs] + [(f"OOS{i}", q) for i, q in enumerate(OUT_OF_SCOPE, 1)]

    session = args.session or f"rag-{tracer.run_id}"
    for qid, q in questions:
        try:
            out = pipe.answer(q, metadata={"question_id": qid, "session": session})
            print(f"[{qid}] {q}\n   top={out['results'][0].score:.3f} -> {str(out['answer'])[:160]!r}\n")
        except Exception as e:     # đã được trace ghi lại với status=error
            print(f"[{qid}] LỖI {type(e).__name__}: {e}\n")
    tracer.flush()
    print("Xem dashboard:  python dashboard.py")


if __name__ == "__main__":
    main()
