"""Hỏi đáp end-to-end trên tài liệu (đủ 6 bước).

    python ask.py "What is the maximum wind speed for opening the dome?"
    python ask.py "..." --strategy fixed_256 --show-chunks
    python ask.py --interactive

Chưa có ANTHROPIC_API_KEY -> tự động DRY RUN: in ra prompt thay vì gọi API.
"""
from __future__ import annotations

import argparse
import sys

import config
from rag.embedder import build_embedder
from rag.llm import LLMRequestError, get_llm, load_dotenv
from rag.loader import load_document
from rag.pipeline import RAGPipeline
from rag.tokenizer import get_tokenizer


def configure_utf8_output() -> None:
    """Keep Vietnamese console output usable on legacy Windows code pages."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def main():
    configure_utf8_output()
    p = argparse.ArgumentParser()
    p.add_argument("question", nargs="?")
    p.add_argument("--document", default=str(config.DOCUMENT_PATH))
    p.add_argument("--strategy", default=config.DEFAULT_STRATEGY, choices=list(config.STRATEGIES))
    p.add_argument("--embedder", choices=["st", "hashing"], default="st")
    p.add_argument("--preset", default=config.EMBEDDING_PRESET, choices=list(config.EMBEDDING_PRESETS))
    p.add_argument("--k", type=int, default=config.TOP_K)
    p.add_argument("--show-chunks", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="không gọi API, chỉ in prompt")
    p.add_argument("--interactive", "-i", action="store_true")
    args = p.parse_args()

    load_dotenv(config.ROOT / ".env")
    preset = config.EMBEDDING_PRESETS[args.preset]
    tok_name = config.CHUNK_TOKENIZER or ("simple" if args.embedder == "hashing" else preset["model_name"])

    llm = get_llm(config.DEFAULT_LLM_MODEL, config.MAX_OUTPUT_TOKENS, config.TEMPERATURE, args.dry_run)
    pipe = RAGPipeline(build_embedder(args.embedder, preset), get_tokenizer(tok_name), llm, args.k)

    text = load_document(args.document)                                        # Bước 1a
    chunks = pipe.index(text, args.strategy, config.STRATEGIES[args.strategy])  # Bước 1b-3
    print(f"Indexed {len(chunks)} chunks ({args.strategy}) | LLM: {llm.model}\n")

    def run(q: str):
        try:
            out = pipe.answer(q)                                                 # Bước 4-6
        except LLMRequestError as exc:
            print(f"LLM error: {exc}", file=sys.stderr)
            return
        if args.show_chunks:
            for r in out["results"]:
                preview = r.chunk.text[:200].replace("\n", " ")
                print(f"  #{r.rank} score={r.score:.3f} chunk={r.chunk.id}: {preview}...")
            print()
        print(out["answer"], "\n")

    if args.interactive:
        while True:
            q = input("Question (Enter để thoát): ").strip()
            if not q:
                break
            run(q)
    elif args.question:
        run(args.question)
    else:
        p.error("Cần truyền câu hỏi hoặc --interactive")


if __name__ == "__main__":
    main()
