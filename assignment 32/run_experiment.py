"""So sánh 3 chiến lược chunking trên cùng 1 tài liệu bằng Recall@5.

Chạy thật (cần tải embedding model lần đầu, KHÔNG cần API key):
    python run_experiment.py

Chạy thử offline, không tải model (embedder lexical, chỉ để kiểm tra code):
    python run_experiment.py --embedder hashing

Thêm câu trả lời của LLM cho từng câu hỏi (cần ANTHROPIC_API_KEY trong .env):
    python run_experiment.py --generate
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import config
from rag.chunking import Chunk
from rag.embedder import build_embedder
from rag.evaluation import EvidenceLocator, aggregate, evaluate_question
from rag.llm import get_llm, load_dotenv
from rag.loader import load_document
from rag.pipeline import RAGPipeline
from rag.tokenizer import get_tokenizer


def configure_utf8_output() -> None:
    """Keep Vietnamese console output usable on legacy Windows code pages."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--document", default=str(config.DOCUMENT_PATH))
    p.add_argument("--questions", default=str(config.QUESTIONS_PATH))
    p.add_argument("--embedder", choices=["st", "hashing"], default="st",
                   help="st = sentence-transformers (thật); hashing = test offline")
    p.add_argument("--preset", default=config.EMBEDDING_PRESET, choices=list(config.EMBEDDING_PRESETS))
    p.add_argument("--tokenizer", default=None, help="'simple' hoặc tên HF tokenizer (mặc định theo config)")
    p.add_argument("--strategies", nargs="+", default=list(config.STRATEGIES))
    p.add_argument("--k", type=int, default=config.TOP_K)
    p.add_argument("--extra-k", type=int, nargs="*", default=[1, 3],
                   help="tính thêm Recall@k phụ (mặc định 1 và 3)")
    p.add_argument("--threshold", type=float, default=config.EVIDENCE_COVERAGE_THRESHOLD)
    p.add_argument("--no-overlap", action="store_true",
                   help="ablation: đặt overlap = 0 cho các strategy fixed")
    p.add_argument("--generate", action="store_true", help="gọi LLM trả lời từng câu hỏi")
    p.add_argument("--out", default=None, help="thư mục kết quả (mặc định results/<timestamp>)")
    return p.parse_args()


def contains_whole(chunks: list[Chunk], span) -> bool:
    return any(c.start <= span[0] and span[1] <= c.end for c in chunks)


def main():
    configure_utf8_output()
    args = parse_args()
    load_dotenv(config.ROOT / ".env")
    preset = config.EMBEDDING_PRESETS[args.preset]

    # ---- Tài liệu + câu hỏi --------------------------------------------------
    document = load_document(args.document)
    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    locator = EvidenceLocator(document)
    missing = []
    for q in questions:
        q["evidence_spans"] = []
        for ev in q["evidence"]:
            span = locator.locate(ev)
            if span is None:
                missing.append((q["id"], ev[:80]))
            q["evidence_spans"].append(span)
    if missing:
        print("LỖI: không tìm thấy các evidence sau trong tài liệu (phải trích NGUYÊN VĂN):")
        for qid, ev in missing:
            print(f"  - {qid}: {ev!r}")
        sys.exit(1)

    # ---- Model ---------------------------------------------------------------
    tok_name = args.tokenizer or config.CHUNK_TOKENIZER or (
        "simple" if args.embedder == "hashing" else preset["model_name"])
    print(f"Tokenizer (đếm token khi chunk): {tok_name}")
    tokenizer = get_tokenizer(tok_name)
    print(f"Embedder: {preset['model_name'] if args.embedder == 'st' else 'hashing'} ...")
    embedder = build_embedder(args.embedder, preset)
    llm = get_llm(config.DEFAULT_LLM_MODEL, config.MAX_OUTPUT_TOKENS, config.TEMPERATURE) if args.generate else None
    print(f"Document: {args.document} ({len(document):,} ký tự, {tokenizer.count(document):,} tokens)")
    print(f"Questions: {len(questions)} | k = {args.k} | coverage threshold = {args.threshold}\n")

    out_dir = Path(args.out) if args.out else config.RESULTS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary, per_q_rows, retrieved_log, answers = {}, [], [], []
    for name in args.strategies:
        cfg = dict(config.STRATEGIES[name])
        if args.no_overlap:
            cfg["overlap"] = 0
        pipe = RAGPipeline(embedder, tokenizer, llm=llm, top_k=args.k)
        t0 = time.perf_counter()
        chunks = pipe.index(document, name, cfg)
        index_time = time.perf_counter() - t0

        sizes = [c.n_tokens for c in chunks]
        truncated = sum(embedder.count_model_tokens(c.text) > embedder.max_seq_length for c in chunks)
        all_ev = [s for q in questions for s in q["evidence_spans"]]
        split_ev = sum(not contains_whole(chunks, s) for s in all_ev)
        (out_dir / f"chunks_{name}.json").write_text(
            json.dumps([c.to_dict() for c in chunks], ensure_ascii=False, indent=1), encoding="utf-8")

        rows, extra_rows = [], {kk: [] for kk in args.extra_k}
        t0 = time.perf_counter()
        for q in questions:
            results = pipe.retrieve(q["question"], max([args.k] + args.extra_k))
            all_spans = [(r.chunk.start, r.chunk.end) for r in results]
            for kk in args.extra_k:
                extra_rows[kk].append(evaluate_question(q["evidence_spans"], all_spans[:kk], args.threshold))
            results, spans = results[:args.k], all_spans[:args.k]
            ev = evaluate_question(q["evidence_spans"], spans, args.threshold)
            rows.append(ev)
            per_q_rows.append({"strategy": name, "id": q["id"], "question": q["question"],
                               "recall": ev["recall"], "strict_recall": ev["strict_recall"],
                               "first_relevant_rank": ev["first_relevant_rank"] or "",
                               "coverages": ev["coverages"],
                               "retrieved_chunk_ids": [r.chunk.id for r in results]})
            retrieved_log.append({"strategy": name, "id": q["id"], "question": q["question"],
                                  "top_k": [{"rank": r.rank, "score": round(r.score, 4), "chunk_id": r.chunk.id,
                                             "text": r.chunk.text} for r in results]})
            if llm is not None:
                out = pipe.answer(q["question"], args.k)
                answers.append({"strategy": name, "id": q["id"], "question": q["question"],
                                "ground_truth": q["answer"], "llm_answer": out["answer"]})
        query_time = (time.perf_counter() - t0) / len(questions)

        agg = aggregate(rows)
        for kk in args.extra_k:
            agg[f"recall@{kk}"] = aggregate(extra_rows[kk])["recall@k"]
        agg.update({"n_chunks": len(chunks), "avg_tokens": statistics.mean(sizes),
                    "min_tokens": min(sizes), "max_tokens": max(sizes),
                    "truncated_by_embedder": truncated, "evidence_split_by_boundary": split_ev,
                    "n_evidence": len(all_ev), "index_time_s": index_time, "query_time_s": query_time,
                    "context_tokens_top_k": args.k * statistics.mean(sizes)})
        summary[name] = agg
        print(f"[{name:<10}] chunks={len(chunks):>3}  avg_tok={agg['avg_tokens']:6.1f}  "
              f"Recall@{args.k}={agg['recall@k']:.3f}  strict={agg['strict_recall@k']:.3f}  "
              f"MRR={agg['mrr']:.3f}  " + "  ".join(f"R@{kk}={agg[f'recall@{kk}']:.2f}" for kk in args.extra_k) +
              f"  split_ev={split_ev}/{len(all_ev)}  truncated={truncated}")

    # ---- Ghi kết quả --------------------------------------------------------
    meta = {"document": args.document, "embedder": args.embedder,
            "embedding_model": preset["model_name"] if args.embedder == "st" else "hashing",
            "tokenizer": tok_name, "k": args.k, "threshold": args.threshold,
            "strategies": {n: config.STRATEGIES[n] for n in args.strategies},
            "no_overlap": args.no_overlap}
    (out_dir / "results.json").write_text(json.dumps({"meta": meta, "summary": summary},
                                                     ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out_dir / "per_question.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(per_q_rows[0].keys()))
        w.writeheader()
        w.writerows(per_q_rows)
    with open(out_dir / "retrieved.jsonl", "w", encoding="utf-8") as f:
        for r in retrieved_log:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if answers:
        with open(out_dir / "answers.jsonl", "w", encoding="utf-8") as f:
            for a in answers:
                f.write(json.dumps(a, ensure_ascii=False) + "\n")

    (out_dir / "summary.md").write_text(render_summary(meta, summary, per_q_rows, questions, args),
                                        encoding="utf-8")
    print(f"\nĐã lưu kết quả vào: {out_dir}")


def render_summary(meta, summary, per_q_rows, questions, args) -> str:
    k = args.k
    names = list(summary)
    L = [f"# Kết quả thí nghiệm chunking (k = {k})", "",
         f"- Tài liệu: `{meta['document']}`",
         f"- Embedding model: `{meta['embedding_model']}` | Tokenizer đếm token: `{meta['tokenizer']}`",
         f"- Evidence coverage threshold: {meta['threshold']}",
         f"- Strategies: `{json.dumps(meta['strategies'])}`" + (" — **ablation: overlap = 0**" if meta["no_overlap"] else ""), "",
         "## Bảng tổng hợp", "",
         f"| Strategy | #chunks | avg tok | Recall@{k} | Strict R@{k} | Hit@{k} | MRR | Evidence bị cắt ở biên | Chunk bị embedder cắt | Context tok (top-{k}) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        s = summary[n]
        L.append(f"| {n} | {s['n_chunks']} | {s['avg_tokens']:.0f} | {s['recall@k']:.2f} | "
                 f"{s['strict_recall@k']:.2f} | {s['hit@k']:.2f} | {s['mrr']:.2f} | "
                 f"{s['evidence_split_by_boundary']}/{s['n_evidence']} | {s['truncated_by_embedder']} | "
                 f"{s['context_tokens_top_k']:.0f} |")
    if args.extra_k:
        ks = sorted(set(args.extra_k))
        L += ["", "## Recall ở các k nhỏ hơn (phân biệt rõ hơn khi tài liệu ngắn)", "",
              "| Strategy | " + " | ".join(f"Recall@{kk}" for kk in ks) + f" | Recall@{k} |",
              "|---|" + "---|" * (len(ks) + 1)]
        for n in names:
            L.append(f"| {n} | " + " | ".join(f"{summary[n][f'recall@{kk}']:.2f}" for kk in ks)
                     + f" | {summary[n]['recall@k']:.2f} |")
    L += ["", f"## Recall@{k} theo từng câu hỏi", "",
          "| Q | Loại | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    by = {(r["strategy"], r["id"]): r for r in per_q_rows}
    for q in questions:
        cells = []
        for n in names:
            r = by[(n, q["id"])]
            rank = r["first_relevant_rank"] or "-"
            cells.append(f"{r['recall']:.2f} (rank {rank})")
        L.append(f"| {q['id']} | {q.get('type', '')} | " + " | ".join(cells) + " |")
    L += ["", "_rank = thứ hạng của chunk đầu tiên chứa evidence. Xem `retrieved.jsonl` để đọc top-k chunk._", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
