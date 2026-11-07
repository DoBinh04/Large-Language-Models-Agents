"""Chạy RAG pipeline (3.2) trên golden set 20 câu và chấm 3 metric.

    python run_evals.py                                   # strategy mặc định (recursive), k=5
    python run_evals.py --strategy fixed_512 --k 3
    python run_evals.py --no-llm-precision                # chỉ dùng nhãn ground-truth cho context precision
    python run_evals.py --reuse results/<run>/details.jsonl   # chấm lại câu trả lời cũ, không sinh lại
    python run_evals.py --ids G06 G13 G20

Cần ANTHROPIC_API_KEY (generator + judge). Model mặc định: Claude Haiku 5.5 cho cả hai.
    generator = --gen-model   > ANTHROPIC_MODEL    > claude-haiku-5-5
    judge     = --judge-model > EVAL_JUDGE_MODEL   > claude-haiku-5-5

Kết quả: results/<timestamp>/ summary.md, per_question.csv, details.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from datetime import datetime
from pathlib import Path

import _bootstrap
import config                                          # config của assignment 32
from rag.embedder import build_embedder
from rag.evaluation import EvidenceLocator
from rag.llm import ClaudeLLM, load_dotenv
from rag.loader import load_document
from rag.pipeline import RAGPipeline
from rag.tokenizer import get_tokenizer

from evals.answer_relevance import AnswerRelevance
from evals.context_precision import agreement, ground_truth_labels, llm_labels, precision_metrics
from evals.faithfulness import faithfulness
from evals.judge import Judge

ROOT = Path(__file__).resolve().parent

DEFAULT_GEN_MODEL = "claude-haiku-5-5"
# Haiku 5.5 bật adaptive thinking mặc định, token thinking tính vào max_tokens.
# 512 (config 3.2) có thể bị thinking ăn hết -> câu trả lời rỗng/cụt, nên nâng lên.
GEN_MAX_TOKENS = 4096


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--golden", default=str(ROOT / "data" / "golden_set.json"))
    p.add_argument("--strategy", default=config.DEFAULT_STRATEGY, choices=list(config.STRATEGIES))
    p.add_argument("--k", type=int, default=config.TOP_K)
    p.add_argument("--embedder", choices=["st", "hashing"], default="st")
    p.add_argument("--gen-model", default=None)
    p.add_argument("--judge-model", default=None)
    p.add_argument("--no-llm-precision", action="store_true")
    p.add_argument("--reuse", default=None, help="details.jsonl cũ: dùng lại câu trả lời, không gọi generator")
    p.add_argument("--ids", nargs="*")
    return p.parse_args()


def mean(values) -> float | None:
    vals = [v for v in values if v is not None]
    return statistics.fmean(vals) if vals else None


def fmt(v, nd=3) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def main():
    args = parse_args()
    load_dotenv(ROOT / ".env")
    load_dotenv(_bootstrap.RAG_DIR / ".env")

    golden = json.loads(Path(args.golden).read_text(encoding="utf-8"))
    if args.ids:
        golden = [g for g in golden if g["id"] in set(args.ids)]
    document = load_document(config.DOCUMENT_PATH)

    # Kiểm tra golden set: mọi evidence phải tìm thấy nguyên văn trong tài liệu
    locator = EvidenceLocator(document)
    for g in golden:
        g["evidence_spans"] = []
        for ev in g["evidence"]:
            span = locator.locate(ev)
            if span is None:
                raise SystemExit(f"{g['id']}: không tìm thấy evidence trong tài liệu: {ev[:80]!r}")
            g["evidence_spans"].append(span)

    preset = config.EMBEDDING_PRESETS[config.EMBEDDING_PRESET]
    tok_name = config.CHUNK_TOKENIZER or ("simple" if args.embedder == "hashing" else preset["model_name"])
    embedder = build_embedder(args.embedder, preset)
    gen_model = args.gen_model or os.environ.get("ANTHROPIC_MODEL") or DEFAULT_GEN_MODEL
    pipe = RAGPipeline(embedder, get_tokenizer(tok_name), ClaudeLLM(gen_model, GEN_MAX_TOKENS), args.k)
    pipe.index(document, args.strategy, config.STRATEGIES[args.strategy])

    judge = Judge(args.judge_model)
    relevance = AnswerRelevance(embedder)
    reused = {}
    if args.reuse:
        for line in Path(args.reuse).read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            reused[d["id"]] = d["answer"]
    print(f"Strategy={args.strategy} k={args.k} | generator={gen_model} | judge={judge.model} | "
          f"embedder={getattr(embedder, 'name', '?')}\n")

    out_dir = ROOT / "results" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(out_dir / "details.jsonl", "w", encoding="utf-8") as f:
        for g in golden:
            q = g["question"]
            results = pipe.retrieve(q, args.k)
            answer = reused.get(g["id"])
            if answer is None:
                answer = pipe.answer(q, args.k)["answer"]       # retrieve lại cùng query -> cùng kết quả
            contexts = [r.chunk.text for r in results]
            answerable = bool(g["evidence"])

            # 1) Faithfulness (LLM-as-judge)
            faith = faithfulness(judge, q, answer, contexts)
            # 2) Answer relevance (cosine)
            rel = relevance(q, answer, g["answer"])
            # 3) Context precision (ground truth + LLM judge)
            gt = precision_metrics(ground_truth_labels([(r.chunk.start, r.chunk.end) for r in results],
                                                       g["evidence_spans"]))
            llm = None
            if not args.no_llm_precision:
                labels, reasons = llm_labels(judge, q, g["answer"], contexts)
                llm = {**precision_metrics(labels), "reasons": reasons, "agreement_with_gt": agreement(labels, gt["labels"])}

            row = {
                "id": g["id"], "type": g["type"], "answerable": answerable,
                "faithfulness": faith["score"], "abstained": faith["abstained"],
                "n_claims": faith["n_claims"], "n_unsupported": faith["n_claims"] - faith["n_supported"],
                "relevance": rel["score"], "relevance_ref": rel.get("reference_score"),
                "relevance_norm": rel.get("normalized"), "answer_vs_ref": rel.get("answer_vs_reference"),
                # với câu không trả lời được, precision không có nghĩa (không có chunk "đúng") -> None
                "precision_gt": gt["precision"] if answerable else None,
                "ap_gt": gt["ap"] if answerable else None,
                "precision_llm": llm["precision"] if llm and answerable else None,
                "ap_llm": llm["ap"] if llm and answerable else None,
                "label_agreement": llm["agreement_with_gt"] if llm and answerable else None,
                "top_scores": [round(r.score, 3) for r in results],
            }
            rows.append(row)
            f.write(json.dumps({**row, "question": q, "reference": g["answer"], "answer": answer,
                                "faithfulness_claims": faith["claims"], "gt_labels": gt["labels"],
                                "llm_labels": llm and llm["labels"], "llm_reasons": llm and llm["reasons"],
                                "chunks": [{"rank": r.rank, "id": r.chunk.id, "score": r.score,
                                            "text": r.chunk.text} for r in results]},
                               ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{g['id']}] faith={fmt(row['faithfulness'], 2)}{' (abstain)' if row['abstained'] else ''} "
                  f"rel={row['relevance']:.3f} P_gt={fmt(row['precision_gt'], 2)} P_llm={fmt(row['precision_llm'], 2)}")

    with open(out_dir / "per_question.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write_summary(out_dir / "summary.md", rows, args, gen_model, judge)
    print("\n" + (out_dir / "summary.md").read_text(encoding="utf-8"))
    print(f"Đã lưu: {out_dir}")


def write_summary(path: Path, rows: list[dict], args, gen_model: str, judge: Judge) -> None:
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    lines = [
        "# RAG evaluation: golden set", "",
        f"- Strategy `{args.strategy}`, top-k = {args.k}, generator `{gen_model}`, judge `{judge.model}`",
        f"- Judge usage: {judge.calls} calls, {judge.input_tokens:,} in / {judge.output_tokens:,} out tokens", "",
        "| Metric | Giá trị | Ghi chú |", "|---|---|---|",
        f"| **Faithfulness** (LLM judge) | {fmt(mean(r['faithfulness'] for r in rows))} | "
        f"{sum(r['n_unsupported'] for r in rows)} claim không có trong context, trên tổng {sum(r['n_claims'] for r in rows)} claim |",
        f"| **Answer relevance** (cosine) | {fmt(mean(r['relevance'] for r in rows))} | "
        f"mốc cos(question, đáp án chuẩn) = {fmt(mean(r['relevance_ref'] for r in rows))}; "
        f"normalized = {fmt(mean(r['relevance_norm'] for r in rows))} |",
        f"| **Context precision@{args.k}** (ground truth) | {fmt(mean(r['precision_gt'] for r in ans))} | "
        f"AP = {fmt(mean(r['ap_gt'] for r in ans))} (thưởng xếp chunk đúng lên đầu) |",
        f"| Context precision@{args.k} (LLM judge) | {fmt(mean(r['precision_llm'] for r in ans))} | "
        f"AP = {fmt(mean(r['ap_llm'] for r in ans))}; đồng thuận nhãn với ground truth = {fmt(mean(r['label_agreement'] for r in ans))} |",
        f"| Abstention | {sum(r['abstained'] for r in unans)}/{len(unans)} câu không-trả-lời-được đã từ chối đúng | "
        f"{sum(r['abstained'] for r in ans)}/{len(ans)} câu trả-lời-được bị từ chối nhầm |",
        "", "## Theo loại câu hỏi", "",
        "| Loại | n | Faithfulness | Relevance | Precision (GT) | Precision (LLM) |", "|---|---|---|---|---|---|",
    ]
    for t in dict.fromkeys(r["type"] for r in rows):
        g = [r for r in rows if r["type"] == t]
        lines.append(f"| {t} | {len(g)} | {fmt(mean(r['faithfulness'] for r in g))} | {fmt(mean(r['relevance'] for r in g))} | "
                     f"{fmt(mean(r['precision_gt'] for r in g))} | {fmt(mean(r['precision_llm'] for r in g))} |")
    lines += ["", "## Từng câu", "",
              "| ID | Faith | Claims (unsup.) | Abstain | Relevance | Rel/ref | P_gt | AP_gt | P_llm | Agree |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['id']} | {fmt(r['faithfulness'], 2)} | {r['n_claims']} ({r['n_unsupported']}) | "
                     f"{'✓' if r['abstained'] else ''} | {r['relevance']:.3f} | {fmt(r['relevance_norm'], 2)} | "
                     f"{fmt(r['precision_gt'], 2)} | {fmt(r['ap_gt'], 2)} | {fmt(r['precision_llm'], 2)} | {fmt(r['label_agreement'], 2)} |")
    flags = [f"- {r['id']}: faithfulness {r['faithfulness']:.2f}" for r in rows
             if r["faithfulness"] is not None and r["faithfulness"] < 1]
    flags += [f"- {r['id']}: precision (GT) = 0, không chunk nào trong top-{args.k} chứa evidence" for r in ans
              if r["precision_gt"] == 0]
    flags += [f"- {r['id']}: câu trả-lời-được nhưng model từ chối" for r in ans if r["abstained"]]
    lines += ["", "## Cần xem lại (mở details.jsonl)", ""] + (flags or ["- (không có)"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
