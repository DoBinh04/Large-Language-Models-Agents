"""Context Precision: trong top-k chunk được retrieve, bao nhiêu % thực sự liên quan?

Hai cách gán nhãn "liên quan" cho từng chunk:
  (a) ground truth : chunk chứa ≥ MIN_EVIDENCE_COVERAGE (30%) ký tự của một evidence trong golden set.
                     Tất định, miễn phí, nhưng có thể bỏ sót chunk hữu ích không trùng evidence.
  (b) LLM judge    : hỏi model "chunk này có giúp trả lời câu hỏi (cho biết đáp án chuẩn) không?".
                     Bắt được chunk hữu ích ngoài evidence, nhưng tốn token và có thể sai.
Báo cáo cả hai cùng tỉ lệ đồng thuận giữa chúng.

Hai con số cho mỗi cách:
  precision@k = #relevant / k
  AP@k        = Σ_i (precision@i · rel_i) / #relevant   (RAGAS "context precision": thưởng cho việc xếp chunk đúng lên đầu)
"""
from __future__ import annotations

from .judge import Judge, format_chunks

MIN_EVIDENCE_COVERAGE = 0.3

SCHEMA = {
    "type": "object",
    "properties": {
        "chunks": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "relevant": {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "required": ["id", "relevant", "reason"],
            "additionalProperties": False,
        }},
    },
    "required": ["chunks"],
    "additionalProperties": False,
}

SYSTEM = """You label retrieved chunks for a retrieval evaluation.
A chunk is RELEVANT if it contains information needed to produce the reference answer
(at least part of it, e.g. one of two required facts). It is NOT relevant if it is only on a similar topic,
or contains numbers or rules about something else. Label every chunk id exactly once."""


def precision_metrics(labels: list[bool]) -> dict:
    k = len(labels)
    n_rel = sum(labels)
    hits, ap = 0, 0.0
    for i, rel in enumerate(labels, 1):
        if rel:
            hits += 1
            ap += hits / i
    return {"precision": n_rel / k if k else 0.0, "ap": ap / n_rel if n_rel else 0.0,
            "n_relevant": n_rel, "k": k, "labels": labels}


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def ground_truth_labels(chunk_spans: list[tuple[int, int]], evidence_spans: list[tuple[int, int]]) -> list[bool]:
    return [any(_overlap(c, ev) >= MIN_EVIDENCE_COVERAGE * (ev[1] - ev[0]) for ev in evidence_spans)
            for c in chunk_spans]


def llm_labels(judge: Judge, question: str, reference_answer: str, chunks: list[str]) -> tuple[list[bool], list[dict]]:
    user = (f"<question>{question}</question>\n<reference_answer>{reference_answer}</reference_answer>\n\n"
            f"<chunks>\n{format_chunks(chunks)}\n</chunks>")
    out = judge.ask(SYSTEM, user, SCHEMA)["chunks"]
    by_id = {c["id"]: c for c in out}
    labels = [bool(by_id.get(i, {}).get("relevant", False)) for i in range(1, len(chunks) + 1)]
    return labels, out


def agreement(a: list[bool], b: list[bool]) -> float:
    return sum(x == y for x, y in zip(a, b)) / len(a) if a else 0.0
