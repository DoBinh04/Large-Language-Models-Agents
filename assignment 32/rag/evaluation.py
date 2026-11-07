"""Đánh giá retrieval: Recall@k, strict Recall@k, Hit@k, MRR.

Vì 3 chiến lược tạo ra 3 bộ chunk KHÁC NHAU, ground truth không thể là "chunk id".
Thay vào đó mỗi câu hỏi có 1+ đoạn "evidence" trích nguyên văn từ tài liệu.
Ta tìm vị trí ký tự của evidence trong tài liệu, rồi kiểm tra top-k chunk
(mỗi chunk có start/end) phủ được bao nhiêu % ký tự của evidence đó.

  coverage(e)  = |ký tự của e nằm trong hợp các top-k chunk| / |e|
  e được tính là retrieved nếu coverage(e) >= threshold (mặc định 0.5)
  Recall@k(q)  = #evidence retrieved / #evidence của câu q
  Recall@k     = trung bình trên các câu hỏi
  Strict       = như trên nhưng threshold = 1.0 (evidence phải nằm trọn)
  Hit@k        = câu hỏi có ít nhất 1 evidence retrieved
  MRR          = 1 / rank của chunk đầu tiên chạm vào evidence bất kỳ
"""
from __future__ import annotations

from typing import Dict, List, Tuple

Span = Tuple[int, int]


def _normalize_with_map(text: str) -> Tuple[str, List[int]]:
    """lowercase + gộp khoảng trắng; trả về map vị trí chuỗi chuẩn hóa -> chuỗi gốc."""
    out, mapping = [], []
    prev_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if prev_space:
                continue
            out.append(" ")
            prev_space = True
        else:
            out.append(ch.lower())
            prev_space = False
        mapping.append(i)
    return "".join(out), mapping


class EvidenceLocator:
    """Tìm vị trí (start, end) của một đoạn evidence trong tài liệu gốc,
    bỏ qua khác biệt hoa/thường và khoảng trắng / xuống dòng."""

    def __init__(self, document: str):
        self.norm_doc, self.mapping = _normalize_with_map(document)

    def locate(self, evidence: str) -> Span | None:
        norm_ev, _ = _normalize_with_map(evidence.strip())
        idx = self.norm_doc.find(norm_ev)
        if idx == -1 or not norm_ev:
            return None
        return self.mapping[idx], self.mapping[idx + len(norm_ev) - 1] + 1


def _overlap(a: Span, b: Span) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def _covered_chars(span: Span, chunk_spans: List[Span]) -> int:
    """Số ký tự của span nằm trong HỢP các chunk (chunk có thể overlap nhau)."""
    pieces = sorted((max(s, span[0]), min(e, span[1])) for s, e in chunk_spans
                    if _overlap((s, e), span) > 0)
    total, cur_s, cur_e = 0, None, None
    for s, e in pieces:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def evaluate_question(evidence_spans: List[Span], retrieved_spans: List[Span],
                      threshold: float) -> Dict:
    coverages = []
    for ev in evidence_spans:
        coverages.append(_covered_chars(ev, retrieved_spans) / max(1, ev[1] - ev[0]))
    n = len(evidence_spans)
    recall = sum(c >= threshold for c in coverages) / n
    strict = sum(c >= 0.999 for c in coverages) / n
    first_rank = None
    for rank, cs in enumerate(retrieved_spans, start=1):
        if any(_overlap(cs, ev) > 0 for ev in evidence_spans):
            first_rank = rank
            break
    return {
        "recall": recall,
        "strict_recall": strict,
        "hit": float(recall > 0),
        "rr": 1.0 / first_rank if first_rank else 0.0,
        "first_relevant_rank": first_rank,
        "coverages": [round(c, 3) for c in coverages],
    }


def aggregate(rows: List[Dict]) -> Dict:
    n = max(1, len(rows))
    return {
        "recall@k": sum(r["recall"] for r in rows) / n,
        "strict_recall@k": sum(r["strict_recall"] for r in rows) / n,
        "hit@k": sum(r["hit"] for r in rows) / n,
        "mrr": sum(r["rr"] for r in rows) / n,
    }
