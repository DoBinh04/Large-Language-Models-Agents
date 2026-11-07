"""Answer Relevance: cosine similarity giữa embedding của câu trả lời và của câu hỏi.

Đo xem câu trả lời có "nói về đúng chuyện được hỏi" không. Nó KHÔNG đo câu trả lời đúng hay sai.
Lưu ý khi đọc số:
- Cosine của sentence embedding hiếm khi gần 1 kể cả khi trả lời hoàn hảo, vì câu hỏi và câu trả lời
  khác nhau về hình thức. Hãy so sánh TƯƠNG ĐỐI giữa các câu/cấu hình. Ta cũng tính
  cos(question, đáp án chuẩn) làm mốc tham chiếu ("ceiling" thực tế cho câu đó).
- Câu trả lời từ chối thường lặp lại chủ đề câu hỏi nên vẫn có cosine khá cao. Đó là điểm mù của metric này,
  vì vậy cần đọc cùng faithfulness/abstained.
- Ký hiệu trích dẫn [1][2] bị bỏ trước khi embed để không gây nhiễu.

Dùng embedder của assignment 3.2, embed cả hai phía như "passage" (không prefix truy vấn) để so sánh đối xứng.
"""
from __future__ import annotations

import re

import numpy as np

_CITATION = re.compile(r"\[\d+\](?:\[\d+\])*")


def _clean(text: str) -> str:
    return _CITATION.sub("", text or "").strip()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


class AnswerRelevance:
    def __init__(self, embedder):
        self.embedder = embedder

    def __call__(self, question: str, answer: str, reference: str | None = None) -> dict:
        texts = [_clean(question), _clean(answer)] + ([_clean(reference)] if reference else [])
        vecs = self.embedder.embed_passages(texts)
        out = {"score": cosine(vecs[0], vecs[1])}
        if reference:
            out["reference_score"] = cosine(vecs[0], vecs[2])          # cos(question, đáp án chuẩn)
            out["answer_vs_reference"] = cosine(vecs[1], vecs[2])      # proxy cho correctness (tham khảo)
            out["normalized"] = out["score"] / out["reference_score"] if out["reference_score"] > 0 else None
        return out
