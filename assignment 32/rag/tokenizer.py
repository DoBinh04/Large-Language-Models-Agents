"""Tokenizer có trả về vị trí ký tự (offset) của từng token.

Cần offset để mỗi chunk biết chính xác nó nằm ở đâu trong tài liệu gốc
(start/end), nhờ đó đánh giá Recall@k công bằng giữa các chiến lược chunk khác nhau.
"""
from __future__ import annotations

import re
from typing import List, Tuple

Span = Tuple[int, int]


class SimpleTokenizer:
    """Tách từ + dấu câu bằng regex. Không cần tải gì, dùng cho test offline."""

    name = "simple"
    _pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def offsets(self, text: str) -> List[Span]:
        return [m.span() for m in self._pattern.finditer(text)]

    def count(self, text: str) -> int:
        return sum(1 for _ in self._pattern.finditer(text))


class HFTokenizer:
    """Tokenizer của HuggingFace (fast tokenizer) - cùng tokenizer với embedding model."""

    def __init__(self, model_name: str):
        from transformers import AutoTokenizer

        self.name = model_name
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.tok.model_max_length = 10**9  # tắt cảnh báo "sequence too long" khi đếm

    def offsets(self, text: str) -> List[Span]:
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        # Bỏ các token rỗng (offset (0,0)) nếu có
        return [(s, e) for s, e in enc["offset_mapping"] if e > s]

    def count(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"])


def get_tokenizer(name: str):
    if name == "simple":
        return SimpleTokenizer()
    return HFTokenizer(name)
