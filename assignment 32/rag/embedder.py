"""Bước 2: embed chunk / query thành vector.

SentenceTransformerEmbedder : embedder thật (dùng cho bài nộp).
HashingEmbedder             : bag-of-words băm, KHÔNG phải semantic embedding.
                              Chỉ để chạy thử pipeline offline (không tải model).
"""
from __future__ import annotations

import hashlib
import re
from typing import List

import numpy as np


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, query_prefix: str = "", passage_prefix: str = "",
                 batch_size: int = 32, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.batch_size = batch_size

    @property
    def max_seq_length(self) -> int:
        return int(self.model.max_seq_length)

    def count_model_tokens(self, text: str) -> int:
        """Số token model thực sự nhận (gồm prefix + [CLS]/[SEP])."""
        return len(self.model.tokenizer(self.passage_prefix + text)["input_ids"])

    def _encode(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=len(texts) > 64,
                                 convert_to_numpy=True, normalize_embeddings=False)
        return vecs.astype(np.float32)

    def embed_passages(self, texts: List[str]) -> np.ndarray:
        return self._encode([self.passage_prefix + t for t in texts])

    def embed_query(self, query: str) -> np.ndarray:
        return self._encode([self.query_prefix + query])[0]


class HashingEmbedder:
    """Baseline lexical (TF có log + hashing trick). Chỉ dùng để smoke-test."""

    name = "hashing-bow (offline test only)"
    max_seq_length = 10**9
    _word = re.compile(r"\w+", re.UNICODE)

    def __init__(self, dim: int = 4096):
        self.dim = dim

    def count_model_tokens(self, text: str) -> int:
        return len(self._word.findall(text))

    def _vec(self, text: str) -> np.ndarray:
        words = [w.lower() for w in self._word.findall(text)]
        feats = words + [a + "_" + b for a, b in zip(words, words[1:])]
        v = np.zeros(self.dim, dtype=np.float32)
        for f in feats:
            h = int(hashlib.md5(f.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        return np.log1p(v)

    def embed_passages(self, texts: List[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, query: str) -> np.ndarray:
        return self._vec(query)


def build_embedder(kind: str, preset: dict):
    if kind == "hashing":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(preset["model_name"], preset.get("query_prefix", ""),
                                       preset.get("passage_prefix", ""))
