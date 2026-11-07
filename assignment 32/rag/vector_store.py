"""Bước 3 + 4: in-memory vector store bằng NumPy và tìm kiếm cosine similarity."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from .chunking import Chunk


@dataclass
class SearchResult:
    rank: int
    score: float
    chunk: Chunk


class InMemoryVectorStore:
    def __init__(self):
        self.chunks: List[Chunk] = []
        self.embeddings: np.ndarray | None = None   # shape (N, d)
        self._norms: np.ndarray | None = None       # ||e_i||, tính sẵn 1 lần

    def __len__(self) -> int:
        return len(self.chunks)

    # ---- Bước 3: lưu ---------------------------------------------------------
    def add(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Số chunk và số embedding không khớp")
        embeddings = np.asarray(embeddings, dtype=np.float32)
        self.chunks.extend(chunks)
        self.embeddings = embeddings if self.embeddings is None else np.vstack([self.embeddings, embeddings])
        self._norms = np.linalg.norm(self.embeddings, axis=1)

    # ---- Bước 4: cosine similarity + top-k -----------------------------------
    def cosine_scores(self, query_vec: np.ndarray) -> np.ndarray:
        """cos(q, e_i) = (e_i · q) / (||e_i|| * ||q||)"""
        q = np.asarray(query_vec, dtype=np.float32)
        denom = self._norms * np.linalg.norm(q) + 1e-12
        return (self.embeddings @ q) / denom

    def search(self, query_vec: np.ndarray, k: int = 5) -> List[SearchResult]:
        if not self.chunks:
            return []
        scores = self.cosine_scores(query_vec)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]          # O(N) chọn k phần tử lớn nhất
        top = top[np.argsort(-scores[top])]                 # rồi mới sort k phần tử đó
        return [SearchResult(rank=r + 1, score=float(scores[i]), chunk=self.chunks[i])
                for r, i in enumerate(top)]

    # ---- Lưu / nạp cache để khỏi embed lại ------------------------------------
    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "embeddings.npy", self.embeddings)
        (d / "chunks.json").write_text(json.dumps([c.to_dict() for c in self.chunks],
                                                  ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> "InMemoryVectorStore":
        d = Path(directory)
        store = cls()
        chunks = [Chunk(**c) for c in json.loads((d / "chunks.json").read_text(encoding="utf-8"))]
        store.add(chunks, np.load(d / "embeddings.npy"))
        return store
