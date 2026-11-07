"""Ghép 6 bước thành một pipeline hoàn chỉnh."""
from __future__ import annotations

from typing import List

from .chunking import Chunk, chunk_document
from .prompt import build_prompt
from .vector_store import InMemoryVectorStore, SearchResult


class RAGPipeline:
    def __init__(self, embedder, tokenizer, llm=None, top_k: int = 5):
        self.embedder = embedder
        self.tokenizer = tokenizer
        self.llm = llm
        self.top_k = top_k
        self.store = InMemoryVectorStore()
        self.chunks: List[Chunk] = []

    # Bước 1 -> 3
    def index(self, text: str, strategy_name: str, strategy_cfg: dict) -> List[Chunk]:
        self.chunks = chunk_document(text, self.tokenizer, strategy_name, strategy_cfg)   # Bước 1
        vectors = self.embedder.embed_passages([c.text for c in self.chunks])            # Bước 2
        self.store = InMemoryVectorStore()
        self.store.add(self.chunks, vectors)                                               # Bước 3
        return self.chunks

    # Bước 4
    def retrieve(self, query: str, k: int | None = None) -> List[SearchResult]:
        q = self.embedder.embed_query(query)
        return self.store.search(q, k or self.top_k)

    # Bước 5 -> 6
    def answer(self, query: str, k: int | None = None) -> dict:
        results = self.retrieve(query, k)
        system, messages = build_prompt(query, results)                                    # Bước 5
        answer = self.llm.generate(system, messages) if self.llm else None                 # Bước 6
        return {"query": query, "answer": answer, "results": results,
                "system": system, "messages": messages}
