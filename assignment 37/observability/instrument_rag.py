"""Instrument RAG pipeline của assignment 3.2 bằng subclass (không sửa code gốc).

- TracedClaudeLLM.generate : 1 generation / LLM call (prompt hash, model, tokens, latency, cost)
- TracedRAGPipeline.retrieve: 1 retrieval (query, top-k chunks + similarity scores, thời gian embed/search)
- TracedRAGPipeline.answer  : 1 trace bao cả hai, đánh dấu failed + lý do
"""
from __future__ import annotations

import time

import _bootstrap  # noqa: F401
from rag.llm import ClaudeLLM, DryRunLLM
from rag.pipeline import RAGPipeline

from .tracer import Tracer

NO_ANSWER_MARKER = "I don't know based on the provided document"
LOW_SCORE_THRESHOLD = 0.55       # top-1 cosine thấp hơn mức này -> retrieval có thể đã trượt (bge-small)


class TracedClaudeLLM(ClaudeLLM):
    def __init__(self, tracer: Tracer, model: str, max_tokens: int = 512):
        super().__init__(model, max_tokens)
        self.tracer = tracer

    def generate(self, system: str, messages: list[dict]) -> str:
        with self.tracer.generation("rag.generate", self.model, system, messages,
                                    model_parameters={"max_tokens": self.max_tokens}) as g:
            resp = self.client.messages.create(model=self.model, max_tokens=self.max_tokens,
                                               system=system, messages=messages)
            text = "".join(b.text for b in resp.content if b.type == "text")
            u = resp.usage
            g.set_usage(resp.model, u.input_tokens, u.output_tokens,
                        getattr(u, "cache_read_input_tokens", 0) or 0,
                        getattr(u, "cache_creation_input_tokens", 0) or 0)
            g.update(output=text, metadata={"stop_reason": resp.stop_reason,
                                            "request_id": getattr(resp, "_request_id", None)})
        return text


class TracedDryRunLLM(DryRunLLM):
    """Không có API key: vẫn sinh generation record (tokens ước tính = ký tự / 4, cost = 0) để thử dashboard."""

    def __init__(self, tracer: Tracer):
        self.tracer = tracer

    def generate(self, system: str, messages: list[dict]) -> str:
        with self.tracer.generation("rag.generate", self.model, system, messages) as g:
            text = super().generate(system, messages)
            n_in = (len(system) + sum(len(str(m["content"])) for m in messages)) // 4
            g.update(model=self.model, input_tokens=n_in, output_tokens=0, cost_usd=0.0,
                     usage_estimated=True, output="[dry run]")
        return text


class TracedRAGPipeline(RAGPipeline):
    def __init__(self, tracer: Tracer, *args, strategy: str = "", low_score: float = LOW_SCORE_THRESHOLD, **kw):
        super().__init__(*args, **kw)
        self.tracer = tracer
        self.strategy = strategy
        self.low_score = low_score

    def retrieve(self, query: str, k: int | None = None):
        k = k or self.top_k
        meta = {"strategy": self.strategy, "embedder": getattr(self.embedder, "name", "?"), "n_chunks": len(self.store)}
        with self.tracer.retrieval("rag.retrieve", query, k, metadata=meta) as obs:
            t0 = time.perf_counter()
            q = self.embedder.embed_query(query)
            t1 = time.perf_counter()
            results = self.store.search(q, k)
            t2 = time.perf_counter()
            chunks = [{"rank": r.rank, "chunk_id": r.chunk.id, "score": round(r.score, 4),
                       "start": r.chunk.start, "end": r.chunk.end,
                       "preview": r.chunk.text[:160].replace("\n", " ")} for r in results]
            scores = [c["score"] for c in chunks] or [0.0]
            obs.update(
                output=chunks,
                retrieval={"query": query, "top_k": k, "chunks": chunks, "max_score": max(scores),
                           "min_score": min(scores), "mean_score": round(sum(scores) / len(scores), 4)},
                embed_ms=round((t1 - t0) * 1000, 2), search_ms=round((t2 - t1) * 1000, 2),
            )
        return results

    def answer(self, query: str, k: int | None = None, metadata: dict | None = None) -> dict:
        with self.tracer.trace("rag.query", input={"query": query}, tags=["rag", self.strategy],
                               metadata=metadata, session_id=(metadata or {}).get("session")) as t:
            out = super().answer(query, k)        # gọi self.retrieve + self.llm.generate -> được lồng vào trace này
            reasons = []
            top = out["results"][0].score if out["results"] else 0.0
            if top < self.low_score:
                reasons.append(f"low_retrieval_score({top:.2f})")
            if out["answer"] and NO_ANSWER_MARKER.lower() in out["answer"].lower():
                reasons.append("no_answer")
            t.update(output=out["answer"], query=query, failed=bool(reasons),
                     failure_reason=", ".join(reasons) or None, top_score=round(top, 4))
        return out
