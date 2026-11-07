"""RAG cho tài liệu chính sách / hướng dẫn sản phẩm: Hybrid Search + Reranking.

    ingest  : đọc data/docs/*.md -> tách theo heading (##) -> chunk theo ký tự, có overlap.
              Mỗi chunk mang theo "breadcrumb" (tên tài liệu > heading) để chunk nhỏ vẫn đủ ngữ cảnh.
    sparse  : BM25 tự viết (không cần thư viện). Token hoá = chữ thường + thêm bản bỏ dấu,
              nên "doi tra" vẫn khớp "đổi trả".
    dense   : sentence-transformers (mặc định multilingual-e5-small). Không cài được -> bỏ nhánh này.
    rewrite : (tuỳ chọn, support/query_rewriter.py) câu hỏi -> nhiều truy vấn + từ khoá BM25.
    fuse    : Reciprocal Rank Fusion  score = Σ 1 / (k + rank_i), cộng trên MỌI danh sách xếp hạng
              (BM25 và dense của từng truy vấn viết lại) -> multi-query retrieval.
    rerank  : cross-encoder (bge-reranker-v2-m3) chấm theo câu hỏi đầy đủ. Không cài được -> giữ thứ tự RRF.

Mỗi lần search trả về cả điểm từng nhánh để trace / debug được vì sao chunk lọt top.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import config
from support.logging_setup import get_logger
from support.tracing import NULL_TRACER, current_tracer

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Ingest
# --------------------------------------------------------------------------- #
@dataclass
class Chunk:
    id: str
    doc: str
    section: str
    text: str

    @property
    def full_text(self) -> str:
        return f"[{self.doc} > {self.section}]\n{self.text}"


def _split_long(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]
    # cắt ở ranh giới đoạn/dòng gần nhất để không cắt giữa câu
    parts, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            cut = text.rfind("\n", start + size // 2, end)
            end = cut if cut != -1 else end
        parts.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [p for p in parts if p]


def ingest(docs_dir: Path = config.DOCS_DIR, size: int = config.CHUNK_CHARS,
           overlap: int = config.CHUNK_OVERLAP) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        title_m = re.search(r"^# (.+)$", raw, re.M)
        title = title_m.group(1).strip() if title_m else path.stem
        sections = re.split(r"^## ", raw, flags=re.M)
        for i, sec in enumerate(sections):
            if i == 0:
                body = re.sub(r"^# .+$", "", sec, flags=re.M).strip()
                heading = "Giới thiệu"
            else:
                heading, _, body = sec.partition("\n")
                heading, body = heading.strip(), body.strip()
            if not body:
                continue
            for j, piece in enumerate(_split_long(body, size, overlap)):
                chunks.append(Chunk(id=f"{path.stem}#{i}.{j}", doc=title, section=heading, text=piece))
    return chunks


# --------------------------------------------------------------------------- #
# Sparse: BM25
# --------------------------------------------------------------------------- #
from support.state import strip_accents  # noqa: E402,F401  (định nghĩa ở state.py để training không phải kéo rag.py)

_TOKEN = re.compile(r"[\w]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    toks = _TOKEN.findall(text.lower())
    out = []
    for t in toks:
        out.append(t)
        plain = strip_accents(t)
        if plain != t:
            out.append(plain)
    # bigram giúp khớp cụm từ tiếng Việt ("đổi trả", "bảo hành")
    out += [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
    return out


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avgdl = sum(self.len) / max(1, len(docs))
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for tf, dl in zip(self.tf, self.len):
            s = 0.0
            for t in query:
                if t not in tf:
                    continue
                f = tf[t]
                s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out


# --------------------------------------------------------------------------- #
# Dense + reranker (optional)
# --------------------------------------------------------------------------- #
class DenseIndex:
    def __init__(self, chunks: list[Chunk], model_name: str = config.EMBED_MODEL):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.e5 = "e5" in model_name
        texts = [("passage: " if self.e5 else "") + c.full_text for c in chunks]
        self.emb = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    def scores(self, query: str) -> list[float]:
        q = self.model.encode([("query: " if self.e5 else "") + query], normalize_embeddings=True,
                              convert_to_numpy=True)[0]
        return (self.emb @ q).tolist()


class Reranker:
    def __init__(self, model_name: str = config.RERANK_MODEL):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, max_length=512)

    def scores(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self.model.predict([(query, t) for t in texts])]


def _ranks(scores: list[float], top: int) -> dict[int, int]:
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return {idx: r + 1 for r, idx in enumerate(order[:top]) if scores[idx] > 0}


@dataclass
class HybridRetriever:
    chunks: list[Chunk] = field(default_factory=ingest)
    use_dense: bool = True
    use_rerank: bool = True

    def __post_init__(self):
        self.bm25 = BM25([tokenize(c.full_text) for c in self.chunks])
        self.dense = self.reranker = None
        if self.use_dense:
            try:
                self.dense = DenseIndex(self.chunks)
            except Exception as e:  # thiếu sentence-transformers / không tải được model
                log.warning("dense search disabled", extra={"error": f"{type(e).__name__}: {e}"})
        if self.use_rerank:
            try:
                self.reranker = Reranker()
            except Exception as e:
                log.warning("reranker disabled", extra={"error": f"{type(e).__name__}: {e}"})
        log.info("rag index ready", extra={"chunks": len(self.chunks), "mode": self.mode})

    @property
    def mode(self) -> str:
        return "+".join(["bm25"] + (["dense"] if self.dense else []) + (["rerank"] if self.reranker else []))

    def search(self, query: str, top_k: int = config.TOP_K_FINAL) -> list[dict]:
        """Một truy vấn (giữ nguyên hành vi cũ)."""
        return self.search_multi([query], top_k=top_k)

    def search_multi(self, queries: list[str], top_k: int = config.TOP_K_FINAL, keywords: list[str] | None = None,
                     rerank_query: str | None = None) -> list[dict]:
        """Multi-query hybrid search.

        queries      : truy vấn đầu là truy vấn chính; mỗi truy vấn chạy BM25 + dense, mọi danh sách vào chung RRF.
        keywords     : cộng vào token BM25 của truy vấn chính (không đưa vào dense vì không phải câu tự nhiên).
        rerank_query : câu cross-encoder dùng để chấm (mặc định = queries[0]).
        """
        queries = [q for q in queries if q and q.strip()] or [""]
        n = config.TOP_K_RETRIEVE
        tracer = current_tracer() or NULL_TRACER
        with tracer.span("rag.search", kind="rag", mode=self.mode, queries=queries, keywords=keywords or []) as sp:
            rank_lists: list[tuple[str, dict[int, int]]] = []
            for qi, q in enumerate(queries):
                toks = tokenize(q) + (tokenize(" ".join(keywords)) if keywords and qi == 0 else [])
                rank_lists.append(("bm25", _ranks(self.bm25.scores(toks), n)))
                if self.dense:
                    rank_lists.append(("dense", _ranks(self.dense.scores(q), n)))

            fused: dict[int, float] = {}
            best: dict[str, dict[int, int]] = {"bm25": {}, "dense": {}}
            for branch, ranks in rank_lists:
                for idx, r in ranks.items():
                    fused[idx] = fused.get(idx, 0.0) + 1.0 / (config.RRF_K + r)
                    if r < best[branch].get(idx, 10**9):
                        best[branch][idx] = r
            # multi-query có nhiều ứng viên hơn -> nới pool cho reranker theo số truy vấn
            pool = max(top_k * 3, top_k) + top_k * (len(queries) - 1)
            cands = sorted(fused, key=fused.get, reverse=True)[:pool]
            rerank = {}
            if self.reranker and cands:
                rs = self.reranker.scores(rerank_query or queries[0], [self.chunks[i].full_text for i in cands])
                rerank = dict(zip(cands, rs))
                cands.sort(key=lambda i: rerank[i], reverse=True)
            hits = [{
                "chunk_id": self.chunks[i].id, "doc": self.chunks[i].doc, "section": self.chunks[i].section,
                "text": self.chunks[i].text, "rrf": round(fused[i], 5),
                "bm25_rank": best["bm25"].get(i), "dense_rank": best["dense"].get(i),
                "rerank_score": round(rerank[i], 4) if i in rerank else None,
            } for i in cands[:top_k]]
            sp.set(hits=[h["chunk_id"] for h in hits], n_candidates=len(fused))
            return hits


_RETRIEVER: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    """Singleton: model embedding / reranker chỉ load một lần. RAG_LIGHT=1 -> chỉ BM25."""
    global _RETRIEVER
    if _RETRIEVER is None:
        import os

        light = os.environ.get("RAG_LIGHT") == "1"
        _RETRIEVER = HybridRetriever(use_dense=not light, use_rerank=not light)
    return _RETRIEVER
