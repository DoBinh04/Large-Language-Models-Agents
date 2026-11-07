"""Test nhanh, không cần internet/model:  python -m tests.test_pipeline"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import config
from rag.chunking import chunk_document
from rag.embedder import HashingEmbedder
from rag.evaluation import EvidenceLocator, evaluate_question
from rag.loader import load_document
from rag.tokenizer import SimpleTokenizer
from rag.vector_store import InMemoryVectorStore

tok = SimpleTokenizer()
doc = load_document(config.DOCUMENT_PATH)


def test_chunks_respect_size_and_offsets():
    for name, cfg in config.STRATEGIES.items():
        chunks = chunk_document(doc, tok, name, cfg)
        assert chunks, name
        for c in chunks:
            assert doc[c.start:c.end] == c.text            # offset khớp tài liệu gốc
            assert c.n_tokens <= cfg["chunk_size"], (name, c.n_tokens)
        # chunk không bỏ sót nội dung: mọi token của tài liệu đều thuộc 1 chunk nào đó
        covered = np.zeros(len(doc), bool)
        for c in chunks:
            covered[c.start:c.end] = True
        assert all(covered[s] for s, _ in tok.offsets(doc)), name


def test_recursive_prefers_paragraph_boundaries():
    chunks = chunk_document(doc, tok, "recursive", config.STRATEGIES["recursive"])
    ends_clean = sum(c.text.rstrip()[-1] in ".:)" or doc[c.end:c.end + 1] in ("\n", "") for c in chunks)
    assert ends_clean / len(chunks) > 0.9


def test_cosine_search():
    store = InMemoryVectorStore()
    from rag.chunking import Chunk
    cs = [Chunk(i, t, 0, 1, 1, "x") for i, t in enumerate(["a", "b", "c"])]
    store.add(cs, np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32))
    res = store.search(np.array([2, 0.1], dtype=np.float32), k=2)
    assert [r.chunk.id for r in res] == [0, 2]
    assert abs(res[0].score - 2 / np.sqrt(4.01)) < 1e-5


def test_evaluation():
    loc = EvidenceLocator("Hello   World.\nFoo bar baz.")
    assert loc.locate("world. foo") == (8, 18)
    r = evaluate_question([(10, 20)], [(0, 12), (12, 16)], threshold=0.5)
    assert r["recall"] == 1.0 and r["strict_recall"] == 0.0 and r["first_relevant_rank"] == 1


def test_end_to_end_hashing():
    from rag.pipeline import RAGPipeline
    pipe = RAGPipeline(HashingEmbedder(), tok, llm=None, top_k=3)
    pipe.index(doc, "recursive", config.STRATEGIES["recursive"])
    out = pipe.answer("What temperature is the data centre kept at?")
    assert any("21 degrees" in r.chunk.text for r in out["results"])
    assert "<context>" in out["messages"][0]["content"]


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("PASS", k)
