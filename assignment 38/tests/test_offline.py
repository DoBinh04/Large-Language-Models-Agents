"""Test không cần API key / model:  python -m tests.test_offline"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _bootstrap  # noqa: E402,F401
import config  # noqa: E402
from rag.embedder import HashingEmbedder  # noqa: E402
from rag.evaluation import EvidenceLocator  # noqa: E402
from rag.loader import load_document  # noqa: E402

from evals.answer_relevance import AnswerRelevance  # noqa: E402
from evals.context_precision import agreement, ground_truth_labels, precision_metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_golden_set_evidence_is_verbatim():
    golden = json.loads((ROOT / "data" / "golden_set.json").read_text(encoding="utf-8"))
    assert len(golden) == 20 and len({g["id"] for g in golden}) == 20
    loc = EvidenceLocator(load_document(config.DOCUMENT_PATH))
    for g in golden:
        for ev in g["evidence"]:
            assert loc.locate(ev) is not None, (g["id"], ev[:60])


def test_precision_metrics():
    m = precision_metrics([True, False, True, False, False])
    assert m["precision"] == 0.4
    assert abs(m["ap"] - (1 / 1 + 2 / 3) / 2) < 1e-9
    assert precision_metrics([False] * 5)["ap"] == 0.0
    # chunk (0,100) chứa 50% evidence (50,150) -> relevant; chunk (140,300) chỉ chứa 10% -> không
    assert ground_truth_labels([(0, 100), (140, 300)], [(50, 150)]) == [True, False]
    assert agreement([True, False], [True, True]) == 0.5


def test_answer_relevance():
    rel = AnswerRelevance(HashingEmbedder())
    good = rel("What temperature is the data centre kept at?", "The data centre is kept at 21 degrees [1].")
    bad = rel("What temperature is the data centre kept at?", "Snow chains are required from November.")
    assert good["score"] > bad["score"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
