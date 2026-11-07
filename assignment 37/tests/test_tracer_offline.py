"""Test tracer + dashboard không cần API key / Langfuse:  python -m tests.test_tracer_offline"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard import compute, load, write_html  # noqa: E402
from observability.pricing import estimate_cost  # noqa: E402
from observability.tracer import Tracer  # noqa: E402


def test_nesting_cost_and_dashboard():
    d = Path(tempfile.mkdtemp())
    tr = Tracer(d, run_id="t", use_langfuse=False)
    with tr.trace("rag.query", input={"query": "q1"}) as t:
        with tr.retrieval("rag.retrieve", "q1", 5) as r:
            r.update(retrieval={"query": "q1", "top_k": 5, "chunks": [], "max_score": 0.4})
        with tr.generation("rag.generate", "claude-haiku-5-5", "sys", [{"role": "user", "content": "hi"}]) as g:
            g.set_usage("claude-haiku-5-5", 1_000_000, 100_000)
        t.update(query="q1", failed=True, failure_reason="low_retrieval_score(0.40)")
    try:
        with tr.trace("react.run", input={"question": "q2"}) as t:
            t.update(query="q2")
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    rows = [json.loads(line) for line in tr.log_file.read_text(encoding="utf-8").splitlines()]
    by_name = {r["name"]: r for r in rows}
    root = by_name["rag.query"]
    assert by_name["rag.generate"]["parent_id"] == root["obs_id"]
    assert by_name["rag.generate"]["trace_id"] == root["trace_id"] == by_name["rag.retrieve"]["trace_id"]
    assert abs(by_name["rag.generate"]["cost_usd"] - (0.10 + 0.05)) < 1e-9     # 1M in × $0.10 + 100K out × $0.50
    assert by_name["react.run"]["status"] == "error" and by_name["react.run"]["failed"]

    s = compute(load(d, None, None))
    assert s["n_traces"] == 2 and s["n_failed"] == 2
    assert abs(s["total_cost"] - 0.15) < 1e-9
    write_html(s, d / "dash.html")
    assert "Top failed queries" in (d / "dash.html").read_text(encoding="utf-8")


def test_pricing_prefix():
    assert estimate_cost("claude-opus-5-5", 1_000_000, 0)["total"] == 4.0
    assert estimate_cost("unknown-model", 1, 1) is None


if __name__ == "__main__":
    test_nesting_cost_and_dashboard()
    print("OK test_nesting_cost_and_dashboard")
    test_pricing_prefix()
    print("OK test_pricing_prefix")
