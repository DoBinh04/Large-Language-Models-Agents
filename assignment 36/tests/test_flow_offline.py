"""Test luồng điều phối không cần API key:  python -m tests.test_flow_offline

FakeLLM thay Claude: reviewer từ chối bản 1, duyệt bản 2.
Kỳ vọng: researcher → writer → reviewer → writer(revise) → reviewer → END.
"""
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multi_agent.agents import build_agents  # noqa: E402
from multi_agent.graph import build_graph, initial_state  # noqa: E402
from multi_agent.llm import Usage  # noqa: E402


class FakeLLM:
    def __init__(self):
        self.usage = Usage()
        self.n = 0

    def tool_loop(self, system, user, tools, execute, schema=None, **kw):
        return {"summary": "s", "findings": [{"claim": "c", "source_ids": [1]}],
                "sources": [{"id": 1, "title": "t", "url": "http://x"}], "open_questions": []}

    def text(self, system, user):
        self.n += 1
        return f"# Report v{self.n}\nclaim [1]"

    def structured(self, system, user, schema):
        self.n += 1
        if self.n == 1:
            return {"strengths": [], "summary": "chưa đạt",
                    "issues": [{"severity": "major", "description": "thiếu", "suggestion": "bổ sung"}]}
        return {"strengths": ["ok"], "issues": [{"severity": "minor", "description": "văn phong", "suggestion": "gọn hơn"}],
                "summary": "tốt"}


def test_revision_loop():
    quiet = lambda *a, **k: None  # noqa: E731
    agents = build_agents(llm_factory=FakeLLM, log=quiet)
    graph = build_graph(agents, max_revisions=2, log=quiet)
    final = graph.invoke(initial_state("topic"), {"configurable": {"thread_id": uuid.uuid4().hex}})

    route = [env["params"]["message"]["metadata"]["skill"] for env in final["mailbox"] if "method" in env]
    assert route == ["web_research", "write_report", "review_report", "revise_report", "review_report"], route
    assert final["status"] == "approved"
    assert final["final_report"].startswith("# Report v2")
    assert [r["approved"] for r in final["reviews"]] == [False, True]
    # mọi request đều có response với cùng JSON-RPC id
    ids_req = [e["id"] for e in final["mailbox"] if "method" in e]
    ids_resp = [e["id"] for e in final["mailbox"] if "result" in e]
    assert ids_req == ids_resp


def test_max_revisions():
    class AlwaysReject(FakeLLM):
        def structured(self, system, user, schema):
            return {"strengths": [], "summary": "",
                    "issues": [{"severity": "major", "description": "sai", "suggestion": "sửa"}]}

    quiet = lambda *a, **k: None  # noqa: E731
    graph = build_graph(build_agents(llm_factory=AlwaysReject, log=quiet), max_revisions=1, log=quiet)
    final = graph.invoke(initial_state("topic"), {"configurable": {"thread_id": uuid.uuid4().hex}})
    assert final["status"] == "max_revisions" and final["draft_version"] == 2


if __name__ == "__main__":
    test_revision_loop()
    print("OK test_revision_loop")
    test_max_revisions()
    print("OK test_max_revisions")
