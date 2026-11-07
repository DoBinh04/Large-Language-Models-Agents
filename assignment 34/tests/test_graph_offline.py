"""Test luồng interrupt/resume mà không cần API key:  python -m tests.test_graph_offline

FakeAgent đóng vai Claude: lượt 1 gọi calculate, lượt 2 gọi write_file (nguy hiểm),
lượt 3 trả lời. Kiểm tra: graph dừng đúng chỗ, resume với "từ chối" thì file KHÔNG được ghi,
resume với "duyệt" thì file được ghi.
"""
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _bootstrap  # noqa: F401,E402
from langgraph.types import Command  # noqa: E402

from lg_agent import tools as T  # noqa: E402
from lg_agent.graph import build_graph, new_turn, pending_interrupts  # noqa: E402


class FakeAgent:
    def __init__(self):
        self.turn = 0

    def __call__(self, state):
        self.turn += 1
        if self.turn == 1:
            content = [{"type": "tool_use", "id": "t1", "name": "calculate", "input": {"expr": "6*7"}}]
        elif self.turn == 2:
            content = [{"type": "tool_use", "id": "t2", "name": "write_file",
                        "input": {"path": "test_out.txt", "content": "42"}}]
        else:
            content = [{"type": "text", "text": "xong"}]
        calls = [b for b in content if b["type"] == "tool_use"]
        return {"messages": [{"role": "assistant", "content": content}], "pending_tool_calls": calls,
                "approvals": {}, "steps": state.get("steps", 0) + 1, "final_answer": None if calls else "xong"}


def _run(decision):
    target = T.WORKSPACE / "test_out.txt"
    target.unlink(missing_ok=True)
    graph = build_graph(agent_node=FakeAgent())
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}

    graph.invoke(new_turn("6*7 rồi ghi file"), cfg)
    intr = pending_interrupts(graph, cfg)
    assert len(intr) == 1 and intr[0].value["tool_calls"][0]["name"] == "write_file"
    assert not target.exists()                                  # chưa duyệt -> chưa ghi

    graph.invoke(Command(resume=decision), cfg)
    state = graph.get_state(cfg).values
    assert not pending_interrupts(graph, cfg)
    assert state["final_answer"] == "xong"
    results = [b for m in state["messages"] if isinstance(m["content"], list)
               for b in m["content"] if b["type"] == "tool_result"]
    return target, results


def test_reject():
    target, results = _run({"approve_all": False, "feedback": "không cho ghi"})
    assert not target.exists()
    assert results[0]["content"] == "42"
    assert results[1]["is_error"] and "REJECTED" in results[1]["content"]


def test_approve():
    target, results = _run(True)
    assert target.read_text(encoding="utf-8") == "42"
    assert not results[1]["is_error"]
    target.unlink()


class DangerousOnceAgent:
    """Gọi một tool nguy hiểm, rồi kết thúc sau khi nhận tool_result."""

    def __init__(self, name, tool_input):
        self.name = name
        self.tool_input = tool_input
        self.turn = 0

    def __call__(self, state):
        self.turn += 1
        if self.turn == 1:
            content = [{"type": "tool_use", "id": "danger", "name": self.name,
                        "input": self.tool_input}]
            return {"messages": [{"role": "assistant", "content": content}],
                    "pending_tool_calls": content, "approvals": {}, "steps": 1, "final_answer": None}
        return {"messages": [{"role": "assistant", "content": [{"type": "text", "text": "xong"}]}],
                "pending_tool_calls": [], "approvals": {}, "steps": 2, "final_answer": "xong"}


def _run_one_dangerous(name, tool_input, decision):
    graph = build_graph(agent_node=DangerousOnceAgent(name, tool_input))
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    graph.invoke(new_turn("test"), cfg)
    assert len(pending_interrupts(graph, cfg)) == 1
    graph.invoke(Command(resume=decision), cfg)
    return graph.get_state(cfg).values


def test_rejected_email_creates_no_outbox_file():
    before = set(T.OUTBOX.glob("*.eml")) if T.OUTBOX.exists() else set()
    state = _run_one_dangerous(
        "send_email", {"to": "test@example.com", "subject": "Test", "body": "No send"}, False)
    after = set(T.OUTBOX.glob("*.eml")) if T.OUTBOX.exists() else set()
    assert after == before
    results = [b for m in state["messages"] if isinstance(m["content"], list)
               for b in m["content"] if b["type"] == "tool_result"]
    assert results[0]["is_error"] and "REJECTED" in results[0]["content"]


def test_approved_path_traversal_is_still_blocked():
    outside = T.ROOT.parent / "blocked-by-test.txt"
    outside.unlink(missing_ok=True)
    state = _run_one_dangerous(
        "write_file", {"path": "../../blocked-by-test.txt", "content": "must not write"}, True)
    assert not outside.exists()
    results = [b for m in state["messages"] if isinstance(m["content"], list)
               for b in m["content"] if b["type"] == "tool_result"]
    assert results[0]["is_error"] and "PermissionError" in results[0]["content"]


if __name__ == "__main__":
    test_reject()
    print("OK test_reject")
    test_approve()
    print("OK test_approve")
    test_rejected_email_creates_no_outbox_file()
    print("OK test_rejected_email_creates_no_outbox_file")
    test_approved_path_traversal_is_still_blocked()
    print("OK test_approved_path_traversal_is_still_blocked")
