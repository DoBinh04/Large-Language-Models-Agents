"""Agent dạng StateGraph.

            ┌──────────────────────────────────────────────┐
            ▼                                              │
  START → agent ──(không gọi tool / hết bước)──→ END       │
            │                                              │
            ├──(chỉ tool an toàn)──────────────→ tools ────┤
            │                                              │
            └──(có write_file / send_email)→ human_approval → tools

- agent           : gọi Claude với danh sách tool -> trả về assistant message (+ tool_use blocks)
- human_approval  : interrupt() — graph DỪNG, lưu state vào checkpointer, chờ người duyệt.
                    Khi resume bằng Command(resume=decision), node chạy lại từ đầu và
                    interrupt() trả về decision.
- tools           : chạy tool; tool nguy hiểm bị từ chối -> tool_result is_error cho model biết.

Vì interrupt cần lưu state giữa 2 lần invoke, graph BẮT BUỘC compile với checkpointer
và mỗi lần gọi phải có config {"configurable": {"thread_id": ...}}.
"""
from __future__ import annotations

import operator
import os
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .tools import DANGEROUS_TOOLS, TOOL_SCHEMAS, describe_call, run_tool

try:
    from langgraph.checkpoint.memory import InMemorySaver
except ImportError:  # bản langgraph cũ
    from langgraph.checkpoint.memory import MemorySaver as InMemorySaver

DEFAULT_MODEL = "claude-opus-5-5"
MAX_STEPS = 12

SYSTEM_PROMPT = """You are a helpful agent that solves multi-step tasks with tools.
- Use calculate for all arithmetic and get_current_time for anything about today/now.
- Use search_web for facts that may have changed; do not rely on memory for them.
- write_file and send_email have side effects. A human must approve them; if a call is rejected,
  do not retry the same call. Read the reviewer's feedback, adjust, or explain to the user.
- Reply in the same language as the user."""


class AgentState(TypedDict):
    # reducer operator.add: mỗi node chỉ trả về message MỚI, LangGraph tự nối vào list
    messages: Annotated[list[dict], operator.add]
    pending_tool_calls: list[dict]          # tool_use blocks của lượt assistant gần nhất
    approvals: dict[str, dict]              # tool_use_id -> {"approved": bool, "feedback": str, "input": dict|None}
    steps: int
    final_answer: str | None


class ClaudeNode:
    """Node 'agent'. Tách thành class để dễ thay LLM (test) hoặc thêm tracing."""

    def __init__(self, model: str | None = None, effort: str = "medium", max_tokens: int = 8000):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("ANTHROPIC_MODEL") or DEFAULT_MODEL
        self.effort = effort
        self.max_tokens = max_tokens

    def __call__(self, state: AgentState) -> dict:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=state["messages"],
            output_config={"effort": self.effort},
        )
        content = [b.to_dict() for b in resp.content]       # giữ cả thinking blocks để gửi lại
        calls = [b for b in content if b["type"] == "tool_use"]
        text = "".join(b.get("text", "") for b in content if b["type"] == "text")
        return {
            "messages": [{"role": "assistant", "content": content}],
            "pending_tool_calls": calls if resp.stop_reason == "tool_use" else [],
            "approvals": {},
            "steps": state.get("steps", 0) + 1,
            "final_answer": None if calls and resp.stop_reason == "tool_use" else text,
        }


def human_approval(state: AgentState) -> dict:
    """Dừng graph và hỏi người. Mọi thứ TRƯỚC interrupt() sẽ chạy lại khi resume -> phải thuần (không side effect)."""
    dangerous = [c for c in state["pending_tool_calls"] if c["name"] in DANGEROUS_TOOLS]
    decision: Any = interrupt({
        "type": "approval_request",
        "tool_calls": [{"id": c["id"], "name": c["name"], "input": c["input"], "preview": describe_call(c)}
                       for c in dangerous],
    })
    return {"approvals": normalize_decision(decision, dangerous)}


def normalize_decision(decision: Any, calls: list[dict]) -> dict[str, dict]:
    """Chấp nhận nhiều dạng resume value:
       True / False                                   -> duyệt / từ chối tất cả
       {"approve_all": bool, "feedback": str}
       {tool_use_id: {"approved": bool, "feedback": str, "input": {...sửa tham số...}}}
    """
    if isinstance(decision, bool):
        decision = {"approve_all": decision}
    out = {}
    for c in calls:
        if isinstance(decision, dict) and "approve_all" in decision:
            d = {"approved": bool(decision["approve_all"]), "feedback": decision.get("feedback", "")}
        else:
            d = dict(decision.get(c["id"], {"approved": False, "feedback": "no decision"}))
        out[c["id"]] = {"approved": bool(d.get("approved")), "feedback": d.get("feedback", ""),
                        "input": d.get("input")}
    return out


def tools_node(state: AgentState) -> dict:
    results = []
    for call in state["pending_tool_calls"]:
        args = call["input"]
        if call["name"] in DANGEROUS_TOOLS:
            d = state.get("approvals", {}).get(call["id"])
            if not d or not d["approved"]:
                fb = (d or {}).get("feedback") or "no reason given"
                results.append({"type": "tool_result", "tool_use_id": call["id"], "is_error": True,
                                "content": f"REJECTED by human reviewer. Feedback: {fb}"})
                continue
            args = d.get("input") or args          # người duyệt có thể sửa tham số
        out, is_error = run_tool(call["name"], args)
        results.append({"type": "tool_result", "tool_use_id": call["id"], "content": out, "is_error": is_error})
    # TẤT CẢ tool_result của một lượt phải nằm trong MỘT user message
    return {"messages": [{"role": "user", "content": results}], "pending_tool_calls": [], "approvals": {}}


def route_after_agent(state: AgentState) -> str:
    calls = state["pending_tool_calls"]
    if not calls:
        return END
    if state["steps"] >= MAX_STEPS:
        return "max_steps"
    if any(c["name"] in DANGEROUS_TOOLS for c in calls):
        return "human_approval"
    return "tools"


def max_steps_node(state: AgentState) -> dict:
    # Gửi tool_result giả cho các tool_use đang treo để lịch sử hội thoại luôn hợp lệ
    results = [{"type": "tool_result", "tool_use_id": c["id"], "is_error": True,
                "content": "Step limit reached; tool not executed."} for c in state["pending_tool_calls"]]
    return {"messages": [{"role": "user", "content": results}], "pending_tool_calls": [],
            "final_answer": f"[Dừng: đã đạt giới hạn {MAX_STEPS} bước]"}


def build_graph(agent_node=None, checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("agent", agent_node or ClaudeNode())
    g.add_node("human_approval", human_approval)
    g.add_node("tools", tools_node)
    g.add_node("max_steps", max_steps_node)

    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route_after_agent,
                            {"tools": "tools", "human_approval": "human_approval",
                             "max_steps": "max_steps", END: END})
    g.add_edge("human_approval", "tools")
    g.add_edge("tools", "agent")
    g.add_edge("max_steps", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())


def new_turn(question: str) -> dict:
    """Input cho một lượt hỏi mới (dùng lại thread_id cũ thì messages được nối tiếp)."""
    return {"messages": [{"role": "user", "content": question}], "pending_tool_calls": [],
            "approvals": {}, "steps": 0, "final_answer": None}


def pending_interrupts(graph, config) -> list:
    """Các interrupt đang chờ của thread (rỗng nếu graph đã chạy xong)."""
    snap = graph.get_state(config)
    return [intr for task in snap.tasks for intr in getattr(task, "interrupts", ())]
