"""Orchestrator + 3 agent thành 4 node riêng biệt trong một StateGraph.

                 ┌────────────► researcher ─────┐
                 │                              │
  START → orchestrator ───────► writer ─────────┤ (A2A response: Task)
         ▲       │                              │
         │       └────────────► reviewer ───────┤
         └──────────────────────────────────────┘
                 │
                 └── (approved / hết lượt sửa) ──► END

Quy tắc giao tiếp:
- Chỉ orchestrator được gửi request (A2A "message/send") và nó gửi tới AGENT CARD theo skill.
- Agent node chỉ đọc state["outbox"] (request gửi cho nó) và ghi state["inbox"] (Task trả về).
  Agent KHÔNG đọc/ghi research, draft, review trong state. Mọi dữ liệu đi qua Part của message,
  orchestrator là bên duy nhất trích artifact ra và quyết định bước tiếp theo.
- Mọi envelope được nối vào state["mailbox"]: đó là transcript đầy đủ của hệ thống.
"""
from __future__ import annotations

import operator
import uuid
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from . import a2a
from .agents import BaseAgent

try:
    from langgraph.checkpoint.memory import InMemorySaver
except ImportError:
    from langgraph.checkpoint.memory import MemorySaver as InMemorySaver

MAX_REVISIONS = 2


class TeamState(TypedDict):
    topic: str
    language: str
    context_id: str
    mailbox: Annotated[list[dict], operator.add]   # mọi A2A envelope (request + response)
    outbox: dict | None                            # request đang chờ giao cho agent
    inbox: dict | None                             # response mới nhất từ agent
    next: str                                      # node tiếp theo do orchestrator chọn
    # Bộ nhớ làm việc RIÊNG của orchestrator (agent không đụng tới)
    research: dict | None
    draft: str | None
    draft_version: int
    review: dict | None
    reviewed_version: int
    reviews: list[dict]
    status: str
    error: str | None
    final_report: str | None


def initial_state(topic: str, language: str = "Vietnamese") -> TeamState:
    return TeamState(topic=topic, language=language, context_id=uuid.uuid4().hex[:12], mailbox=[],
                     outbox=None, inbox=None, next="", research=None, draft=None, draft_version=0,
                     review=None, reviewed_version=0, reviews=[], status="running", error=None,
                     final_report=None)


class Orchestrator:
    def __init__(self, cards: dict[str, a2a.AgentCard], max_revisions: int = MAX_REVISIONS, log=print):
        self.cards = cards
        self.max_revisions = max_revisions
        self.log = log

    def find_agent(self, skill: str) -> str:
        """Định tuyến theo skill trong Agent Card (không hard-code tên agent)."""
        for name, card in self.cards.items():
            if card.has_skill(skill):
                return name
        raise LookupError(f"không agent nào có skill '{skill}'")

    def _request(self, state: TeamState, skill: str, instruction: str, data: dict) -> dict:
        to = self.find_agent(skill)
        msg = a2a.make_message("user", "orchestrator", to, [a2a.text_part(instruction), a2a.data_part(data)],
                               state["context_id"], skill=skill)
        env = a2a.send_request(msg)
        self.log(f"  ✉  {a2a.summarize_envelope(env)}")
        return {"outbox": env, "inbox": None, "mailbox": [env], "next": to}

    def _finish(self, status: str, report: str | None, error: str | None = None) -> dict:
        self.log(f"  ■ kết thúc: {status}")
        return {"next": END, "outbox": None, "status": status, "final_report": report, "error": error}

    def __call__(self, state: TeamState) -> dict:
        update: dict = {}
        # 1) Tiếp nhận kết quả (Task) mà agent vừa trả về
        if state.get("inbox"):
            task = state["inbox"]["result"]
            sender = task["status"].get("message", {}).get("metadata", {}).get("from")
            if task["status"]["state"] != "completed":
                reason = a2a.get_text(task["status"].get("message", {"parts": []}))
                return self._finish("failed", state.get("draft"), f"{sender}: {reason}")
            if sender == "researcher":
                update["research"] = a2a.artifact_data(task, "research")
            elif sender == "writer":
                update["draft"] = a2a.artifact_data(task, "report")
                update["draft_version"] = state["draft_version"] + 1
            elif sender == "reviewer":
                review = a2a.artifact_data(task, "review")
                update["review"] = review
                update["reviewed_version"] = state["draft_version"]
                update["reviews"] = state["reviews"] + [{"version": state["draft_version"], **review}]
            update["inbox"] = None
        s = {**state, **update}

        # 2) Quyết định bước tiếp theo (policy tất định, dễ test, không tốn token)
        lang = s["language"]
        if s["research"] is None:
            step = self._request(s, "web_research", f"Research this topic thoroughly: {s['topic']}",
                                 {"topic": s["topic"]})
        elif s["draft"] is None:
            step = self._request(s, "write_report", f"Write the report in {lang}.",
                                 {"topic": s["topic"], "research": s["research"]})
        elif s["reviewed_version"] < s["draft_version"]:
            step = self._request(s, "review_report", "Review this draft.",
                                 {"topic": s["topic"], "research": s["research"], "draft": s["draft"]})
        elif s["review"]["approved"]:
            step = self._finish("approved", s["draft"])
        elif s["draft_version"] > self.max_revisions:          # bản đầu + max_revisions lần sửa
            step = self._finish("max_revisions", s["draft"])
        else:
            step = self._request(s, "revise_report", f"Revise the report in {lang} according to the review.",
                                 {"topic": s["topic"], "research": s["research"], "draft": s["draft"],
                                  "review": s["review"]})
        return {**update, **step}


def make_agent_node(agent: BaseAgent, log=print):
    name = agent.card.name

    def node(state: TeamState) -> dict:
        env = state["outbox"]
        msg = env["params"]["message"]
        assert msg["metadata"]["to"] == name, f"message gửi cho {msg['metadata']['to']} nhưng tới {name}"
        log(f"  ⚙  {name} đang xử lý skill={msg['metadata']['skill']} ...")
        task = agent.handle(msg)
        resp = a2a.send_response(env, task)
        log(f"  ✉  {a2a.summarize_envelope(resp)}")
        return {"inbox": resp, "outbox": None, "mailbox": [resp]}

    node.__name__ = f"{name}_node"
    return node


def build_graph(agents: dict[str, BaseAgent], max_revisions: int = MAX_REVISIONS, log=print, checkpointer=None):
    g = StateGraph(TeamState)
    g.add_node("orchestrator", Orchestrator({n: a.card for n, a in agents.items()}, max_revisions, log))
    for name, agent in agents.items():
        g.add_node(name, make_agent_node(agent, log))
        g.add_edge(name, "orchestrator")
    g.add_edge(START, "orchestrator")
    g.add_conditional_edges("orchestrator", lambda s: s["next"], {**{n: n for n in agents}, END: END})
    return g.compile(checkpointer=checkpointer or InMemorySaver())
