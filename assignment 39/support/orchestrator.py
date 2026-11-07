"""Orchestrator: State-aware routing -> plan -> dispatch qua A2A -> tổng hợp câu trả lời.

Một lượt hội thoại:
  1. Slot filling bằng regex (mã đơn, SĐT, mã SP) -> merge vào state.
  2. Router (API / Small LLM / Rule) đọc state + tin nhắn -> RoutingDecision.
  3. Thiếu thông tin -> hỏi lại khách, lưu pending_intent (không gọi agent nào => tiết kiệm chi phí).
  4. agents rỗng (chitchat / out_of_scope) -> trả lời bằng reply của router.
  5. Chạy plan tuần tự; kết quả bước trước được gửi kèm (data part) cho bước sau.
     Agent trả input-required -> dừng, hỏi khách.
  6. 1 agent -> dùng thẳng câu trả lời (không tốn thêm lời gọi LLM); nhiều agent -> Composer gộp lại.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import config
from support import a2a
from support.agents import build_agents
from support.llm import Claude
from support.state import ASK_ORDER_ID, ConversationState, extract_slots  # noqa: F401  (ASK_ORDER_ID re-export)
from support.logging_setup import get_logger
from support.tracing import NULL_TRACER, Tracer

log = get_logger(__name__)

COMPOSER_SYSTEM = """Bạn gộp các câu trả lời của nhiều bộ phận thành MỘT câu trả lời mạch lạc cho khách hàng TechViet.
Giữ nguyên mọi con số, mã đơn, mã phiếu, ngày tháng. Không thêm thông tin mới. Tiếng Việt, ngắn gọn, không nhắc tới "bộ phận"."""



@dataclass
class TurnResult:
    reply: str
    decision: dict
    agent_results: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    trace_id: str = ""


class Orchestrator:
    def __init__(self, router, backend, tracer: Tracer = NULL_TRACER, agent_model: str = config.BASE_MODEL,
                 agent_effort: str = config.BASE_EFFORT, composer: tuple[str, str] | None = None,
                 agent_overrides: dict | None = None, transport=None):
        self.router = router
        self.tracer = tracer
        self.backend = backend
        self.agents = build_agents(backend, agent_model, agent_effort, tracer, agent_overrides)
        self.a2a = a2a.A2AClient(transport or a2a.InProcessTransport(self.agents), tracer)
        cm, ce = composer or (agent_model, "low")
        self.composer = Claude(model=cm, effort=ce, tracer=tracer, name="composer", max_tokens=4000)
        self.state = ConversationState()

    def reset(self) -> None:
        self.state = ConversationState()
        self.backend.reset()

    def agent_cards(self) -> list[dict]:
        return self.a2a.transport.cards()

    # ------------------------------------------------------------------ #
    def chat(self, user_msg: str) -> TurnResult:
        st = self.state
        with self.tracer.span("turn", kind="turn", user=user_msg, turn=st.turn) as root:
            st.merge(extract_slots(user_msg))
            decision = self.router.decide(st, user_msg)
            log.info("routed", extra={"turn": st.turn, "intent": decision["intent"], "agents": decision["agents"],
                                      "missing_info": decision["missing_info"], "pending_intent": st.pending_intent})
            st.merge(decision["entities"])
            res = TurnResult(reply="", decision=decision, trace_id=root.trace_id)

            if decision["missing_info"]:
                st.pending_intent, st.pending_missing = decision["intent"], decision["missing_info"]
                res.reply = decision["reply"] or ASK_ORDER_ID
            elif not decision["agents"]:
                res.reply = decision["reply"] or "TechViet có thể giúp gì thêm cho bạn?"
                st.pending_intent, st.pending_missing = "", []
            else:
                self._run_plan(decision, user_msg, res)

            st.last_agents = decision["agents"] or st.last_agents
            st.add_turn(user_msg, res.reply)
            root.set(reply=res.reply, decision=decision, tool_calls=res.tool_calls)
            return res

    def _run_plan(self, decision: dict, user_msg: str, res: TurnResult) -> None:
        st = self.state
        previous: list[dict] = []
        need_input = None
        for step in decision["plan"]:
            data = {"user_message": user_msg, "entities": {k: v for k, v in st.entities.items() if v},
                    "history": st.history[-2:], "previous_results": previous}
            task = self.a2a.send(step["agent"], step["task"] or user_msg, data, context_id=res.trace_id)
            art = a2a.get_data(task["artifacts"][0]) if task["artifacts"] else {}
            answer = art.get("answer") or a2a.get_text(task["status"]["message"])
            calls = art.get("tool_calls", [])
            res.tool_calls += calls
            res.agent_results.append({"agent": step["agent"], "state": task["status"]["state"], "answer": answer})
            previous.append({"agent": step["agent"], "answer": answer})
            for c in calls:                                  # cập nhật state từ tham số tool đã dùng
                st.merge({k: c["args"].get(k) for k in ("order_id", "phone", "product_id")})
            if task["status"]["state"] == "input-required":
                need_input = answer
                break
            if task["status"]["state"] == "failed":
                log.warning("agent task failed", extra={"agent": step["agent"], "detail": answer})
                answer = "Xin lỗi, hệ thống đang gặp sự cố khi xử lý yêu cầu này."
                previous[-1]["answer"] = answer

        if need_input:
            st.pending_intent, st.pending_missing = decision["intent"], []
            res.reply = need_input if len(previous) == 1 else self._compose(user_msg, previous)
            return
        st.pending_intent, st.pending_missing = "", []
        res.reply = previous[0]["answer"] if len(previous) == 1 else self._compose(user_msg, previous)

    def _compose(self, user_msg: str, parts: list[dict]) -> str:
        body = "\n\n".join(f"[{p['agent']}]\n{p['answer']}" for p in parts)
        with self.tracer.span("compose", kind="compose"):
            return self.composer.text(COMPOSER_SYSTEM, f"Câu hỏi của khách: {user_msg}\n\n{body}")
