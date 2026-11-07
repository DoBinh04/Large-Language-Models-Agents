"""Conversation state cho State-aware Orchestration.

State được giữ qua nhiều lượt và là input của Router:
- entities     : order_id / phone / product đã biết (slot filling). Regex bắt slot chắc chắn
                 (mã đơn DHxxxx, SĐT 10 số) ngay từ câu khách; router LLM bổ sung phần còn lại.
- pending      : intent còn dở vì thiếu thông tin (vd. khách muốn đổi trả nhưng chưa đưa mã đơn).
                 Lượt sau khách chỉ gõ "DH1005" -> router phải hiểu là tiếp tục đổi trả.
- last_agents  : agent vừa xử lý lượt trước (câu hỏi nối tiếp "còn bảo hành thì sao?").
- history      : vài lượt gần nhất (rút gọn) để router hiểu đại từ "nó", "cái đó".

`router_view()` là đúng chuỗi JSON router nhìn thấy — dùng chung cho API router, Small LLM và
khi sinh dữ liệu train, để train/inference không lệch format.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field

ASK_ORDER_ID = "Bạn vui lòng cho mình mã đơn hàng (dạng DHxxxx) hoặc số điện thoại đặt hàng để mình kiểm tra nhé."

ORDER_RE = re.compile(r"\bD\s?H\s?-?(\d{4})\b", re.I)
PHONE_RE = re.compile(r"\b0\d{9}\b")
PRODUCT_RE = re.compile(r"\bP\d{3}\b", re.I)


def strip_accents(s: str) -> str:
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def extract_slots(text: str) -> dict:
    out = {}
    if m := ORDER_RE.search(text):
        out["order_id"] = f"DH{m.group(1)}"
    if m := PHONE_RE.search(text):
        out["phone"] = m.group(0)
    if m := PRODUCT_RE.search(text):
        out["product_id"] = m.group(0).upper()
    return out


@dataclass
class ConversationState:
    entities: dict = field(default_factory=lambda: {"order_id": "", "phone": "", "product": ""})
    pending_intent: str = ""
    pending_missing: list[str] = field(default_factory=list)
    last_agents: list[str] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    turn: int = 0

    def merge(self, slots: dict) -> None:
        for k, v in slots.items():
            if not v:
                continue
            key = "product" if k in ("product", "product_id") else k
            if key in self.entities:
                self.entities[key] = v

    def router_view(self, max_turns: int = 2) -> str:
        view = {
            "entities": {k: v for k, v in self.entities.items() if v},
            "pending_intent": self.pending_intent,
            "pending_missing": self.pending_missing,
            "last_agents": self.last_agents,
            "recent_turns": [{"user": h["user"][:200], "assistant": h["assistant"][:200]}
                             for h in self.history[-max_turns:]],
        }
        return json.dumps(view, ensure_ascii=False)

    def add_turn(self, user: str, assistant: str) -> None:
        self.history.append({"user": user, "assistant": assistant})
        self.turn += 1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_view(cls, view: str | dict) -> "ConversationState":
        """Dựng lại state từ router_view() (dùng cho routing eval trên tập test đã sinh sẵn)."""
        v = json.loads(view) if isinstance(view, str) else view
        st = cls(pending_intent=v.get("pending_intent", ""), pending_missing=list(v.get("pending_missing", [])),
                 last_agents=list(v.get("last_agents", [])), history=list(v.get("recent_turns", [])))
        st.merge(v.get("entities", {}))
        st.turn = len(st.history)
        return st
