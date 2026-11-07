"""Mô hình dữ liệu kiểu A2A (Agent2Agent protocol), bản rút gọn.

Các khái niệm chính lấy từ A2A:
- AgentCard : "danh thiếp" mỗi agent công bố: tên, mô tả, danh sách skill. Orchestrator
              chọn agent theo skill, không gọi thẳng class hay hàm của agent.
- Message   : một lượt giao tiếp, gồm nhiều Part (TextPart / DataPart), có role user|agent.
- Task      : đơn vị công việc có vòng đời submitted -> working -> completed | failed | input-required,
              kết quả nằm trong artifacts[].
- Envelope  : JSON-RPC 2.0, method "message/send" (request) và result = Task (response).

Ở đây "transport" là state của LangGraph (mailbox) thay vì HTTP. Nhưng mọi thứ đi qua mailbox
đều là JSON thuần, nên muốn tách agent ra thành service HTTP riêng chỉ cần thay tầng gửi/nhận.
Mọi object là dict để LangGraph checkpoint/serialize được.
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

TASK_STATES = ("submitted", "working", "input-required", "completed", "failed", "canceled")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _id() -> str:
    return uuid.uuid4().hex[:12]


# --------------------------------------------------------------------------- #
# Agent Card
# --------------------------------------------------------------------------- #
@dataclass
class AgentSkill:
    id: str
    name: str
    description: str
    input_parts: list[str] = field(default_factory=lambda: ["text", "data"])
    output_parts: list[str] = field(default_factory=lambda: ["data"])


@dataclass
class AgentCard:
    name: str
    description: str
    skills: list[AgentSkill]
    version: str = "1.0.0"
    url: str = ""            # A2A thật: endpoint HTTP của agent; ở đây là tên node trong graph

    def to_dict(self) -> dict:
        return asdict(self)

    def has_skill(self, skill_id: str) -> bool:
        return any(s.id == skill_id for s in self.skills)


# --------------------------------------------------------------------------- #
# Parts / Message / Task
# --------------------------------------------------------------------------- #
def text_part(text: str) -> dict:
    return {"kind": "text", "text": text}


def data_part(data: Any) -> dict:
    return {"kind": "data", "data": data}


def make_message(role: str, sender: str, receiver: str, parts: list[dict], context_id: str,
                 task_id: str | None = None, skill: str | None = None) -> dict:
    return {
        "kind": "message", "messageId": _id(), "role": role, "parts": parts,
        "contextId": context_id, "taskId": task_id,
        "metadata": {"from": sender, "to": receiver, "skill": skill, "timestamp": _now()},
    }


def make_task(request: dict, state: str = "submitted") -> dict:
    return {
        "kind": "task", "id": request.get("taskId") or _id(), "contextId": request["contextId"],
        "status": {"state": state, "timestamp": _now()},
        "artifacts": [], "history": [request],
    }


def set_status(task: dict, state: str, message: dict | None = None) -> dict:
    assert state in TASK_STATES, state
    task["status"] = {"state": state, "timestamp": _now(), **({"message": message} if message else {})}
    if message:
        task["history"].append(message)
    return task


def add_artifact(task: dict, name: str, parts: list[dict]) -> dict:
    task["artifacts"].append({"artifactId": _id(), "name": name, "parts": parts})
    return task


# --------------------------------------------------------------------------- #
# JSON-RPC envelopes (A2A: method "message/send")
# --------------------------------------------------------------------------- #
def send_request(message: dict) -> dict:
    return {"jsonrpc": "2.0", "id": _id(), "method": "message/send", "params": {"message": message}}


def send_response(request_envelope: dict, task: dict) -> dict:
    return {"jsonrpc": "2.0", "id": request_envelope["id"], "result": task}


# --------------------------------------------------------------------------- #
# Helpers đọc nội dung
# --------------------------------------------------------------------------- #
def get_text(message: dict) -> str:
    return "\n".join(p["text"] for p in message["parts"] if p["kind"] == "text")


def get_data(message: dict) -> dict:
    out: dict = {}
    for p in message["parts"]:
        if p["kind"] == "data" and isinstance(p["data"], dict):
            out.update(p["data"])
    return out


def artifact_data(task: dict, name: str) -> Any:
    for a in task.get("artifacts", []):
        if a["name"] == name:
            for p in a["parts"]:
                if p["kind"] == "data":
                    return p["data"]
                if p["kind"] == "text":
                    return p["text"]
    return None


def summarize_envelope(env: dict) -> str:
    """Một dòng mô tả envelope để in transcript."""
    if "method" in env:
        m = env["params"]["message"]
        md = m["metadata"]
        return f"{md['from']} → {md['to']}  [{env['method']} skill={md['skill']}] {get_text(m)[:90]!r}"
    t = env["result"]
    st = t["status"]
    sender = st.get("message", {}).get("metadata", {}).get("from", "?")
    arts = ", ".join(a["name"] for a in t["artifacts"]) or "-"
    return f"{sender} → orchestrator  [task {t['id']} {st['state']}] artifacts: {arts}"
