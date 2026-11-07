"""A2A (Agent2Agent) tối giản nhưng đúng hình dạng giao thức.

- AgentCard  : agent tự mô tả (name, description, url, skills). Phục vụ ở
               GET /.well-known/agent-card.json khi chạy HTTP.
- JSON-RPC 2.0, method "message/send": params.message = Message{role, parts[], contextId, messageId}
               -> result = Task{id, contextId, status{state, message}, artifacts[]}.
- Task state : submitted -> working -> completed | failed | input-required.
- Part       : {"kind":"text","text":...} hoặc {"kind":"data","data":{...}}.

Transport:
- InProcessTransport: vẫn serialize request/response ra JSON rồi parse lại, nên agent không thể
  "lách" bằng object Python; đổi sang HTTP không phải sửa agent.
- HttpTransport + serve_agent(): mỗi agent một HTTP server (stdlib), Orchestrator gọi qua urllib.
"""
from __future__ import annotations

import json
import threading
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Protocol

from support.tracing import NULL_TRACER, Tracer


def _id() -> str:
    return uuid.uuid4().hex[:12]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class AgentSkill:
    id: str
    name: str
    description: str
    examples: list[str] = field(default_factory=list)


@dataclass
class AgentCard:
    name: str
    description: str
    skills: list[AgentSkill]
    url: str = ""
    version: str = "1.0.0"
    protocolVersion: str = "0.3.0"
    defaultInputModes: list[str] = field(default_factory=lambda: ["text/plain", "application/json"])
    defaultOutputModes: list[str] = field(default_factory=lambda: ["text/plain", "application/json"])

    def to_dict(self) -> dict:
        return asdict(self)


def text_part(text: str) -> dict:
    return {"kind": "text", "text": text}


def data_part(data: Any) -> dict:
    return {"kind": "data", "data": data}


def make_message(text: str, data: dict | None = None, role: str = "user", context_id: str | None = None) -> dict:
    parts = [text_part(text)] + ([data_part(data)] if data is not None else [])
    return {"kind": "message", "messageId": _id(), "role": role, "parts": parts,
            "contextId": context_id or _id(), "metadata": {"timestamp": _now()}}


def make_task(message: dict, state: str, reply_text: str = "", artifacts: list[dict] | None = None) -> dict:
    return {"kind": "task", "id": _id(), "contextId": message.get("contextId"),
            "status": {"state": state, "timestamp": _now(),
                       "message": make_message(reply_text, role="agent", context_id=message.get("contextId"))},
            "artifacts": artifacts or [], "history": [message]}


def artifact(name: str, data: dict, text: str = "") -> dict:
    parts = ([text_part(text)] if text else []) + [data_part(data)]
    return {"artifactId": _id(), "name": name, "parts": parts}


def get_text(msg: dict) -> str:
    return "\n".join(p["text"] for p in msg.get("parts", []) if p.get("kind") == "text")


def get_data(msg: dict) -> dict:
    out: dict = {}
    for p in msg.get("parts", []):
        if p.get("kind") == "data" and isinstance(p.get("data"), dict):
            out.update(p["data"])
    return out


def rpc_request(message: dict) -> dict:
    return {"jsonrpc": "2.0", "id": _id(), "method": "message/send", "params": {"message": message}}


def rpc_dispatch(req: dict, handler: Callable[[dict], dict]) -> dict:
    if req.get("method") != "message/send":
        return {"jsonrpc": "2.0", "id": req.get("id"), "error": {"code": -32601, "message": "Method not found"}}
    try:
        task = handler(req["params"]["message"])
        return {"jsonrpc": "2.0", "id": req.get("id"), "result": task}
    except Exception as e:  # lỗi agent -> Task failed, không làm sập orchestrator
        msg = req["params"]["message"]
        return {"jsonrpc": "2.0", "id": req.get("id"),
                "result": make_task(msg, "failed", f"{type(e).__name__}: {e}")}


# --------------------------------------------------------------------------- #
# Transports
# --------------------------------------------------------------------------- #
class A2AServer(Protocol):
    card: AgentCard

    def handle(self, message: dict) -> dict: ...


class InProcessTransport:
    def __init__(self, agents: dict[str, A2AServer]):
        self.agents = agents

    def cards(self) -> list[dict]:
        return [a.card.to_dict() for a in self.agents.values()]

    def send(self, agent: str, message: dict) -> dict:
        wire = json.dumps(rpc_request(message), ensure_ascii=False)           # đi qua "dây" JSON
        resp = rpc_dispatch(json.loads(wire), self.agents[agent].handle)
        return json.loads(json.dumps(resp, ensure_ascii=False))


class HttpTransport:
    def __init__(self, urls: dict[str, str]):
        self.urls = urls        # name -> http://host:port

    def cards(self) -> list[dict]:
        out = []
        for u in self.urls.values():
            with urllib.request.urlopen(f"{u}/.well-known/agent-card.json", timeout=10) as r:
                out.append(json.loads(r.read()))
        return out

    def send(self, agent: str, message: dict) -> dict:
        body = json.dumps(rpc_request(message), ensure_ascii=False).encode()
        req = urllib.request.Request(self.urls[agent], data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())


def serve_agent(agent: A2AServer, port: int, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    agent.card.url = f"http://{host}:{port}"

    class H(BaseHTTPRequestHandler):
        def _send(self, code: int, obj: dict):
            data = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/.well-known/agent-card.json":
                self._send(200, agent.card.to_dict())
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            self._send(200, rpc_dispatch(req, agent.handle))

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer((host, port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


# --------------------------------------------------------------------------- #
# Client có tracing
# --------------------------------------------------------------------------- #
class A2AClient:
    def __init__(self, transport, tracer: Tracer = NULL_TRACER):
        self.transport = transport
        self.tracer = tracer

    def send(self, agent: str, text: str, data: dict | None = None, context_id: str | None = None) -> dict:
        msg = make_message(text, data, context_id=context_id)
        with self.tracer.span(f"a2a.send.{agent}", kind="a2a", to=agent, request=msg) as sp:
            resp = self.transport.send(agent, msg)
            if "error" in resp:
                sp.status = "error"
                raise RuntimeError(f"A2A error từ {agent}: {resp['error']}")
            task = resp["result"]
            sp.set(task_state=task["status"]["state"], task_id=task["id"])
            return task
