"""Tool cho agent LangGraph.

- 3 tool "an toàn" dùng lại nguyên hàm của assignment 3.3 (search_web, calculate, get_current_time)
- 2 tool "nguy hiểm" (có side effect, không hoàn tác được): write_file, send_email
  -> graph sẽ interrupt và chờ người duyệt trước khi chạy.

Ở bài này agent dùng NATIVE tool use của Claude (JSON schema) thay vì parse text ReAct:
LangGraph lo phần vòng lặp/điều phối, còn model trả về tool_use block có cấu trúc.
"""
from __future__ import annotations

import json
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

import _bootstrap  # noqa: F401  (thêm "assignment 33" vào sys.path)
from react_agent.tools import calculate, get_current_time, search_web

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT / "workspace"      # write_file chỉ được ghi trong thư mục này
OUTBOX = ROOT / "outbox"            # send_email: mô phỏng, lưu .eml vào đây

DANGEROUS_TOOLS = {"write_file", "send_email"}


def _safe_path(rel: str) -> Path:
    p = (WORKSPACE / rel).resolve()
    if not p.is_relative_to(WORKSPACE.resolve()):
        raise PermissionError(f"đường dẫn '{rel}' nằm ngoài workspace")
    return p


def write_file(path: str, content: str, mode: str = "overwrite") -> str:
    p = _safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a" if mode == "append" else "w", encoding="utf-8") as f:
        f.write(content)
    return f"Đã ghi {len(content)} ký tự vào workspace/{p.relative_to(WORKSPACE.resolve()).as_posix()} ({mode})"


def send_email(to: str, subject: str, body: str) -> str:
    """MÔ PHỎNG gửi email: tạo file .eml trong outbox/ (không gửi thật ra ngoài)."""
    msg = EmailMessage()
    msg["From"] = "agent@phase3.local"
    msg["To"] = to
    msg["Subject"] = subject
    msg["Date"] = datetime.now().astimezone().strftime("%a, %d %b %Y %H:%M:%S %z")
    msg.set_content(body)
    OUTBOX.mkdir(exist_ok=True)
    f = OUTBOX / f"{datetime.now():%Y%m%d-%H%M%S-%f}.eml"
    f.write_bytes(bytes(msg))
    return f"Đã gửi (mô phỏng) email tới {to}, lưu tại outbox/{f.name}"


FUNCTIONS = {
    "search_web": search_web,
    "calculate": calculate,
    "get_current_time": get_current_time,
    "write_file": write_file,
    "send_email": send_email,
}

TOOL_SCHEMAS = [
    {
        "name": "search_web",
        "description": "Search the web. Use for facts and numbers that may change over time.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Short search query, preferably in English"},
                "max_results": {"type": "integer", "description": "Number of results (default 5)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculate",
        "description": "Evaluate an arithmetic expression. Always use this instead of mental math. "
                       "Supports + - * / // % **, sqrt, log, sin, cos, pi, e.",
        "input_schema": {
            "type": "object",
            "properties": {"expr": {"type": "string", "description": "e.g. '(1.065**5) * 10000000'"}},
            "required": ["expr"],
        },
    },
    {
        "name": "get_current_time",
        "description": "Get the current date and time (with weekday and day of year) in an IANA timezone.",
        "input_schema": {
            "type": "object",
            "properties": {"timezone": {"type": "string", "description": "e.g. 'Asia/Ho_Chi_Minh' (default)"}},
        },
    },
    {
        "name": "write_file",
        "description": "Write text to a file inside the workspace folder. DANGEROUS: requires human approval.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path inside the workspace, e.g. 'notes/report.md'"},
                "content": {"type": "string"},
                "mode": {"type": "string", "enum": ["overwrite", "append"]},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "send_email",
        "description": "Send an email. DANGEROUS: requires human approval.",
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["to", "subject", "body"],
        },
    },
]


def run_tool(name: str, args: dict) -> tuple[str, bool]:
    """Trả về (output, is_error). Không raise: lỗi được gửi lại cho model dưới dạng tool_result lỗi."""
    fn = FUNCTIONS.get(name)
    if fn is None:
        return f"Unknown tool '{name}'", True
    try:
        out = str(fn(**args))
    except Exception as e:
        return f"{type(e).__name__}: {e}", True
    # Tool của 3.3 báo lỗi bằng chuỗi "ERROR: ..." thay vì raise
    return out, out.startswith("ERROR")


def describe_call(call: dict) -> str:
    return f"{call['name']}({json.dumps(call['input'], ensure_ascii=False)[:400]})"
