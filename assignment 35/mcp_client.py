"""MCP client tối giản qua stdio (tự viết, không dùng SDK).

Client spawn server như một subprocess, rồi nói chuyện với nó bằng JSON-RPC,
mỗi message một dòng trên stdin/stdout của subprocess.
"""
from __future__ import annotations

import itertools
import json
import subprocess
import sys
from pathlib import Path

PROTOCOL_VERSION = "2025-11-25"
CLIENT_INFO = {"name": "phase3-custom-client", "version": "1.0.0"}


def setup_console() -> None:
    """Windows: khi redirect (`> file.txt`) stdout mặc định là cp1252 -> không in được tiếng Việt / '≤'.
    Ép UTF-8 cho cả console lẫn file."""
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")


def _color(code: str, text: str) -> str:
    # Chỉ tô màu khi in ra terminal; ghi ra file thì bỏ mã ANSI cho dễ đọc
    return f"\033[{code}m{text}\033[0m" if sys.stdout.isatty() else text


class MCPError(Exception):
    def __init__(self, error: dict):
        super().__init__(f"[{error.get('code')}] {error.get('message')}")
        self.error = error


class StdioMCPClient:
    def __init__(self, command: list[str], show_wire: bool = False, cwd: str | Path | None = None):
        self.proc = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,   # stderr của server -> console
            text=True, encoding="utf-8", bufsize=1, cwd=cwd,
        )
        self.show_wire = show_wire
        self._ids = itertools.count(1)
        self.server_info: dict = {}
        self.server_capabilities: dict = {}
        self.protocol_version: str | None = None

    # ---- tầng wire ---------------------------------------------------------------
    def _write(self, msg: dict) -> None:
        line = json.dumps(msg, ensure_ascii=False)
        if self.show_wire:
            print(_color("36", f"--> {line}"))
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _read(self) -> dict:
        line = self.proc.stdout.readline()
        if not line:
            raise ConnectionError("server đóng stdout (đã thoát?)")
        if self.show_wire:
            print(_color("33", f"<-- {line.rstrip()}"))
        return json.loads(line)

    def request(self, method: str, params: dict | None = None) -> dict:
        req_id = next(self._ids)
        msg = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            msg["params"] = params
        self._write(msg)
        while True:
            resp = self._read()
            if resp.get("id") != req_id:          # notification / request từ server: client này bỏ qua
                continue
            if "error" in resp:
                raise MCPError(resp["error"])
            return resp["result"]

    def notify(self, method: str, params: dict | None = None) -> None:
        msg = {"jsonrpc": "2.0", "method": method}        # không có "id" -> notification
        if params is not None:
            msg["params"] = params
        self._write(msg)

    def send_raw(self, line: str) -> dict:
        """Gửi một dòng tuỳ ý (để thử lỗi parse...) và đọc 1 response."""
        if self.show_wire:
            print(_color("36", f"--> {line}"))
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        return self._read()

    # ---- tầng MCP ----------------------------------------------------------------
    def initialize(self) -> dict:
        result = self.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},                 # client này không cung cấp sampling/roots/elicitation
            "clientInfo": CLIENT_INFO,
        })
        self.protocol_version = result["protocolVersion"]
        self.server_info = result.get("serverInfo", {})
        self.server_capabilities = result.get("capabilities", {})
        self.notify("notifications/initialized")
        return result

    def list_tools(self) -> list[dict]:
        tools, cursor = [], None
        while True:                              # tools/list có phân trang bằng nextCursor
            result = self.request("tools/list", {"cursor": cursor} if cursor else {})
            tools += result["tools"]
            cursor = result.get("nextCursor")
            if not cursor:
                return tools

    def call_tool(self, name: str, arguments: dict) -> dict:
        return self.request("tools/call", {"name": name, "arguments": arguments})

    def close(self) -> None:
        # Spec stdio: client đóng stdin của server, đợi, rồi mới kill nếu cần
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def default_server_command(root: str | None = None, extra: list[str] | None = None) -> list[str]:
    server = Path(__file__).resolve().parent / "mcp_server.py"
    cmd = [sys.executable, str(server)]
    if root:
        cmd += ["--root", root]
    return cmd + (extra or [])


def result_text(result: dict) -> str:
    return "\n".join(c.get("text", f"<{c.get('type')} content>") for c in result.get("content", []))
