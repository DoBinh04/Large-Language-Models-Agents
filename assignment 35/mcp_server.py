"""MCP server tự viết (KHÔNG dùng SDK `mcp`) để thấy rõ wire protocol.

Transport:
    stdio (mặc định)      : mỗi message JSON-RPC là MỘT dòng trên stdin/stdout (UTF-8, kết thúc '\n').
                            stdout CHỈ dành cho protocol -> mọi log phải ra stderr.
    Streamable HTTP       : python mcp_server.py --http 8765  -> POST http://127.0.0.1:8765/mcp

Tools: read_file, list_directory, execute_python (đều bị giới hạn trong --root).

Vòng đời một phiên:
    client → initialize                  (thoả thuận protocolVersion + capabilities)
    server ← result
    client → notifications/initialized   (notification: không có "id", không cần trả lời)
    client → tools/list   → server trả danh sách tool + JSON Schema
    client → tools/call   → server chạy tool, trả về content[]
"""
from __future__ import annotations

import argparse
import codecs
import json
import os
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SERVER_INFO = {"name": "phase3-fs-python", "version": "1.0.0"}
SUPPORTED_VERSIONS = ["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"]   # mới nhất trước

# Mã lỗi chuẩn JSON-RPC 2.0
PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS, INTERNAL_ERROR = -32700, -32600, -32601, -32602, -32603

MAX_READ_BYTES = 200_000
MAX_OUTPUT_CHARS = 20_000


def log(*args) -> None:
    print(f"[{datetime.now():%H:%M:%S}] [mcp-server]", *args, file=sys.stderr, flush=True)


class ToolError(Exception):
    """Lỗi khi chạy tool -> trả về result với isError=true (KHÔNG phải JSON-RPC error)."""


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
class Tools:
    def __init__(self, root: Path, python_timeout: float = 10.0, allow_exec: bool = True):
        self.root = root.resolve()
        self.python_timeout = python_timeout
        self.allow_exec = allow_exec

    def _resolve(self, path: str) -> Path:
        p = (self.root / (path or ".")).resolve()
        if not p.is_relative_to(self.root):
            raise ToolError(f"Access denied: '{path}' is outside the server root")
        return p

    # ---- read_file -----------------------------------------------------------
    def read_file(self, path: str, max_bytes: int = MAX_READ_BYTES) -> str:
        p = self._resolve(path)
        if not p.is_file():
            raise ToolError(f"Not a file: {path}")
        data = p.read_bytes()[: max(1, min(int(max_bytes), MAX_READ_BYTES))]
        truncated = p.stat().st_size > len(data)
        try:
            # final=False: nếu cắt đúng giữa 1 ký tự nhiều byte (tiếng Việt...) thì bỏ phần dở dang ở cuối
            # thay vì báo nhầm "không phải UTF-8"
            text = codecs.getincrementaldecoder("utf-8")().decode(data, final=not truncated)
        except UnicodeDecodeError:
            raise ToolError(f"{path} is not a UTF-8 text file ({p.stat().st_size} bytes)")
        if truncated:
            text += f"\n...[truncated, file has {p.stat().st_size} bytes]"
        return text

    # ---- list_directory ------------------------------------------------------
    def list_directory(self, path: str = ".") -> str:
        p = self._resolve(path)
        if not p.is_dir():
            raise ToolError(f"Not a directory: {path}")
        lines = []
        for child in sorted(p.iterdir(), key=lambda c: (not c.is_dir(), c.name.lower())):
            if child.is_dir():
                lines.append(f"[DIR]  {child.name}/")
            else:
                lines.append(f"[FILE] {child.name}  ({child.stat().st_size} bytes)")
        rel = p.relative_to(self.root).as_posix() or "."
        return f"Contents of {rel} ({len(lines)} entries):\n" + ("\n".join(lines) or "(empty)")

    # ---- execute_python ------------------------------------------------------
    def execute_python(self, code: str, timeout: float | None = None) -> str:
        """Chạy code trong một PROCESS RIÊNG (python -I), cwd = root, có timeout.
        LƯU Ý: đây KHÔNG phải sandbox bảo mật. Code vẫn có quyền của user đang chạy server.
        Chạy thật thì nên đặt trong container / VM / user hạn chế quyền."""
        if not self.allow_exec:
            raise ToolError("execute_python is disabled on this server (--no-exec)")
        t = min(float(timeout or self.python_timeout), 60.0)
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(code)
            script = f.name
        # Không truyền các biến môi trường bí mật (API key...) cho code do LLM viết
        env = {k: v for k, v in os.environ.items()
               if not any(s in k.upper() for s in ("KEY", "TOKEN", "SECRET", "PASSWORD"))}
        env["PYTHONIOENCODING"] = "utf-8"
        try:
            proc = subprocess.run([sys.executable, "-I", script], cwd=self.root, capture_output=True,
                                  text=True, encoding="utf-8", errors="replace", timeout=t, env=env)
        except subprocess.TimeoutExpired:
            raise ToolError(f"Execution timed out after {t}s")
        finally:
            os.unlink(script)
        out = f"exit_code: {proc.returncode}\n--- stdout ---\n{proc.stdout}"
        if proc.stderr:
            out += f"\n--- stderr ---\n{proc.stderr}"
        if len(out) > MAX_OUTPUT_CHARS:
            out = out[:MAX_OUTPUT_CHARS] + "\n...[truncated]"
        if proc.returncode != 0:
            raise ToolError(out)
        return out

    # ---- định nghĩa cho tools/list -------------------------------------------
    def definitions(self) -> list[dict]:
        defs = [
            {
                "name": "read_file",
                "title": "Read file",
                "description": "Read a UTF-8 text file. Paths are relative to the server root.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path relative to the root"},
                        "max_bytes": {"type": "integer", "description": f"Max bytes to read (≤ {MAX_READ_BYTES})"},
                    },
                    "required": ["path"],
                },
                "annotations": {"readOnlyHint": True},
            },
            {
                "name": "list_directory",
                "title": "List directory",
                "description": "List files and sub-directories. Paths are relative to the server root.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"path": {"type": "string", "description": "Directory path, default '.'"}},
                },
                "annotations": {"readOnlyHint": True},
            },
        ]
        if self.allow_exec:
            defs.append({
                "name": "execute_python",
                "title": "Execute Python",
                "description": "Run a Python script in a separate process (cwd = server root) and return "
                               "stdout/stderr. Use print() to show results.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python source code"},
                        "timeout": {"type": "number", "description": "Seconds (default 10, max 60)"},
                    },
                    "required": ["code"],
                },
                "annotations": {"readOnlyHint": False, "destructiveHint": True, "openWorldHint": True},
            })
        return defs

    def call(self, name: str, arguments: dict) -> str:
        fn = {"read_file": self.read_file, "list_directory": self.list_directory,
              "execute_python": self.execute_python if self.allow_exec else None}.get(name)
        if fn is None:
            raise KeyError(name)
        schema = next(d for d in self.definitions() if d["name"] == name)["inputSchema"]
        missing = [k for k in schema.get("required", []) if k not in arguments]
        unknown = [k for k in arguments if k not in schema["properties"]]
        if missing or unknown:
            raise ValueError(f"missing={missing} unknown={unknown}")
        return fn(**arguments)


# --------------------------------------------------------------------------- #
# JSON-RPC dispatcher (dùng chung cho stdio và HTTP)
# --------------------------------------------------------------------------- #
class MCPServer:
    def __init__(self, tools: Tools):
        self.tools = tools
        self.initialized = False
        self.protocol_version: str | None = None

    @staticmethod
    def _result(req_id, result) -> dict:
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    @staticmethod
    def _error(req_id, code: int, message: str, data=None) -> dict:
        err = {"code": code, "message": message}
        if data is not None:
            err["data"] = data
        return {"jsonrpc": "2.0", "id": req_id, "error": err}

    def handle(self, msg) -> dict | list | None:
        """Nhận 1 message (hoặc batch) đã parse. Trả về response, hoặc None nếu là notification/response."""
        if isinstance(msg, list):                          # batch (spec 2025-03-26)
            out = [r for r in (self.handle(m) for m in msg) if r is not None]
            return out or None
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
            return self._error(None, INVALID_REQUEST, "Invalid Request")
        if "method" not in msg:                            # là response cho request của server -> bỏ qua
            return None
        method, params, req_id = msg["method"], msg.get("params") or {}, msg.get("id")
        is_notification = "id" not in msg

        if is_notification:
            if method == "notifications/initialized":
                self.initialized = True
                log("client đã sẵn sàng (notifications/initialized)")
            elif method == "notifications/cancelled":
                log(f"client huỷ request {params.get('requestId')}")
            return None                                    # KHÔNG BAO GIỜ trả lời notification

        try:
            if method == "initialize":
                return self._result(req_id, self._initialize(params))
            if method == "ping":
                return self._result(req_id, {})
            if method == "tools/list":
                return self._result(req_id, {"tools": self.tools.definitions()})
            if method == "tools/call":
                return self._tools_call(req_id, params)
            return self._error(req_id, METHOD_NOT_FOUND, f"Method not found: {method}")
        except Exception as e:  # lỗi ngoài dự kiến của server
            log(f"internal error: {e!r}")
            return self._error(req_id, INTERNAL_ERROR, str(e))

    def _initialize(self, params: dict) -> dict:
        requested = params.get("protocolVersion")
        # Spec: nếu hỗ trợ version client đề nghị -> trả lại đúng version đó; nếu không -> version mới nhất của server
        self.protocol_version = requested if requested in SUPPORTED_VERSIONS else SUPPORTED_VERSIONS[0]
        client = params.get("clientInfo", {})
        log(f"initialize từ {client.get('name')} {client.get('version')}, protocol {requested} -> {self.protocol_version}")
        return {
            "protocolVersion": self.protocol_version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": SERVER_INFO,
            "instructions": f"File tools are sandboxed to {self.tools.root}. Paths are relative to that root.",
        }

    def _tools_call(self, req_id, params: dict) -> dict:
        name, args = params.get("name"), params.get("arguments") or {}
        if not isinstance(args, dict):
            return self._error(req_id, INVALID_PARAMS, "arguments must be an object")
        log(f"tools/call {name} {json.dumps(args, ensure_ascii=False)[:200]}")
        try:
            text = self.tools.call(name, args)
            return self._result(req_id, {"content": [{"type": "text", "text": text}], "isError": False})
        except KeyError:
            return self._error(req_id, INVALID_PARAMS, f"Unknown tool: {name}")
        except (ValueError, TypeError) as e:
            return self._error(req_id, INVALID_PARAMS, f"Invalid arguments for {name}: {e}")
        except (ToolError, OSError) as e:
            # Lỗi nghiệp vụ của tool: trả về như kết quả để LLM đọc được và tự sửa
            return self._result(req_id, {"content": [{"type": "text", "text": str(e)}], "isError": True})


# --------------------------------------------------------------------------- #
# Transport 1: stdio
# --------------------------------------------------------------------------- #
def serve_stdio(server: MCPServer, wire_log: Path | None) -> None:
    stdin = open(sys.stdin.fileno(), "r", encoding="utf-8", newline="\n", closefd=False)
    stdout = open(sys.stdout.fileno(), "w", encoding="utf-8", newline="\n", closefd=False)
    wl = open(wire_log, "a", encoding="utf-8") if wire_log else None

    def trace(direction: str, line: str):
        if wl:
            wl.write(f"{datetime.now().isoformat(timespec='milliseconds')} {direction} {line}\n")
            wl.flush()

    log(f"stdio server sẵn sàng, root = {server.tools.root}")
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        trace("-->", line)
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            resp = MCPServer._error(None, PARSE_ERROR, f"Parse error: {e}")
        else:
            resp = server.handle(msg)
        if resp is not None:
            out = json.dumps(resp, ensure_ascii=False)       # json.dumps không chứa '\n' thật -> 1 dòng
            trace("<--", out)
            stdout.write(out + "\n")
            stdout.flush()
    log("stdin đóng, server thoát")


# --------------------------------------------------------------------------- #
# Transport 2: Streamable HTTP (bản tối giản: luôn trả application/json, không SSE)
# --------------------------------------------------------------------------- #
def serve_http(tools: Tools, host: str, port: int) -> None:
    sessions: dict[str, MCPServer] = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            log("http", fmt % args)

        def _send(self, status: int, body: dict | list | None = None, headers: dict | None = None):
            data = json.dumps(body, ensure_ascii=False).encode() if body is not None else b""
            self.send_response(status)
            if body is not None:
                self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if self.path.rstrip("/") != "/mcp":
                return self._send(404, {"error": "use POST /mcp"})
            # Chống DNS-rebinding: chỉ nhận Origin localhost (spec yêu cầu kiểm tra Origin)
            origin = self.headers.get("Origin")
            if origin and not origin.startswith(("http://localhost", "http://127.0.0.1")):
                return self._send(403, {"error": "forbidden origin"})
            try:
                msg = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"null")
            except json.JSONDecodeError as e:
                return self._send(400, MCPServer._error(None, PARSE_ERROR, str(e)))

            sid = self.headers.get("Mcp-Session-Id")
            is_init = isinstance(msg, dict) and msg.get("method") == "initialize"
            if is_init:
                sid = uuid.uuid4().hex
                sessions[sid] = MCPServer(tools)
            elif sid not in sessions:
                return self._send(404 if sid else 400, MCPServer._error(None, INVALID_REQUEST, "unknown or missing Mcp-Session-Id"))

            resp = sessions[sid].handle(msg)
            if resp is None:                     # chỉ có notification/response -> 202 Accepted, không body
                return self._send(202)
            return self._send(200, resp, {"Mcp-Session-Id": sid} if is_init else None)

        def do_GET(self):
            # Server này không mở SSE stream cho message do server chủ động gửi
            self._send(405, {"error": "SSE stream not supported"}, {"Allow": "POST, DELETE"})

        def do_DELETE(self):
            sessions.pop(self.headers.get("Mcp-Session-Id", ""), None)
            self._send(200, {})

    log(f"HTTP server tại http://{host}:{port}/mcp, root = {tools.root}")
    ThreadingHTTPServer((host, port), Handler).serve_forever()


def main():
    p = argparse.ArgumentParser(description="Minimal MCP server (raw JSON-RPC)")
    p.add_argument("--root", default=str(Path(__file__).resolve().parent / "sandbox"),
                   help="thư mục gốc mà tool được phép truy cập")
    p.add_argument("--http", type=int, metavar="PORT", help="chạy Streamable HTTP thay vì stdio")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--timeout", type=float, default=10.0, help="timeout mặc định của execute_python")
    p.add_argument("--no-exec", action="store_true", help="tắt execute_python")
    p.add_argument("--wire-log", default=None, help="ghi mọi message vào/ra file này (stdio)")
    args = p.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    tools = Tools(root, args.timeout, allow_exec=not args.no_exec)
    if args.http:
        serve_http(tools, args.host, args.http)
    else:
        serve_stdio(MCPServer(tools), Path(args.wire_log) if args.wire_log else None)


if __name__ == "__main__":
    main()
