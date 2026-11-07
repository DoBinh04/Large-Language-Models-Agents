"""Test dispatcher (không spawn process, không cần mạng):  python -m tests.test_server"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp_server import INVALID_PARAMS, METHOD_NOT_FOUND, MCPServer, Tools  # noqa: E402

root = Path(tempfile.mkdtemp())
(root / "a.txt").write_text("hello", encoding="utf-8")
s = MCPServer(Tools(root))


def rpc(method, params=None, id_=1):
    msg = {"jsonrpc": "2.0", "method": method, "id": id_}
    if params is not None:
        msg["params"] = params
    return s.handle(msg)


def test_lifecycle():
    r = rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t"}})
    assert r["result"]["protocolVersion"] == "2025-06-18"          # echo version client đề nghị
    r = rpc("initialize", {"protocolVersion": "1999-01-01"})
    assert r["result"]["protocolVersion"] == "2025-11-25"          # không hỗ trợ -> version mới nhất
    assert s.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None   # notification: không trả lời
    assert rpc("ping")["result"] == {}


def test_tools():
    names = [t["name"] for t in rpc("tools/list")["result"]["tools"]]
    assert names == ["read_file", "list_directory", "execute_python"]
    r = rpc("tools/call", {"name": "read_file", "arguments": {"path": "a.txt"}})["result"]
    assert r == {"content": [{"type": "text", "text": "hello"}], "isError": False}
    r = rpc("tools/call", {"name": "list_directory", "arguments": {}})["result"]
    assert "a.txt" in r["content"][0]["text"]
    r = rpc("tools/call", {"name": "execute_python", "arguments": {"code": "print(6*7)"}})["result"]
    assert "42" in r["content"][0]["text"] and not r["isError"]


def test_errors():
    r = rpc("tools/call", {"name": "read_file", "arguments": {"path": "../outside.txt"}})["result"]
    assert r["isError"] and "outside" in r["content"][0]["text"]               # lỗi tool -> isError
    assert rpc("tools/call", {"name": "execute_python", "arguments": {"code": "raise SystemExit(3)"}})["result"]["isError"]
    assert rpc("tools/call", {"name": "nope", "arguments": {}})["error"]["code"] == INVALID_PARAMS
    assert rpc("tools/call", {"name": "read_file", "arguments": {}})["error"]["code"] == INVALID_PARAMS
    assert rpc("foo/bar")["error"]["code"] == METHOD_NOT_FOUND
    batch = s.handle([{"jsonrpc": "2.0", "id": 7, "method": "ping"},
                      {"jsonrpc": "2.0", "method": "notifications/initialized"}])
    assert [m["id"] for m in batch] == [7]
    json.dumps(batch)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
