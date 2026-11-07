"""Demo wire protocol: in ra TỪNG message JSON-RPC đi/về giữa client và server.

    python mcp_client_demo.py
    python mcp_client_demo.py --root "C:\\phase3\\assignment 32"

Màu xanh (-->) = client gửi, màu vàng (<--) = server trả.
Không cần API key: không có LLM ở đây, chỉ có protocol.
"""
from __future__ import annotations

import argparse
import json

from mcp_client import MCPError, StdioMCPClient, _color, default_server_command, result_text, setup_console


def section(title: str) -> None:
    print("\n" + _color("1", f"=== {title} ==="))


def main():
    setup_console()
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=None)
    args = p.parse_args()

    with StdioMCPClient(default_server_command(args.root), show_wire=True) as c:
        section("1. Handshake: initialize -> result -> notifications/initialized")
        info = c.initialize()
        print(f"server = {info['serverInfo']}, protocol = {info['protocolVersion']}")

        section("2. ping")
        c.request("ping")

        section("3. tools/list")
        for t in c.list_tools():
            print(f"- {t['name']}: {json.dumps(t['inputSchema'])}")

        section("4. tools/call list_directory")
        print(result_text(c.call_tool("list_directory", {"path": "."})))

        section("5. tools/call execute_python (tạo file rồi đọc lại bằng read_file)")
        code = "import platform, math\nopen('hello.txt','w').write('xin chao MCP')\nprint(platform.python_version(), math.factorial(20))"
        print(result_text(c.call_tool("execute_python", {"code": code})))
        print(result_text(c.call_tool("read_file", {"path": "hello.txt"})))

        section("6. Lỗi của TOOL -> result.isError = true (LLM đọc được và tự sửa)")
        r = c.call_tool("read_file", {"path": "../../Windows/win.ini"})
        print(f"isError={r['isError']}: {result_text(r)}")
        r = c.call_tool("execute_python", {"code": "1/0"})
        print(f"isError={r['isError']}")

        section("7. Lỗi của PROTOCOL -> JSON-RPC error object")
        for method, params in [("tools/call", {"name": "delete_everything", "arguments": {}}),
                               ("tools/call", {"name": "read_file", "arguments": {}}),
                               ("resources/list", {})]:
            try:
                c.request(method, params)
            except MCPError as e:
                print(f"{method}: {e}")
        c.send_raw("{this is not json")
    print("\nĐã đóng stdin -> server thoát.")


if __name__ == "__main__":
    main()
