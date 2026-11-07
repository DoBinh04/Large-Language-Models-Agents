"""Custom agent (Claude) dùng tool từ MCP server qua stdio.

    python mcp_agent.py "Liệt kê file trong thư mục, đọc README rồi đếm số dòng bằng Python"
    python mcp_agent.py --root "C:\\phase3\\assignment 32" "Tài liệu trong data/ nói về gì? Đếm số section."
    python mcp_agent.py --confirm-exec "..."     # hỏi trước mỗi lần execute_python
    python mcp_agent.py --wire "..."             # in cả message JSON-RPC

Agent KHÔNG biết trước có tool gì: nó hỏi server qua tools/list rồi chuyển
inputSchema của MCP thành input_schema của Claude. Đây chính là giá trị của MCP:
một server viết một lần, dùng được cho Claude Desktop, IDE, hay agent tự viết như file này.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from mcp_client import StdioMCPClient, default_server_command, result_text, setup_console

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "assignment 33"))
from react_agent.llm import load_dotenv  # noqa: E402

DEFAULT_MODEL = "claude-opus-5-5"
MAX_TURNS = 15


def mcp_to_claude_tools(mcp_tools: list[dict]) -> list[dict]:
    return [{"name": t["name"], "description": t.get("description", ""), "input_schema": t["inputSchema"]}
            for t in mcp_tools]


def main():
    setup_console()
    p = argparse.ArgumentParser()
    p.add_argument("task")
    p.add_argument("--root", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--confirm-exec", action="store_true")
    p.add_argument("--wire", action="store_true")
    args = p.parse_args()

    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT.parent / "assignment 33" / ".env")
    import anthropic

    client = anthropic.Anthropic()
    model = args.model or os.environ.get("ANTHROPIC_MODEL") or DEFAULT_MODEL

    with StdioMCPClient(default_server_command(args.root), show_wire=args.wire) as mcp:
        init = mcp.initialize()
        tools = mcp_to_claude_tools(mcp.list_tools())
        print(f"Kết nối {init['serverInfo']['name']} (protocol {init['protocolVersion']}), "
              f"tools: {[t['name'] for t in tools]}")
        system = ("You are an assistant with access to a file system and a Python runtime through MCP tools. "
                  + init.get("instructions", "") + " Reply in the same language as the user.")

        messages = [{"role": "user", "content": args.task}]
        for _ in range(MAX_TURNS):
            resp = client.messages.create(model=model, max_tokens=8000, system=system,
                                          tools=tools, messages=messages)
            messages.append({"role": "assistant", "content": [b.to_dict() for b in resp.content]})
            for b in resp.content:
                if b.type == "text" and b.text:
                    print(f"\nClaude: {b.text}")
            if resp.stop_reason != "tool_use":
                break

            results = []
            for b in resp.content:
                if b.type != "tool_use":
                    continue
                print(f"\n→ {b.name}({json.dumps(b.input, ensure_ascii=False)[:300]})")
                if b.name == "execute_python" and args.confirm_exec:
                    print(b.input.get("code", ""))
                    if input("Chạy đoạn code này? [y/N] ").strip().lower() != "y":
                        results.append({"type": "tool_result", "tool_use_id": b.id, "is_error": True,
                                        "content": "User declined to run this code."})
                        continue
                try:
                    r = mcp.call_tool(b.name, b.input)
                    text, is_error = result_text(r), bool(r.get("isError"))
                except Exception as e:      # MCPError (lỗi protocol) hoặc server chết
                    text, is_error = f"MCP error: {e}", True
                print(f"← {'ERROR ' if is_error else ''}{text[:500]}")
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": text, "is_error": is_error})
            messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    main()
