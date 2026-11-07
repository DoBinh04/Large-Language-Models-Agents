"""MCP Server: expose toàn bộ business tools của cửa hàng qua Model Context Protocol.

Chạy:
    python -m support.mcp_server                 # stdio (agent / Claude Desktop spawn process này)
    python -m support.mcp_server --http 8765     # Streamable HTTP tại http://127.0.0.1:8765/mcp

Dùng FastMCP của SDK chính thức (`pip install mcp`). Schema của tool được FastMCP sinh từ type hint
của hàm trong support/tools.py, nên registry chỉ có một nguồn sự thật.
Ngoài tools còn expose 2 resource: catalog sản phẩm và danh sách tài liệu chính sách.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from support import tools  # noqa: E402


def build_server(port: int = 8765):
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("techviet-support", port=port)
    for name, spec in tools.TOOLS.items():
        mcp.add_tool(spec["fn"], name=name, description=f"[{spec['agent']}] {spec['description']}")

    @mcp.resource("catalog://products")
    def catalog() -> str:
        return json.dumps([tools._brief(p) for p in tools.PRODUCTS.values()], ensure_ascii=False)

    @mcp.resource("docs://policies")
    def policies() -> str:
        return "\n".join(p.name for p in sorted(config.DOCS_DIR.glob("*.md")))

    return mcp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", type=int, default=None, help="chạy Streamable HTTP ở port này thay vì stdio")
    a = ap.parse_args()
    from support.logging_setup import setup_logging

    setup_logging(default_level="WARNING", force=True)          # log ra stderr, không đụng stdout của stdio
    server = build_server(a.http or 8765)
    server.run(transport="streamable-http" if a.http else "stdio")
