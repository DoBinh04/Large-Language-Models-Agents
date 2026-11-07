"""Hai cách agent gọi tool, cùng interface `call(name, args) -> dict`:

- LocalTools : gọi thẳng hàm Python (nhanh, dùng mặc định khi eval).
- MCPTools   : spawn `python -m support.mcp_server` và gọi qua MCP (stdio) bằng SDK `mcp`.
               SDK là async nên chạy một event loop riêng trong thread nền; agent vẫn gọi đồng bộ.

Chọn bằng `--tools local|mcp` ở run_chat.py / eval/run_eval.py. So latency hai chế độ trong trace
(span kind="tool", attr backend) để thấy overhead của MCP.
"""
from __future__ import annotations

import asyncio
import json
import sys
import threading

import config
from support import tools


class LocalTools:
    backend = "local"

    def call(self, name: str, args: dict) -> dict:
        return tools.call(name, args)

    def schemas(self, names: list[str]) -> list[dict]:
        return tools.anthropic_schema(names)

    def reset(self) -> None:
        tools.reset()

    def close(self) -> None:
        pass


class MCPTools:
    backend = "mcp"

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        self._ready = threading.Event()
        self._stop: asyncio.Event | None = None
        self.session = None
        self._tool_schemas: dict[str, dict] = {}
        asyncio.run_coroutine_threadsafe(self._main(), self.loop)
        if not self._ready.wait(timeout=60):
            raise RuntimeError("Không khởi động được MCP server")

    async def _main(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(command=sys.executable, args=["-m", "support.mcp_server"],
                                       cwd=str(config.ROOT))
        self._stop = asyncio.Event()
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_tools()
                self._tool_schemas = {t.name: {"name": t.name, "description": t.description or "",
                                               "input_schema": t.inputSchema} for t in listed.tools}
                self.session = session
                self._ready.set()
                await self._stop.wait()

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout=120)

    def call(self, name: str, args: dict) -> dict:
        res = self._run(self.session.call_tool(name, args))
        if getattr(res, "structuredContent", None):
            data = res.structuredContent
            return data.get("result", data) if isinstance(data, dict) else data
        text = "".join(getattr(c, "text", "") for c in res.content)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"error": text} if res.isError else {"text": text}

    def schemas(self, names: list[str]) -> list[dict]:
        # Dùng schema của registry gốc (có enum, mô tả đầy đủ); MCP chỉ là transport.
        return tools.anthropic_schema(names)

    def reset(self) -> None:
        # Server MCP là process riêng: state ghi (phiếu, huỷ đơn) nằm bên đó. Reset bằng cách gọi
        # tools.reset() phía client không có tác dụng; với eval nhiều kịch bản nên dùng LocalTools.
        pass

    def close(self) -> None:
        if self._stop:
            self.loop.call_soon_threadsafe(self._stop.set)


def make_backend(kind: str = "local"):
    return MCPTools() if kind == "mcp" else LocalTools()
