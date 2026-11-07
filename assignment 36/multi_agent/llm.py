"""Helper gọi Claude dùng chung cho 3 agent.

- text()       : 1 lượt, trả về text
- structured() : 1 lượt, output ràng buộc bởi JSON Schema (output_config.format)
- tool_loop()  : vòng lặp tool use thủ công, có thể kèm JSON Schema cho câu trả lời cuối
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable

DEFAULT_MODEL = "claude-haiku-5-5"


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, u) -> None:
        self.calls += 1
        self.input_tokens += u.input_tokens
        self.output_tokens += u.output_tokens


@dataclass
class Claude:
    model: str | None = None
    effort: str = "medium"
    max_tokens: int = 16000
    usage: Usage = field(default_factory=Usage)

    def __post_init__(self):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = self.model or os.environ.get("ANTHROPIC_MODEL") or DEFAULT_MODEL

    def _create(self, **kw):
        resp = self.client.messages.create(model=self.model, max_tokens=self.max_tokens,
                                           output_config=kw.pop("output_config", {"effort": self.effort}), **kw)
        self.usage.add(resp.usage)
        if resp.stop_reason == "refusal":
            raise RuntimeError("model refused the request")
        if resp.stop_reason == "max_tokens":
            raise RuntimeError("output bị cắt do max_tokens")
        return resp

    @staticmethod
    def _text(resp) -> str:
        return "".join(b.text for b in resp.content if b.type == "text")

    def text(self, system: str, user: str) -> str:
        return self._text(self._create(system=system, messages=[{"role": "user", "content": user}]))

    def structured(self, system: str, user: str, schema: dict) -> dict:
        resp = self._create(system=system, messages=[{"role": "user", "content": user}],
                            output_config={"effort": self.effort,
                                           "format": {"type": "json_schema", "schema": schema}})
        return json.loads(self._text(resp))

    def tool_loop(self, system: str, user: str, tools: list[dict],
                  execute: Callable[[str, dict], str], schema: dict | None = None,
                  max_turns: int = 8, on_tool: Callable[[str, dict, str], None] | None = None):
        output_config = {"effort": self.effort}
        if schema:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        messages = [{"role": "user", "content": user}]
        for turn in range(max_turns):
            last = turn == max_turns - 1
            resp = self._create(system=system, messages=messages, output_config=output_config,
                                tools=tools, tool_choice={"type": "none"} if last else {"type": "auto"})
            messages.append({"role": "assistant", "content": [b.to_dict() for b in resp.content]})
            if resp.stop_reason != "tool_use":
                text = self._text(resp)
                return json.loads(text) if schema else text
            results = []
            for b in resp.content:
                if b.type == "tool_use":
                    out = execute(b.name, b.input)
                    if on_tool:
                        on_tool(b.name, b.input, out)
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": out})
            messages.append({"role": "user", "content": results})
        raise RuntimeError("tool_loop: hết lượt mà chưa có câu trả lời")
