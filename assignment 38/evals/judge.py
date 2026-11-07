"""Client LLM-as-judge: một lời gọi Claude, output ràng buộc bằng JSON Schema."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

DEFAULT_JUDGE_MODEL = "claude-haiku-5-5"   # cùng model với generator; đổi bằng EVAL_JUDGE_MODEL / --judge-model


@dataclass
class Judge:
    model: str | None = None
    effort: str = "medium"
    max_tokens: int = 8000
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    _client: object = field(default=None, repr=False)

    def __post_init__(self):
        import anthropic

        self._client = anthropic.Anthropic()
        self.model = self.model or os.environ.get("EVAL_JUDGE_MODEL") or DEFAULT_JUDGE_MODEL

    def ask(self, system: str, user: str, schema: dict) -> dict:
        resp = self._client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
        )
        self.calls += 1
        self.input_tokens += resp.usage.input_tokens
        self.output_tokens += resp.usage.output_tokens
        if resp.stop_reason in ("refusal", "max_tokens"):
            raise RuntimeError(f"judge stop_reason={resp.stop_reason}")
        return json.loads("".join(b.text for b in resp.content if b.type == "text"))


def format_chunks(chunks: list[str]) -> str:
    return "\n\n".join(f'<chunk id="{i}">\n{c}\n</chunk>' for i, c in enumerate(chunks, 1))
