"""Gọi Claude cho ReAct.

Agent KHÔNG dùng native tool use của API: LLM chỉ sinh text theo format ReAct,
còn việc parse Action và chạy tool là do code của mình làm (đúng tinh thần "from scratch").

Stop sequence "Observation:" buộc model dừng ngay sau khi viết xong Action Input,
để Observation luôn là kết quả THẬT của tool chứ không phải do model bịa ra.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "medium"     # low | medium | high | xhigh | max
MAX_OUTPUT_TOKENS = 4096


def load_dotenv(path: str | Path) -> None:
    """Đọc file .env đơn giản (KEY=VALUE) vào os.environ, không cần thư viện ngoài."""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass
class LLMResponse:
    text: str                      # chỉ phần text (thinking bị bỏ)
    content: list                  # toàn bộ content blocks -> gửi lại nguyên vẹn ở lượt sau
    stop_reason: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    latency_ms: float = 0.0
    request_id: str | None = None
    extra: dict = field(default_factory=dict)


class ClaudeClient:
    def __init__(self, model: str | None = None, effort: str | None = None,
                 max_tokens: int = MAX_OUTPUT_TOKENS):
        import anthropic

        self.client = anthropic.Anthropic()   # đọc ANTHROPIC_API_KEY từ môi trường
        self.model = model or os.environ.get("REACT_MODEL") or os.environ.get("ANTHROPIC_MODEL") or DEFAULT_MODEL
        self.effort = effort or os.environ.get("REACT_EFFORT") or DEFAULT_EFFORT
        self.max_tokens = max_tokens

    def complete(self, system: str, messages: List[dict], stop: List[str] | None = None) -> LLMResponse:
        extra = {"stop_sequences": stop} if stop else {}
        t0 = time.perf_counter()
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
            output_config={"effort": self.effort},
            **extra,
        )
        latency = (time.perf_counter() - t0) * 1000
        u = resp.usage
        return LLMResponse(
            text="".join(b.text for b in resp.content if b.type == "text"),
            # to_dict(): chỉ giữ các field API trả về -> gửi lại được ở lượt sau
            content=[b.to_dict() for b in resp.content],
            stop_reason=resp.stop_reason,
            model=resp.model,
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
            latency_ms=latency,
            request_id=getattr(resp, "_request_id", None),
        )
