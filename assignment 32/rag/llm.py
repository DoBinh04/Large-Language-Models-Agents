"""Bước 6: gọi LLM.

ClaudeLLM  : gọi Anthropic Messages API (cần ANTHROPIC_API_KEY).
DryRunLLM  : khi chưa có key -> không gọi API, trả về prompt để bạn kiểm tra.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

from .prompt import render_prompt


class LLMRequestError(RuntimeError):
    """Lỗi có thể hành động được khi provider không sinh được câu trả lời."""


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


class ClaudeLLM:
    def __init__(self, model: str, max_tokens: int = 512, temperature: float = 0.0):
        import anthropic

        self.client = anthropic.Anthropic()  # tự đọc ANTHROPIC_API_KEY
        self.model = model
        self.max_tokens = max_tokens
        # SDK anthropic >= 1.x / API 2026-10-21 đã bỏ temperature/top_p/top_k (gửi lên -> lỗi 400).
        # Giữ tham số để không phá chữ ký hàm, nhưng KHÔNG gửi lên API.
        self.temperature = temperature

    def generate(self, system: str, messages: List[dict]) -> str:
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=messages,
            )
        except Exception as exc:
            raise LLMRequestError(f"Claude API không thể tạo câu trả lời: {exc}") from exc
        return "".join(block.text for block in resp.content if block.type == "text")


class DryRunLLM:
    model = "dry-run (no API call)"

    def generate(self, system: str, messages: List[dict]) -> str:
        return ("[DRY RUN] Chưa có ANTHROPIC_API_KEY nên không gọi LLM. "
                "Prompt sẽ được gửi:\n\n" + render_prompt(system, messages))


def get_llm(default_model: str, max_tokens: int, temperature: float, dry_run: bool = False):
    if dry_run or not os.environ.get("ANTHROPIC_API_KEY"):
        return DryRunLLM()
    model = os.environ.get("ANTHROPIC_MODEL", default_model)
    return ClaudeLLM(model, max_tokens, temperature)
