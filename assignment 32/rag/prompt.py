"""Bước 5: prompt = [system] + [chunks] + [user query]."""
from __future__ import annotations

from typing import List, Tuple

from .vector_store import SearchResult

SYSTEM_PROMPT = """You are a careful question-answering assistant.
Answer the user's question using ONLY the information inside <context>.
Rules:
- Cite the chunks you used with their numbers, e.g. [1] or [2][4].
- If the context does not contain the answer, say exactly: "I don't know based on the provided document."
- Do not use outside knowledge. Be concise (at most a few sentences).
- Reply in the same language as the question."""


def format_context(results: List[SearchResult]) -> str:
    blocks = []
    for r in results:
        blocks.append(f'<chunk id="{r.rank}" score="{r.score:.3f}">\n{r.chunk.text}\n</chunk>')
    return "<context>\n" + "\n\n".join(blocks) + "\n</context>"


def build_prompt(query: str, results: List[SearchResult]) -> Tuple[str, List[dict]]:
    """Trả về (system, messages) đúng format của Anthropic Messages API."""
    user_content = f"{format_context(results)}\n\nQuestion: {query}"
    return SYSTEM_PROMPT, [{"role": "user", "content": user_content}]


def render_prompt(system: str, messages: List[dict]) -> str:
    """In prompt ra dạng đọc được (dùng cho --dry-run / debug)."""
    parts = [f"===== [SYSTEM] =====\n{system}"]
    for m in messages:
        parts.append(f"===== [{m['role'].upper()}] =====\n{m['content']}")
    return "\n\n".join(parts)
