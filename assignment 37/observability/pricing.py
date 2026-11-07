"""Ước tính chi phí LLM call (USD).

Giá theo bảng giá Anthropic API (USD / 1 triệu token), cập nhật 2026-10.
Kiểm tra lại tại https://www.anthropic.com/pricing trước khi dùng số liệu cho báo cáo chính thức.

Prompt caching: ghi cache = 1.25 × giá input (TTL 5 phút), đọc cache = 0.1 × giá input.
Claude Haiku 5.5 có giá cao hơn cho prompt > 100K tokens; RAG/agent ở đây không chạm ngưỡng đó nên bỏ qua.
"""
from __future__ import annotations

PRICES_PER_MTOK: dict[str, tuple[float, float]] = {   # model -> (input, output)
    "claude-fable-5-1": (10.00, 50.00),
    "claude-fable-5": (10.00, 50.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5": (1.00, 5.00),
}
CACHE_WRITE_MULT = 1.25
CACHE_READ_MULT = 0.10
# Model có giá đọc cache riêng, không theo 0.1 × input (USD / 1M token)
CACHE_READ_PER_MTOK: dict[str, float] = {
    "claude-fable-5-1": 0.25,
    "claude-opus-5-5": 0.20,
    "claude-sonnet-5-5": 0.20,
}


def _match(model: str, table: dict):
    if model in table:
        return table[model]
    # model id có hậu tố (vd snapshot) -> khớp tiền tố dài nhất
    matches = [k for k in table if model.startswith(k)]
    return table[max(matches, key=len)] if matches else None


def _lookup(model: str) -> tuple[float, float] | None:
    return _match(model, PRICES_PER_MTOK)


def estimate_cost(model: str, input_tokens: int, output_tokens: int,
                  cache_read_tokens: int = 0, cache_write_tokens: int = 0) -> dict | None:
    """Trả về {"input", "output", "cache_read", "cache_write", "total"} (USD), hoặc None nếu không biết giá."""
    price = _lookup(model or "")
    if price is None:
        return None
    p_in, p_out = price[0] / 1e6, price[1] / 1e6
    read_override = _match(model or "", CACHE_READ_PER_MTOK)
    p_read = read_override / 1e6 if read_override is not None else p_in * CACHE_READ_MULT
    cost = {
        "input": input_tokens * p_in,
        "output": output_tokens * p_out,
        "cache_read": cache_read_tokens * p_read,
        "cache_write": cache_write_tokens * p_in * CACHE_WRITE_MULT,
    }
    cost["total"] = sum(cost.values())
    return cost
