"""Wrapper gọi Claude dùng chung cho Router (API) / Product / Order / Policy agent / Judge.

- Mỗi lời gọi = 1 span "llm" (model, effort, tokens, cost, latency).
- Prompt caching: system prompt + danh sách tool được đánh cache_control -> các lượt sau trong
  vòng tool-use và các request sau đọc từ cache (rẻ hơn ~20 lần với Opus 5.5).
- Refusal: bật server-side fallback (beta `server-side-fallback-2026-07-01`, fallbacks="default")
  cho các model hỗ trợ; vẫn kiểm tra stop_reason == "refusal" trước khi đọc content.
- Không gửi temperature (API version mới trả 400); độ sâu suy luận điều khiển bằng output_config.effort.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import config
from support.tracing import NULL_TRACER, Tracer, llm_cost


class RefusalError(RuntimeError):
    pass


class BudgetExceeded(BaseException):
    """Hết ngân sách API. Kế thừa BaseException để các `except Exception` (A2A, eval) không nuốt mất -> dừng hẳn."""


# ---- Sổ chi tiêu dùng chung mọi process (bật khi đặt API_BUDGET_USD) -------------------------------------
def _ledger_path() -> Path:
    return Path(os.environ.get("SPEND_LEDGER") or config.RESULTS_DIR / "spend.json")


@contextmanager
def _locked():
    path = _ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path) + ".lock", "w") as lk:
        try:
            import fcntl

            fcntl.flock(lk, fcntl.LOCK_EX)
        except ImportError:                       # Windows: chạy tuần tự là đủ
            pass
        yield path


def spent() -> dict:
    p = _ledger_path()
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"total_usd": 0.0, "calls": 0, "by_model": {}}


def _budget() -> float:
    return float(os.environ.get("API_BUDGET_USD", "0") or 0)


def check_budget() -> None:
    b = _budget()
    if b and spent()["total_usd"] >= b:
        raise BudgetExceeded(f"Đã tiêu {spent()['total_usd']:.4f}$ >= ngân sách {b}$ (API_BUDGET_USD)")


def record_spend(model: str, usd: float) -> None:
    if not _budget():
        return
    with _locked() as p:
        d = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"total_usd": 0.0, "calls": 0, "by_model": {}}
        d["total_usd"] = round(d["total_usd"] + usd, 6)
        d["calls"] += 1
        d["by_model"][model] = round(d["by_model"].get(model, 0.0) + usd, 6)
        p.write_text(json.dumps(d, indent=1), encoding="utf-8")


def _usage_dict(u) -> dict:
    return {"input_tokens": u.input_tokens or 0, "output_tokens": u.output_tokens or 0,
            "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
            "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0}


def text_of(resp) -> str:
    return "".join(b.text for b in resp.content if b.type == "text")


@dataclass
class Claude:
    model: str = config.BASE_MODEL
    effort: str = config.BASE_EFFORT
    max_tokens: int = config.AGENT_MAX_TOKENS
    tracer: Tracer = field(default=NULL_TRACER)
    name: str = "llm"

    def __post_init__(self):
        import anthropic

        self.client = anthropic.Anthropic()

    @property
    def label(self) -> str:
        return f"{self.model}@{self.effort}"

    # ------------------------------------------------------------------ #
    def create(self, system: str, messages: list[dict], tools: list[dict] | None = None,
               schema: dict | None = None, tool_choice: dict | None = None):
        output_config: dict = {"effort": self.effort}
        if schema:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        kw: dict = {
            "model": self.model, "max_tokens": self.max_tokens, "messages": messages,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "output_config": output_config,
        }
        if tools:
            tools = [dict(t) for t in tools]
            tools[-1]["cache_control"] = {"type": "ephemeral"}       # cache toàn bộ khối tools
            kw["tools"] = tools
            kw["tool_choice"] = tool_choice or {"type": "auto"}

        check_budget()
        with self.tracer.span(f"llm.{self.name}", kind="llm", model=self.model, effort=self.effort) as sp:
            t0 = time.perf_counter()
            if config.USE_FALLBACKS and self.model in config.FALLBACK_MODELS:
                resp = self.client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kw)
            else:
                resp = self.client.messages.create(**kw)
            usage = _usage_dict(resp.usage)
            served_by = getattr(resp, "model", self.model) or self.model
            cost = llm_cost(served_by, usage)
            record_spend(served_by, cost)
            sp.set(**usage, served_by=served_by, stop_reason=resp.stop_reason,
                   cost_usd=cost, api_latency_ms=round((time.perf_counter() - t0) * 1000, 1))
            if resp.stop_reason == "refusal":
                raise RefusalError(f"model từ chối: {getattr(resp, 'stop_details', None)}")
            return resp

    # ------------------------------------------------------------------ #
    def structured(self, system: str, user: str | list[dict], schema: dict) -> dict:
        messages = [{"role": "user", "content": user}] if isinstance(user, str) else user
        resp = self.create(system, messages, schema=schema)
        if resp.stop_reason == "max_tokens":
            raise RuntimeError("structured output bị cắt do max_tokens")
        return json.loads(text_of(resp))

    def text(self, system: str, user: str) -> str:
        return text_of(self.create(system, [{"role": "user", "content": user}]))

    def tool_loop(self, system: str, messages: list[dict], tools: list[dict],
                  execute: Callable[[str, dict], dict], max_turns: int = config.MAX_TOOL_TURNS) -> tuple[str, list[dict]]:
        """Vòng tool use thủ công. Trả về (câu trả lời cuối, danh sách tool call đã thực hiện)."""
        messages = list(messages)
        calls: list[dict] = []
        for turn in range(max_turns):
            last = turn == max_turns - 1
            resp = self.create(system, messages, tools=tools,
                               tool_choice={"type": "none"} if last else {"type": "auto"})
            # giữ nguyên các block (kể cả thinking) để gửi lại đúng thứ tự; bỏ block "fallback" (metadata)
            messages.append({"role": "assistant",
                             "content": [b.to_dict() for b in resp.content if b.type != "fallback"]})
            if resp.stop_reason != "tool_use":
                return text_of(resp), calls
            results = []
            for b in resp.content:
                if b.type != "tool_use":
                    continue
                out = execute(b.name, dict(b.input))
                calls.append({"name": b.name, "args": dict(b.input)})
                results.append({"type": "tool_result", "tool_use_id": b.id,
                                "content": json.dumps(out, ensure_ascii=False),
                                "is_error": isinstance(out, dict) and "error" in out})
            messages.append({"role": "user", "content": results})
        return "Xin lỗi, tôi chưa xử lý xong yêu cầu này.", calls
