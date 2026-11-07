"""Observability: tracing dạng span lồng nhau + tính token / chi phí / latency.

    with tracer.span("agent.order", kind="agent", input=...) as sp:
        ...
        sp.set(output=..., tokens=...)

- Mỗi request của khách = 1 trace (trace_id). Span con: router, a2a.send, agent, llm, tool, rag.
- Span "llm" mang model, effort, input/output/cache tokens, cost_usd. Tổng của trace = cộng các span llm.
- Ghi JSONL (results/traces/<run>.jsonl), mỗi dòng là 1 span đã đóng -> dashboard.py đọc lại.
- Thread-safe đủ dùng cho eval tuần tự; parent span theo contextvars nên A2A in-process vẫn lồng đúng.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import config

_current: ContextVar["Span | None"] = ContextVar("current_span", default=None)
_current_tracer: ContextVar["Tracer | None"] = ContextVar("current_tracer", default=None)
_log = logging.getLogger("support.tracing")


def llm_cost(model: str, usage: dict) -> float:
    """usage: input_tokens, output_tokens, cache_read_input_tokens, cache_creation_input_tokens."""
    p_in, p_out, p_cache = config.PRICING.get(model, config.PRICING[config.BASE_MODEL])
    return (usage.get("input_tokens", 0) * p_in
            + usage.get("cache_creation_input_tokens", 0) * p_in * config.CACHE_WRITE_MULT
            + usage.get("cache_read_input_tokens", 0) * p_cache
            + usage.get("output_tokens", 0) * p_out) / 1_000_000


def gpu_cost(seconds: float) -> float:
    return seconds / 3600 * config.GPU_COST_PER_HOUR


@dataclass
class Span:
    name: str
    kind: str
    trace_id: str
    parent_id: str | None
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    start: float = field(default_factory=time.time)
    attrs: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    latency_ms: float = 0.0

    def set(self, **kw) -> None:
        self.attrs.update(kw)

    def to_dict(self) -> dict:
        return {"trace_id": self.trace_id, "span_id": self.span_id, "parent_id": self.parent_id,
                "name": self.name, "kind": self.kind, "start": self.start,
                "latency_ms": round(self.latency_ms, 1), "status": self.status, "attrs": self.attrs}


class Tracer:
    def __init__(self, run_name: str | None = None, out_dir: Path = config.TRACE_DIR, enabled: bool = True,
                 max_spans: int | None = None):
        """max_spans: chỉ giữ N span gần nhất trong RAM (API server chạy lâu); file JSONL vẫn ghi đủ."""
        self.enabled = enabled
        self.spans: list[dict] | deque = deque(maxlen=max_spans) if max_spans else []
        self._lock = threading.Lock()
        self.path = None
        if enabled:
            out_dir.mkdir(parents=True, exist_ok=True)
            self.path = out_dir / f"{run_name or time.strftime('%Y%m%d-%H%M%S')}.jsonl"

    @contextmanager
    def span(self, name: str, kind: str = "internal", trace_id: str | None = None, **attrs):
        parent = _current.get()
        sp = Span(name=name, kind=kind, trace_id=trace_id or (parent.trace_id if parent else uuid.uuid4().hex[:16]),
                  parent_id=parent.span_id if parent else None, attrs=_safe(attrs))
        token = _current.set(sp)
        ttoken = _current_tracer.set(self)
        t0 = time.perf_counter()
        try:
            yield sp
        except Exception as e:
            sp.status = "error"
            sp.attrs["error"] = f"{type(e).__name__}: {e}"
            _log.warning("span failed", extra={"span": name, "kind": kind, "error": sp.attrs["error"]})
            raise
        finally:
            sp.latency_ms = (time.perf_counter() - t0) * 1000
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug("span closed", extra={"span": name, "kind": kind,
                                                 "latency_ms": round(sp.latency_ms, 1), "status": sp.status})
            _current.reset(token)
            _current_tracer.reset(ttoken)
            self._emit(sp)

    def _emit(self, sp: Span) -> None:
        d = sp.to_dict()
        d["attrs"] = _safe(d["attrs"])
        line = json.dumps(d, ensure_ascii=False) + "\n"
        with self._lock:                                   # API server: nhiều request ghi cùng file
            self.spans.append(d)
            if self.path:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(line)

    # ---- tổng hợp cho 1 trace -------------------------------------------------
    def summary(self, trace_id: str) -> dict:
        with self._lock:
            spans = [s for s in self.spans if s["trace_id"] == trace_id]
        llm = [s for s in spans if s["kind"] == "llm"]
        tools = [s for s in spans if s["kind"] == "tool"]
        root = next((s for s in spans if s["parent_id"] is None), None)
        return {
            "trace_id": trace_id,
            "latency_ms": root["latency_ms"] if root else sum(s["latency_ms"] for s in spans),
            "llm_calls": len(llm),
            "input_tokens": sum(s["attrs"].get("input_tokens", 0) for s in llm),
            "output_tokens": sum(s["attrs"].get("output_tokens", 0) for s in llm),
            "cache_read_tokens": sum(s["attrs"].get("cache_read_input_tokens", 0) for s in llm),
            "cost_usd": round(sum(s["attrs"].get("cost_usd", 0.0) for s in spans), 6),
            "tool_calls": [{"name": s["attrs"].get("tool"), "args": s["attrs"].get("args")} for s in tools],
            "errors": [s["name"] for s in spans if s["status"] == "error"],
        }


def current_trace_id() -> str | None:
    sp = _current.get()
    return sp.trace_id if sp else None


def current_span() -> Span | None:
    return _current.get()


def current_tracer() -> "Tracer | None":
    """Tracer của span đang mở: code sâu bên dưới (vd. query rewriter trong tool) ghi span vào đúng trace."""
    return _current_tracer.get()


def _safe(obj: Any, limit: int = 4000) -> Any:
    """Cắt bớt chuỗi dài để file trace không phình."""
    if isinstance(obj, str):
        return obj if len(obj) <= limit else obj[:limit] + f"...(+{len(obj) - limit})"
    if isinstance(obj, dict):
        return {k: _safe(v, limit) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe(v, limit) for v in obj]
    if isinstance(obj, (int, float, bool)) or obj is None:
        return obj
    return _safe(str(obj), limit)


NULL_TRACER = Tracer(enabled=False, max_spans=1000)   # không ghi file; giới hạn RAM khi chạy lâu
