"""Tracer: ghi mọi observation ra JSONL cục bộ và (nếu cấu hình) gửi Langfuse.

Cấu trúc lồng nhau giống Langfuse:

    trace  rag.query                      (1 câu hỏi của user)
     ├─ retrieval rag.retrieve            query, top-k chunks, similarity scores
     └─ generation rag.generate           model, prompt hash, tokens in/out, latency, cost

    trace  react.run
     ├─ generation react.step1
     ├─ tool tool.search_web
     ├─ generation react.step2
     └─ ...

Quan hệ cha/con lấy từ contextvars, nên code được instrument không phải truyền tracer/parent đi khắp nơi.
Mỗi observation khi KẾT THÚC sinh đúng 1 dòng JSON trong logs/traces-YYYYMMDD.jsonl.
dashboard.py đọc các file này.
"""
from __future__ import annotations

import contextvars
import hashlib
import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .langfuse_sink import LangfuseSink
from .pricing import estimate_cost

MAX_FIELD_CHARS = int(os.environ.get("OBS_MAX_FIELD_CHARS", "4000"))
_current: contextvars.ContextVar["Observation | None"] = contextvars.ContextVar("current_observation", default=None)


def prompt_hash(model: str, system: Any, messages: Any) -> str:
    """Hash ổn định của toàn bộ prompt. Cùng hash = cùng request = có thể cache / so sánh."""
    blob = json.dumps({"model": model, "system": system, "messages": messages},
                      sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def short_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:12]


def _truncate(value: Any) -> Any:
    """Giữ log gọn: chuỗi dài bị cắt, object lớn được serialize rồi cắt."""
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        return value if len(value) <= MAX_FIELD_CHARS else value[:MAX_FIELD_CHARS] + f"...[+{len(value) - MAX_FIELD_CHARS} chars]"
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= MAX_FIELD_CHARS:
        return value
    return text[:MAX_FIELD_CHARS] + f"...[+{len(text) - MAX_FIELD_CHARS} chars]"


class Observation:
    def __init__(self, tracer: "Tracer", kind: str, name: str, input=None, metadata=None, **fields):
        parent = _current.get()
        self.tracer = tracer
        self.id = uuid.uuid4().hex[:16]
        self.parent_id = parent.id if parent else None
        self.trace_id = parent.trace_id if parent else self.id
        self.record: dict = {
            "kind": kind, "name": name, "input": input, "metadata": dict(metadata or {}),
            "status": "ok", **fields,
        }
        self._token = None
        self._lf = None
        self._t0 = 0.0

    # ---- API cho code được instrument ----------------------------------------
    def update(self, **fields) -> "Observation":
        meta = fields.pop("metadata", None)
        if meta:
            self.record["metadata"].update(meta)
        self.record.update(fields)
        return self

    def set_usage(self, model: str, input_tokens: int, output_tokens: int,
                  cache_read_tokens: int = 0, cache_write_tokens: int = 0, estimated: bool = False) -> "Observation":
        cost = estimate_cost(model, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens)
        return self.update(model=model, input_tokens=input_tokens, output_tokens=output_tokens,
                           cache_read_tokens=cache_read_tokens, cache_write_tokens=cache_write_tokens,
                           cost_usd=cost["total"] if cost else None, cost_details=cost,
                           usage_estimated=estimated)

    # ---- context manager -------------------------------------------------------
    def __enter__(self) -> "Observation":
        self._t0 = time.perf_counter()
        self.record["start_time"] = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        trace_attrs = None
        if self.parent_id is None:
            trace_attrs = {"trace_name": self.record["name"],
                           "tags": [t for t in (self.record.get("tags") or []) if t] or None,
                           "session_id": self.record.get("session_id")}
        self._lf = self.tracer.langfuse.start(self.record["kind"], self.record["name"], self.record["input"],
                                              self.record["metadata"], self.record.get("model"), trace_attrs)
        self._token = _current.set(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        _current.reset(self._token)
        self.record["latency_ms"] = round((time.perf_counter() - self._t0) * 1000, 2)
        if exc is not None:
            self.record["status"] = "error"
            self.record["error"] = f"{exc_type.__name__}: {exc}"
            if self.parent_id is None:
                self.record.setdefault("failed", True)
                self.record.setdefault("failure_reason", "exception")
        full = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "run_id": self.tracer.run_id, "trace_id": self.trace_id, "obs_id": self.id,
                "parent_id": self.parent_id, **self.record}
        self.tracer.langfuse.end(self._lf, full, (exc_type, exc, tb))
        if self.parent_id is None:
            full["langfuse_url"] = self.tracer.langfuse.trace_url(self._lf)
        self.tracer.write(full)
        return False                       # không nuốt exception


class Tracer:
    def __init__(self, log_dir: str | Path, run_id: str | None = None, use_langfuse: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.langfuse = LangfuseSink(enabled=use_langfuse)
        self._lock = threading.Lock()

    @property
    def log_file(self) -> Path:
        return self.log_dir / f"traces-{datetime.now():%Y%m%d}.jsonl"

    def write(self, record: dict) -> None:
        line = json.dumps({k: _truncate(v) if k in ("input", "output") else v for k, v in record.items()},
                          ensure_ascii=False, default=str)
        with self._lock, open(self.log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    # ---- các loại observation --------------------------------------------------
    def trace(self, name: str, input=None, metadata=None, tags=None, session_id=None) -> Observation:
        return Observation(self, "trace", name, input, metadata, tags=tags, session_id=session_id)

    def span(self, name: str, input=None, metadata=None, kind: str = "span") -> Observation:
        return Observation(self, kind, name, input, metadata)

    def retrieval(self, name: str, query: str, top_k: int, metadata=None) -> Observation:
        return Observation(self, "retrieval", name, {"query": query, "top_k": top_k}, metadata)

    def tool(self, name: str, input=None, metadata=None) -> Observation:
        return Observation(self, "tool", name, input, metadata)

    def generation(self, name: str, model: str, system=None, messages=None, metadata=None,
                   model_parameters: dict | None = None) -> Observation:
        ph = prompt_hash(model, system, messages)
        return Observation(self, "generation", name,
                           {"system": system, "messages": messages}, metadata,
                           model=model, prompt_hash=ph, system_hash=short_hash(system),
                           model_parameters=model_parameters)

    def flush(self) -> None:
        self.langfuse.flush()


_default: Tracer | None = None


def get_tracer(log_dir: str | Path | None = None, **kw) -> Tracer:
    global _default
    if _default is None:
        _default = Tracer(log_dir or Path(__file__).resolve().parent.parent / "logs", **kw)
    return _default
