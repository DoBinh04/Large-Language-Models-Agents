"""Structured logging (stdlib, không cần thư viện ngoài).

    from support.logging_setup import get_logger
    log = get_logger(__name__)
    log.info("turn done", extra={"intent": "return_request", "cost_usd": 0.0123})

- LOG_FORMAT=json (mặc định trong Docker): mỗi dòng là 1 JSON object
      {"ts": "...", "level": "INFO", "logger": "support.orchestrator", "msg": "turn done",
       "trace_id": "9f2c...", "span_id": "...", "intent": "return_request", "cost_usd": 0.0123}
  -> đưa thẳng vào Loki / ELK / CloudWatch, lọc theo trace_id để nối log với trace JSONL (support/tracing.py).
- LOG_FORMAT=text: dạng dễ đọc khi chạy local, các field extra in ở cuối dòng dạng key=value.
- LOG_LEVEL (mặc định INFO; run_chat.py dùng WARNING để không lẫn với hội thoại), LOG_FILE (tuỳ chọn, luôn JSON).
- Log luôn ra stderr: MCP stdio server dùng stdout cho giao thức, ghi log vào stdout sẽ làm hỏng kết nối.
- trace_id / span_id lấy từ span đang mở (contextvars), không cần truyền tay.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
from datetime import datetime, timezone

# Thuộc tính sẵn có của LogRecord; mọi thứ khác trong record.__dict__ là field "extra" của người gọi.
_RESERVED = set(vars(logging.LogRecord("x", 0, "x", 0, "x", None, None))) | {"message", "asctime", "trace_id", "span_id"}
_lock = threading.Lock()
_configured = False


class ContextFilter(logging.Filter):
    """Gắn trace_id / span_id của span hiện tại vào mọi log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        from support.tracing import current_span

        if getattr(record, "trace_id", None):                 # người gọi tự truyền trace_id qua extra
            record.span_id = getattr(record, "span_id", None)
            return True
        sp = current_span()
        record.trace_id = sp.trace_id if sp else None
        record.span_id = sp.span_id if sp else None
        return True


def _extras(record: logging.LogRecord) -> dict:
    return {k: v for k, v in record.__dict__.items() if k not in _RESERVED and not k.startswith("_")}


def _jsonable(v):
    try:
        json.dumps(v)
        return v
    except (TypeError, ValueError):
        return str(v)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if getattr(record, "trace_id", None):
            out["trace_id"] = record.trace_id
            out["span_id"] = record.span_id
        out.update({k: _jsonable(v) for k, v in _extras(record).items()})
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    def __init__(self):
        super().__init__("%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        extra = _extras(record)
        if getattr(record, "trace_id", None):
            extra = {"trace": record.trace_id, **extra}
        if extra:
            line += "  " + " ".join(f"{k}={json.dumps(_jsonable(v), ensure_ascii=False)}" for k, v in extra.items())
        return line


def setup_logging(level: str | None = None, fmt: str | None = None, default_level: str = "INFO",
                  force: bool = False) -> None:
    """Idempotent. Gọi ở entrypoint (api, run_chat, mcp_server); get_logger() tự gọi nếu chưa ai gọi."""
    global _configured
    with _lock:
        if _configured and not force:
            return
        level = (level or os.environ.get("LOG_LEVEL") or default_level).upper()
        fmt = (fmt or os.environ.get("LOG_FORMAT") or "text").lower()

        root = logging.getLogger()
        for h in list(root.handlers):
            if getattr(h, "_techviet", False):
                root.removeHandler(h)

        handlers: list[logging.Handler] = []
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())
        handlers.append(console)
        if path := os.environ.get("LOG_FILE"):
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            fh = logging.FileHandler(path, encoding="utf-8")
            fh.setFormatter(JsonFormatter())
            handlers.append(fh)
        for h in handlers:
            h._techviet = True                                   # type: ignore[attr-defined]
            h.addFilter(ContextFilter())
            root.addHandler(h)
        root.setLevel(level)
        # thư viện ồn ào
        for noisy in ("httpx", "httpcore", "anthropic", "urllib3", "sentence_transformers", "filelock"):
            logging.getLogger(noisy).setLevel(max(logging.WARNING, root.level))
        _configured = True


def get_logger(name: str) -> logging.Logger:
    if not _configured:
        setup_logging()
    return logging.getLogger(name)
