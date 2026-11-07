"""Gửi observation sang Langfuse (Python SDK v3+, nền tảng OpenTelemetry).

Nguyên tắc: observability KHÔNG BAO GIỜ được làm hỏng ứng dụng.
Mọi lời gọi Langfuse đều được bọc try/except. Lỗi thì ghi một cảnh báo ra stderr và bỏ qua.

Cấu hình qua biến môi trường (xem .env.example):
    LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST (hoặc LANGFUSE_BASE_URL)

Các bản SDK khác nhau đặt tên API hơi khác nhau, nên adapter ưu tiên
`start_as_current_observation(as_type=...)` (SDK mới) và lùi về
`start_as_current_span / start_as_current_generation` (SDK v3).
"""
from __future__ import annotations

import os
import sys

# kind của tracer -> as_type của Langfuse
AS_TYPE = {"trace": "span", "span": "span", "generation": "generation", "retrieval": "retriever", "tool": "tool"}


class LangfuseSink:
    def __init__(self, enabled: bool = True):
        self.client = None
        self._warned = set()
        if not enabled or not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
            return
        # SDK mới đọc LANGFUSE_BASE_URL, SDK v3 đọc LANGFUSE_HOST -> đồng bộ hai biến
        if os.environ.get("LANGFUSE_HOST") and not os.environ.get("LANGFUSE_BASE_URL"):
            os.environ["LANGFUSE_BASE_URL"] = os.environ["LANGFUSE_HOST"]
        if os.environ.get("LANGFUSE_BASE_URL") and not os.environ.get("LANGFUSE_HOST"):
            os.environ["LANGFUSE_HOST"] = os.environ["LANGFUSE_BASE_URL"]
        try:
            from langfuse import get_client

            self.client = get_client()
        except Exception as e:
            self._warn("init", e)

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def _warn(self, where: str, e: Exception) -> None:
        if where not in self._warned:
            self._warned.add(where)
            print(f"[observability] Langfuse {where} lỗi, bỏ qua: {type(e).__name__}: {e}", file=sys.stderr)

    # ---- vòng đời một observation -----------------------------------------------
    def start(self, kind: str, name: str, input=None, metadata=None, model=None, trace_attrs: dict | None = None):
        """Mở observation và đặt nó làm 'current' (để observation con lồng vào). Trả về handle.

        trace_attrs (chỉ cho observation gốc): trace_name, tags, session_id.
        SDK v4+ bỏ update_trace(); thuộc tính trace được gắn qua propagate_attributes(),
        phải mở TRƯỚC observation gốc để mọi span con đều mang cùng tags/session.
        """
        if not self.client:
            return None
        kwargs = {"name": name, "input": input, "metadata": metadata}
        if model:
            kwargs["model"] = model
        pcm = None
        if trace_attrs:
            try:
                from langfuse import propagate_attributes

                attrs = {k: v for k, v in trace_attrs.items() if v}
                pcm = propagate_attributes(**attrs)
                pcm.__enter__()
            except ImportError:
                pcm = None                              # SDK v3 cũ -> dùng update_trace ở end()
            except Exception as e:
                self._warn("propagate", e)
                pcm = None
        try:
            if hasattr(self.client, "start_as_current_observation"):
                as_type = AS_TYPE.get(kind, "span")
                try:
                    cm = self.client.start_as_current_observation(as_type=as_type, **kwargs)
                except (TypeError, ValueError):     # SDK cũ chưa có as_type "retriever"/"tool"
                    if kind == "generation":
                        raise
                    kwargs.pop("model", None)
                    cm = self.client.start_as_current_observation(as_type="span", **kwargs)
            elif kind == "generation":
                cm = self.client.start_as_current_generation(**kwargs)
            else:
                kwargs.pop("model", None)
                cm = self.client.start_as_current_span(**kwargs)
            obs = cm.__enter__()
            return cm, obs, pcm
        except Exception as e:
            self._warn("start", e)
            if pcm is not None:
                try:
                    pcm.__exit__(None, None, None)
                except Exception:
                    pass
            return None

    def end(self, handle, record: dict, exc_info=(None, None, None)) -> None:
        if handle is None:
            return
        cm, obs, pcm = handle
        try:
            update = {"output": record.get("output"), "metadata": record.get("metadata") or None}
            if record.get("status") == "error":
                update["level"] = "ERROR"
                update["status_message"] = record.get("error")
            elif record.get("failed"):
                update["level"] = "WARNING"
                update["status_message"] = record.get("failure_reason")
            if record["kind"] == "generation":
                update["usage_details"] = {k: v for k, v in {
                    "input": record.get("input_tokens"),
                    "output": record.get("output_tokens"),
                    "cache_read_input_tokens": record.get("cache_read_tokens"),
                    "cache_creation_input_tokens": record.get("cache_write_tokens"),
                }.items() if v}
                if record.get("cost_details"):
                    # key của cost_details phải trùng key của usage_details để Langfuse ghép đúng
                    c = record["cost_details"]
                    update["cost_details"] = {"input": c["input"], "output": c["output"],
                                              "cache_read_input_tokens": c["cache_read"],
                                              "cache_creation_input_tokens": c["cache_write"],
                                              "total": c["total"]}
            if record["kind"] == "retrieval":
                update["metadata"] = {**(update["metadata"] or {}), "retrieval": record.get("retrieval")}
            obs.update(**{k: v for k, v in update.items() if v is not None})

            if record.get("parent_id") is None:            # observation gốc -> cập nhật trace
                trace_fields = {"name": record["name"], "input": record.get("input"),
                                "output": record.get("output"), "tags": record.get("tags"),
                                "session_id": record.get("session_id"), "metadata": record.get("metadata")}
                if pcm is None and hasattr(obs, "update_trace"):      # SDK v3; v4+ đã dùng propagate_attributes
                    obs.update_trace(**{k: v for k, v in trace_fields.items() if v is not None})
                if hasattr(obs, "score_trace") and "failed" in record:
                    obs.score_trace(name="failed", value=1.0 if record["failed"] else 0.0,
                                    comment=record.get("failure_reason"))
        except Exception as e:
            self._warn("update", e)
        finally:
            try:
                cm.__exit__(*exc_info)
            except Exception as e:
                self._warn("end", e)
            if pcm is not None:
                try:
                    pcm.__exit__(*exc_info)
                except Exception as e:
                    self._warn("end", e)

    def trace_url(self, handle) -> str | None:
        if handle is None:
            return None
        try:
            tid = getattr(handle[1], "trace_id", None)
            return self.client.get_trace_url(trace_id=tid) if tid else None
        except Exception:
            return None

    def flush(self) -> None:
        if self.client:
            try:
                self.client.flush()
            except Exception as e:
                self._warn("flush", e)
