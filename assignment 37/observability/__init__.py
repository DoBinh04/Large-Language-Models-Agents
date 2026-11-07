"""Observability cho RAG pipeline (3.2) và ReAct agent (3.3).

tracer.py          : Tracer / Observation (trace > span > generation), ghi JSONL cục bộ + gửi Langfuse
langfuse_sink.py   : adapter Langfuse Python SDK (tự tắt nếu chưa cấu hình / lỗi mạng)
pricing.py         : bảng giá -> ước tính cost mỗi LLM call
instrument_rag.py  : TracedClaudeLLM, TracedRAGPipeline   (subclass, không sửa code 3.2)
instrument_agent.py: TracedReActAgent                     (override hook của 3.3)
"""
from .tracer import Tracer, get_tracer  # noqa: F401
