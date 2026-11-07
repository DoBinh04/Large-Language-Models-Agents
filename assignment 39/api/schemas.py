"""Pydantic models cho request / response của API."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, examples=["Màn hình mình mới mua bị điểm chết"])
    session_id: str | None = Field(None, max_length=64, description="Bỏ trống để tạo hội thoại mới")


class TurnMetrics(BaseModel):
    latency_ms: float
    llm_calls: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cost_usd: float


class ChatResponse(BaseModel):
    session_id: str
    turn: int
    reply: str
    intent: str
    agents: list[str]
    missing_info: list[str]
    plan: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    agent_results: list[dict[str, Any]]
    trace_id: str
    metrics: TurnMetrics


class SessionInfo(BaseModel):
    session_id: str
    turn: int
    created_at: float
    last_used: float
    state: dict[str, Any]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, examples=["tai nghe tws bóc seal r đổi đc k"])
    top_k: int = Field(4, ge=1, le=20)
    rewrite: Literal["off", "rules", "llm"] | None = Field(None, description="Mặc định theo env QUERY_REWRITE")


class SearchHit(BaseModel):
    chunk_id: str
    doc: str
    section: str
    text: str
    rrf: float
    bm25_rank: int | None = None
    dense_rank: int | None = None
    rerank_score: float | None = None


class SearchResponse(BaseModel):
    rag_mode: str
    rewrite: dict[str, Any]
    hits: list[SearchHit]
    latency_ms: float


class HealthResponse(BaseModel):
    status: Literal["ok"]
    router: str
    agent_model: str
    tools_backend: str
    query_rewrite: str
    rag_loaded: bool
    rag_mode: str | None
    sessions: int
    uptime_s: float
