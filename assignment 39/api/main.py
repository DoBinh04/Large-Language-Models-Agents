"""FastAPI cho hệ thống chăm sóc khách hàng TechViet.

    python -m api.main                                   # 0.0.0.0:8000, log theo LOG_FORMAT
    uvicorn api.main:app --port 8000 --reload            # dev; log access của uvicorn vẫn theo format của uvicorn
    docs: http://localhost:8000/docs

Endpoints
    GET    /health                  liveness + cấu hình đang chạy (không tải model RAG)
    GET    /ready                   readiness: tải RAG index nếu chưa có (lần đầu có thể mất vài chục giây)
    GET    /agents                  A2A AgentCard của 3 agent
    POST   /chat                    1 lượt hội thoại; bỏ trống session_id để mở hội thoại mới
    GET    /sessions/{id}           ConversationState hiện tại
    POST   /sessions/{id}/reset     xoá state hội thoại (giữ session_id)
    DELETE /sessions/{id}
    POST   /rag/search              debug RAG: xem query rewrite + chunk trả về, không gọi agent
    POST   /admin/reset-data        khôi phục dữ liệu đơn hàng / phiếu (cần header X-Admin-Token = ADMIN_TOKEN)

Cấu hình qua env: API_ROUTER, BASE_MODEL, BASE_EFFORT, API_TOOLS, QUERY_REWRITE, RAG_LIGHT, SESSION_TTL_S,
MAX_SESSIONS, WARMUP=1 (tải RAG lúc khởi động), CORS_ORIGINS, LOG_FORMAT, LOG_LEVEL, ADMIN_TOKEN.
"""
from __future__ import annotations

import hmac
import os
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from support.logging_setup import get_logger, setup_logging  # noqa: E402

setup_logging(default_level="INFO", force=True)
log = get_logger("api")

from fastapi import FastAPI, Header, HTTPException, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from api.schemas import (ChatRequest, ChatResponse, HealthResponse, SearchRequest, SearchResponse,  # noqa: E402
                         SessionInfo, TurnMetrics)
from api.sessions import SessionStore  # noqa: E402
from support import rag, tools  # noqa: E402
from support.llm import RefusalError  # noqa: E402
from support.orchestrator import Orchestrator  # noqa: E402
from support.query_rewriter import current_mode, rewrite_query  # noqa: E402
from support.router import make_router  # noqa: E402
from support.tool_backend import make_backend  # noqa: E402
from support.tracing import Tracer  # noqa: E402


class LockedRouter:
    """Small LLM (transformers.generate) không an toàn khi gọi song song trên 1 model -> tuần tự hoá."""

    def __init__(self, inner):
        self.inner = inner
        self._lock = threading.Lock()

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def decide(self, state, user_msg):
        with self._lock:
            return self.inner.decide(state, user_msg)


class AppState:
    tracer: Tracer
    store: SessionStore
    base: Orchestrator
    started: float


S = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    S.started = time.time()
    S.tracer = Tracer(run_name=f"api-{time.strftime('%Y%m%d-%H%M%S')}", max_spans=20_000)
    router = make_router(config.API_ROUTER, S.tracer, model=config.BASE_MODEL, effort=config.BASE_EFFORT)
    if config.API_ROUTER in ("small", "sft", "dpo"):
        router = LockedRouter(router)
    backend = make_backend(config.API_TOOLS)
    S.base = Orchestrator(router, backend, S.tracer, agent_model=config.BASE_MODEL, agent_effort=config.BASE_EFFORT)
    S.store = SessionStore(S.base, ttl_s=config.SESSION_TTL_S, max_sessions=config.MAX_SESSIONS)
    log.info("api started", extra={"router": router.label, "agent_model": f"{config.BASE_MODEL}@{config.BASE_EFFORT}",
                                   "tools": config.API_TOOLS, "query_rewrite": current_mode(),
                                   "trace_file": str(S.tracer.path)})
    if os.environ.get("WARMUP") == "1":
        rag.get_retriever()
    yield
    backend.close()
    log.info("api stopped")


app = FastAPI(title="TechViet Support API", version="1.0.0", lifespan=lifespan,
              description="Multi-agent customer support: Router -> A2A -> Product / Order / Policy agents (MCP tools, Hybrid RAG).")

_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]
if _origins:
    app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_methods=["*"], allow_headers=["*"])


# --------------------------------------------------------------------------- #
# Middleware + lỗi
# --------------------------------------------------------------------------- #
@app.middleware("http")
async def access_log(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.request_id = rid
    t0 = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = rid
        return response
    finally:
        ms = round((time.perf_counter() - t0) * 1000, 1)
        if request.url.path not in ("/health",):              # không spam log healthcheck của Docker
            log.info("http request", extra={"request_id": rid, "method": request.method, "path": request.url.path,
                                            "status": status, "latency_ms": ms})


@app.exception_handler(RefusalError)
async def _refusal(request: Request, exc: RefusalError):
    log.warning("model refusal", extra={"request_id": getattr(request.state, "request_id", None), "error": str(exc)})
    return JSONResponse(status_code=422, content={"error": "refusal", "detail": str(exc),
                                                  "request_id": getattr(request.state, "request_id", None)})


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    rid = getattr(request.state, "request_id", None)
    log.error("unhandled error", exc_info=exc, extra={"request_id": rid, "path": request.url.path})
    status = 502 if type(exc).__module__.startswith("anthropic") else 500
    return JSONResponse(status_code=status, content={"error": type(exc).__name__, "detail": str(exc)[:500],
                                                     "request_id": rid})


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health", response_model=HealthResponse)
def health():
    r = rag._RETRIEVER
    return HealthResponse(status="ok", router=S.base.router.label, agent_model=f"{config.BASE_MODEL}@{config.BASE_EFFORT}",
                          tools_backend=config.API_TOOLS, query_rewrite=current_mode(), rag_loaded=r is not None,
                          rag_mode=r.mode if r else None, sessions=len(S.store),
                          uptime_s=round(time.time() - S.started, 1))


@app.get("/ready")
def ready():
    r = rag.get_retriever()
    return {"ready": True, "rag_mode": r.mode, "chunks": len(r.chunks)}


@app.get("/agents")
def agents():
    return S.base.agent_cards()


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    # def (không async): Orchestrator gọi API đồng bộ -> FastAPI chạy trong threadpool, không chặn event loop
    sess = S.store.get_or_create(req.session_id)
    with sess.lock:
        res = sess.orch.chat(req.message)
        turn = sess.orch.state.turn
    m = S.tracer.summary(res.trace_id)
    d = res.decision
    log.info("turn done", extra={"request_id": request.state.request_id, "session_id": sess.id, "turn": turn,
                                 "trace_id": res.trace_id, "intent": d["intent"], "agents": d["agents"],
                                 "tools": [c["name"] for c in res.tool_calls], "latency_ms": m["latency_ms"],
                                 "llm_calls": m["llm_calls"], "cost_usd": m["cost_usd"], "errors": m["errors"]})
    return ChatResponse(
        session_id=sess.id, turn=turn, reply=res.reply, intent=d["intent"], agents=d["agents"],
        missing_info=d["missing_info"], plan=d["plan"], tool_calls=res.tool_calls, agent_results=res.agent_results,
        trace_id=res.trace_id,
        metrics=TurnMetrics(latency_ms=m["latency_ms"], llm_calls=m["llm_calls"], input_tokens=m["input_tokens"],
                            output_tokens=m["output_tokens"], cache_read_tokens=m["cache_read_tokens"],
                            cost_usd=m["cost_usd"]))


@app.get("/sessions/{session_id}", response_model=SessionInfo)
def get_session(session_id: str):
    s = S.store.get(session_id)
    if not s:
        raise HTTPException(404, "session không tồn tại hoặc đã hết hạn")
    with s.lock:
        st = s.orch.state
        return SessionInfo(session_id=s.id, turn=st.turn, created_at=s.created_at, last_used=s.last_used,
                           state=st.to_dict())


@app.post("/sessions/{session_id}/reset")
def reset_session(session_id: str):
    if not S.store.reset(session_id):
        raise HTTPException(404, "session không tồn tại hoặc đã hết hạn")
    return {"ok": True, "session_id": session_id}


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    if not S.store.delete(session_id):
        raise HTTPException(404, "session không tồn tại hoặc đã hết hạn")
    return {"ok": True}


@app.post("/rag/search", response_model=SearchResponse)
def rag_search(req: SearchRequest):
    t0 = time.perf_counter()
    with S.tracer.span("api.rag_search", kind="rag", query=req.query):
        rq = rewrite_query(req.query, mode=req.rewrite, tracer=S.tracer)
        retriever = rag.get_retriever()
        hits = retriever.search_multi(rq.queries, top_k=req.top_k, keywords=rq.keywords, rerank_query=rq.rerank_query)
    return SearchResponse(rag_mode=retriever.mode, rewrite=rq.to_dict(), hits=hits,
                          latency_ms=round((time.perf_counter() - t0) * 1000, 1))


@app.post("/admin/reset-data")
def reset_data(x_admin_token: str | None = Header(None)):
    token = os.environ.get("ADMIN_TOKEN", "")
    if not token or not x_admin_token or not hmac.compare_digest(token, x_admin_token):
        raise HTTPException(403, "cần header X-Admin-Token khớp ADMIN_TOKEN")
    if config.API_TOOLS == "mcp":
        raise HTTPException(409, "API_TOOLS=mcp: dữ liệu nằm trong process MCP server, restart server để reset")
    tools.reset()
    log.warning("business data reset")
    return {"ok": True}


def main():
    import uvicorn

    # log_config=None: uvicorn không tự cài handler -> log của uvicorn đi qua formatter JSON/text của mình
    uvicorn.run(app, host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "8000")),
                log_config=None, access_log=False)


if __name__ == "__main__":
    main()
