"""Chat thử với hệ thống multi-agent.

    python run_chat.py                                  # router API (Opus 5.5 @ medium), tools in-process, A2A in-process
    python run_chat.py --router rule                    # router từ khoá (không tốn API cho bước router)
    python run_chat.py --router dpo                     # router Small LLM + adapter DPO (cần torch/peft)
    python run_chat.py --tools mcp --a2a http           # tool qua MCP server, agent là A2A HTTP service
    python run_chat.py --model claude-haiku-5-5 --effort low

Lệnh trong chat: /state  /cards  /reset  /quit
Mỗi lượt in: route, tool calls, latency, tokens, chi phí. Trace JSONL ở results/traces/.
"""
from __future__ import annotations

import argparse
import json
import os

import config
from support import a2a
from support.logging_setup import setup_logging
from support.orchestrator import Orchestrator
from support.router import make_router
from support.tool_backend import make_backend
from support.tracing import Tracer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--router", default="api", choices=["api", "rule", "small", "sft", "dpo"])
    ap.add_argument("--model", default=config.BASE_MODEL, help="model cho router API + agents")
    ap.add_argument("--effort", default=config.BASE_EFFORT, choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--tools", default="local", choices=["local", "mcp"])
    ap.add_argument("--a2a", default="inproc", choices=["inproc", "http"])
    ap.add_argument("--rag-light", action="store_true", help="chỉ BM25, không tải embedding / reranker")
    ap.add_argument("--rewrite", default=None, choices=["off", "rules", "llm"], help="query rewrite cho RAG")
    a = ap.parse_args()
    if a.rag_light:
        os.environ["RAG_LIGHT"] = "1"
    if a.rewrite:
        os.environ["QUERY_REWRITE"] = a.rewrite
    setup_logging(default_level="WARNING", force=True)          # LOG_LEVEL=INFO để xem log từng bước

    tracer = Tracer(run_name=f"chat-{a.router}")
    router = make_router(a.router, tracer, model=a.model, effort=a.effort)
    orch = Orchestrator(router, make_backend(a.tools), tracer, agent_model=a.model, agent_effort=a.effort)
    if a.a2a == "http":
        urls = {}
        for i, (name, agent) in enumerate(orch.agents.items()):
            a2a.serve_agent(agent, 9101 + i)
            urls[name] = agent.card.url
        orch.a2a.transport = a2a.HttpTransport(urls)
        print("A2A agents:", urls)

    print(f"Router={router.label} | agents={a.model}@{a.effort} | tools={a.tools} | a2a={a.a2a}")
    print("Gõ /quit để thoát.\n")
    while True:
        try:
            msg = input("Bạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not msg:
            continue
        if msg == "/quit":
            break
        if msg == "/reset":
            orch.reset()
            print("(đã reset hội thoại)\n")
            continue
        if msg == "/state":
            print(json.dumps(orch.state.to_dict(), ensure_ascii=False, indent=1), "\n")
            continue
        if msg == "/cards":
            print(json.dumps(orch.agent_cards(), ensure_ascii=False, indent=1), "\n")
            continue
        res = orch.chat(msg)
        s = tracer.summary(res.trace_id)
        d = res.decision
        print(f"\nTrợ lý: {res.reply}\n")
        print(f"  ↳ intent={d['intent']} agents={d['agents']} missing={d['missing_info']} | "
              f"tools={[c['name'] for c in res.tool_calls]}")
        print(f"  ↳ {s['latency_ms'] / 1000:.1f}s | {s['llm_calls']} LLM calls | in {s['input_tokens']} "
              f"out {s['output_tokens']} cache {s['cache_read_tokens']} | ${s['cost_usd']:.4f}\n")
    orch.backend.close()
    print(f"trace: {tracer.path}")


if __name__ == "__main__":
    main()
