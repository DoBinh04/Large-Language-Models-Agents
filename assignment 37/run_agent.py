"""Chạy ReAct agent (3.3) có tracing trên 10 câu test.

    python run_agent.py
    python run_agent.py --ids T02 T04 --model claude-haiku-5-5
    python run_agent.py -q "Bây giờ ở Paris mấy giờ?"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap
from react_agent.llm import ClaudeClient, load_dotenv

from observability import get_tracer
from observability.instrument_agent import TracedReActAgent

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-q", "--question", action="append")
    p.add_argument("--ids", nargs="*")
    p.add_argument("--model", default=None)
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--no-langfuse", action="store_true")
    args = p.parse_args()

    load_dotenv(ROOT / ".env")
    load_dotenv(_bootstrap.AGENT_DIR / ".env")
    tracer = get_tracer(ROOT / "logs", use_langfuse=not args.no_langfuse)
    print(f"Langfuse: {'BẬT' if tracer.langfuse.enabled else 'TẮT (chỉ log JSONL)'} | log: {tracer.log_file}")

    agent = TracedReActAgent(tracer, ClaudeClient(args.model), max_steps=args.max_steps, verbose=False)
    if args.question:
        questions = [(None, q) for q in args.question]
    else:
        qs = json.loads((_bootstrap.AGENT_DIR / "data" / "test_questions.json").read_text(encoding="utf-8"))
        questions = [(q["id"], q["question"]) for q in qs if not args.ids or q["id"] in args.ids]

    session = f"agent-{tracer.run_id}"
    for qid, q in questions:
        try:
            res = agent.run(q, metadata={"question_id": qid, "session": session})
            print(f"[{qid}] {res.status}, {len(res.steps)} bước, tools={res.tools_used}\n   -> {str(res.answer)[:200]!r}\n")
        except Exception as e:
            print(f"[{qid}] LỖI {type(e).__name__}: {e}\n")
    tracer.flush()
    print("Xem dashboard:  python dashboard.py")


if __name__ == "__main__":
    main()
