"""Hỏi agent một câu (hoặc chế độ tương tác).

    python ask.py "Bây giờ ở London là mấy giờ? Cộng thêm 7 tiếng thì là mấy giờ?"
    python ask.py -i
"""
from __future__ import annotations

import argparse
from pathlib import Path

from react_agent.agent import ReActAgent
from react_agent.llm import ClaudeClient, load_dotenv

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("question", nargs="?")
    p.add_argument("--model", default=None)
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--interactive", "-i", action="store_true")
    args = p.parse_args()

    load_dotenv(ROOT / ".env")
    agent = ReActAgent(ClaudeClient(args.model), max_steps=args.max_steps)

    def run(q: str):
        res = agent.run(q)
        print(f"\n>>> {res.answer}\n[{res.status}, {len(res.steps)} bước, "
              f"{res.input_tokens}+{res.output_tokens} tokens, {res.latency_ms / 1000:.1f}s]\n")

    if args.interactive:
        while q := input("Question (Enter để thoát): ").strip():
            run(q)
    elif args.question:
        run(args.question)
    else:
        p.error("Cần truyền câu hỏi hoặc --interactive")


if __name__ == "__main__":
    main()
