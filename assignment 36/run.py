"""Chạy hệ thống Researcher → Writer → Reviewer (→ Writer ...) do Orchestrator điều phối.

    python run.py "Tình hình xe điện ở Đông Nam Á năm 2025"
    python run.py "State of small language models" --language English --max-revisions 1
    python run.py --draw

Kết quả: output/<timestamp>/
    report.md          báo cáo cuối
    a2a_log.jsonl      mọi A2A envelope (request message/send + response Task)
    agent_cards.json   Agent Card của 3 agent
    reviews.json       các vòng review (điểm, issues)
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401
from react_agent.llm import load_dotenv

from multi_agent.a2a import summarize_envelope
from multi_agent.agents import build_agents
from multi_agent.graph import MAX_REVISIONS, build_graph, initial_state

ROOT = Path(__file__).resolve().parent


def configure_console() -> None:
    """Use UTF-8 when the program is run from a legacy Windows console."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def main():
    configure_console()
    p = argparse.ArgumentParser()
    p.add_argument("topic", nargs="?")
    p.add_argument("--language", default="Vietnamese")
    p.add_argument("--max-revisions", type=int, default=MAX_REVISIONS)
    p.add_argument("--draw", action="store_true")
    args = p.parse_args()

    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT.parent / "assignment 33" / ".env")
    agents = build_agents()
    graph = build_graph(agents, args.max_revisions)

    if args.draw:
        (ROOT / "graph.mmd").write_text(graph.get_graph().draw_mermaid(), encoding="utf-8")
        print("Đã ghi graph.mmd")
        if not args.topic:
            return
    if not args.topic:
        p.error("Cần chủ đề (topic)")

    config = {"configurable": {"thread_id": uuid.uuid4().hex}, "recursion_limit": 50}
    print(f"Chủ đề: {args.topic}\n")
    final = graph.invoke(initial_state(args.topic, args.language), config)

    out = ROOT / "output" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(final["final_report"] or "", encoding="utf-8")
    with open(out / "a2a_log.jsonl", "w", encoding="utf-8") as f:
        for env in final["mailbox"]:
            f.write(json.dumps(env, ensure_ascii=False) + "\n")
    (out / "agent_cards.json").write_text(
        json.dumps([a.card.to_dict() for a in agents.values()], ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "reviews.json").write_text(json.dumps(final["reviews"], ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Transcript A2A ===")
    for env in final["mailbox"]:
        print(" ", summarize_envelope(env))
    print("\n=== Vòng review ===")
    for r in final["reviews"]:
        majors = sum(i["severity"] == "major" for i in r["issues"])
        print(f"  v{r['version']}: approved={r['approved']} major={majors} minor={len(r['issues']) - majors}")
    print("\n=== Token theo agent ===")
    for name, a in agents.items():
        u = a.llm.usage
        print(f"  {name:10s} {u.calls} calls, in {u.input_tokens:,} / out {u.output_tokens:,}")
    print(f"\nTrạng thái: {final['status']}" + (f" ({final['error']})" if final.get("error") else ""))
    print(f"Đã lưu: {out}")


if __name__ == "__main__":
    main()
