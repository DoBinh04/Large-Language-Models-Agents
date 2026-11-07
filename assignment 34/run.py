"""Chạy agent LangGraph với human-in-the-loop.

    python run.py "Tính 15% của 2.400.000 rồi ghi kết quả vào file ketqua.txt"
    python run.py "Gửi email cho boss@example.com tóm tắt giờ hiện tại ở Tokyo" --auto reject
    python run.py -i                 # hội thoại nhiều lượt trên cùng thread
    python run.py --draw             # in sơ đồ graph (Mermaid) ra graph.mmd

Khi agent muốn gọi write_file / send_email, graph dừng ở node human_approval
và hỏi bạn:  [y] duyệt  [n] từ chối (kèm lý do)  [e] sửa tham số JSON rồi duyệt.
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

import _bootstrap  # noqa: F401
from langgraph.types import Command

from lg_agent.graph import build_graph, new_turn, pending_interrupts
from react_agent.llm import load_dotenv

ROOT = Path(__file__).resolve().parent


def ask_human(request: dict, auto: str | None) -> dict:
    decisions = {}
    for call in request["tool_calls"]:
        print(f"\n⚠️  Agent muốn chạy: {call['preview']}")
        if auto:
            decisions[call["id"]] = {"approved": auto == "approve", "feedback": f"auto-{auto}"}
            print(f"   -> {auto} (tự động)")
            continue
        while True:
            ans = input("   Duyệt? [y]es / [n]o / [e]dit: ").strip().lower()
            if ans in ("y", "yes"):
                decisions[call["id"]] = {"approved": True}
            elif ans in ("n", "no"):
                decisions[call["id"]] = {"approved": False, "feedback": input("   Lý do: ").strip()}
            elif ans in ("e", "edit"):
                raw = input(f"   JSON tham số mới (hiện tại {json.dumps(call['input'], ensure_ascii=False)}): ")
                try:
                    decisions[call["id"]] = {"approved": True, "input": json.loads(raw)}
                except json.JSONDecodeError as e:
                    print(f"   JSON lỗi: {e}")
                    continue
            else:
                continue
            break
    return decisions


def print_updates(chunk: dict) -> None:
    for node, update in chunk.items():
        if node == "__interrupt__":
            print("⏸  graph tạm dừng (interrupt), state đã lưu vào checkpointer")
            continue
        if not update:
            continue
        for msg in update.get("messages", []):
            for block in msg["content"] if isinstance(msg["content"], list) else []:
                if block["type"] == "tool_use":
                    print(f"[{node}] tool_use → {block['name']} {json.dumps(block['input'], ensure_ascii=False)[:200]}")
                elif block["type"] == "tool_result":
                    tag = "ERROR" if block.get("is_error") else "ok"
                    print(f"[{node}] tool_result ({tag}): {str(block['content'])[:200]}")
                elif block["type"] == "text" and block.get("text"):
                    print(f"[{node}] {block['text'][:300]}")


def run_turn(graph, config, question: str, auto: str | None) -> str | None:
    payload = new_turn(question)
    while True:
        for chunk in graph.stream(payload, config, stream_mode="updates"):
            print_updates(chunk)
        interrupts = pending_interrupts(graph, config)
        if not interrupts:
            break
        # Resume: node human_approval chạy lại, interrupt() trả về giá trị này
        payload = Command(resume=ask_human(interrupts[0].value, auto))
    return graph.get_state(config).values.get("final_answer")


def main():
    # Một số Windows terminal cũ dùng CP1252.  Dùng UTF-8 để thông báo tiếng
    # Việt không làm CLI dừng sau khi graph đã chạy xong (ví dụ với --draw).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("question", nargs="?")
    p.add_argument("--interactive", "-i", action="store_true")
    p.add_argument("--auto", choices=["approve", "reject"], help="tự động duyệt/từ chối (để test)")
    p.add_argument("--thread", default=None, help="thread_id (mặc định: ngẫu nhiên)")
    p.add_argument("--draw", action="store_true")
    args = p.parse_args()

    load_dotenv(ROOT / ".env")
    graph = build_graph()
    if args.draw:
        (ROOT / "graph.mmd").write_text(graph.get_graph().draw_mermaid(), encoding="utf-8")
        print("Đã ghi graph.mmd (dán vào https://mermaid.live để xem)")
        if not (args.question or args.interactive):
            return

    config = {"configurable": {"thread_id": args.thread or str(uuid.uuid4())},
              "recursion_limit": 60}
    if args.interactive:
        while q := input("\nBạn: ").strip():
            print(f"\nAgent: {run_turn(graph, config, q, args.auto)}")
    elif args.question:
        print(f"\nAgent: {run_turn(graph, config, args.question, args.auto)}")
    else:
        p.error("Cần câu hỏi, --interactive hoặc --draw")


if __name__ == "__main__":
    main()
