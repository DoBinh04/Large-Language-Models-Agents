"""Chạy agent trên 10 câu hỏi multi-step và lưu kết quả.

    python run_tests.py                     # cả 10 câu
    python run_tests.py --ids T02 T07       # chỉ vài câu
    python run_tests.py --model claude-haiku-5-5 --max-steps 8

Kết quả: results/<timestamp>/
    summary.md     bảng tổng hợp (pass/fail, số bước, tool đã dùng, tokens, latency)
    traces.jsonl   toàn bộ Thought / Action / Observation của từng câu
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from react_agent.agent import ReActAgent
from react_agent.checks import check_answer
from react_agent.llm import ClaudeClient, load_dotenv

ROOT = Path(__file__).resolve().parent


def tool_coverage(expected: list[str], used: list[str]) -> bool:
    """Agent có dùng đủ các LOẠI tool mà lời giải cần không (không bắt đúng thứ tự/số lần)."""
    return set(expected) <= set(used)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--questions", default=str(ROOT / "data" / "test_questions.json"))
    p.add_argument("--ids", nargs="*")
    p.add_argument("--model", default=None)
    p.add_argument("--effort", default=None)
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    load_dotenv(ROOT / ".env")
    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    if args.ids:
        questions = [q for q in questions if q["id"] in set(args.ids)]

    agent = ReActAgent(ClaudeClient(args.model, args.effort), max_steps=args.max_steps, verbose=not args.quiet)
    out_dir = ROOT / "results" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(out_dir / "traces.jsonl", "w", encoding="utf-8") as f:
        for q in questions:
            try:
                res = agent.run(q["question"])
            except Exception as e:  # lỗi API, mạng... -> ghi lại và chạy câu tiếp theo
                print(f"!! {q['id']} lỗi: {type(e).__name__}: {e}")
                rows.append({"id": q["id"], "status": f"exception: {type(e).__name__}", "passed": False,
                             "why": str(e)[:80], "steps": 0, "tools": [], "coverage": False,
                             "tokens_in": 0, "tokens_out": 0, "latency_s": 0, "answer": ""})
                continue
            passed, why = check_answer(res.answer, q.get("check", {}))
            row = {
                "id": q["id"], "status": res.status, "passed": passed, "why": why,
                "steps": len(res.steps), "tools": res.tools_used,
                "coverage": tool_coverage(q.get("expected_tools", []), res.tools_used),
                "tokens_in": res.input_tokens, "tokens_out": res.output_tokens,
                "latency_s": round(res.latency_ms / 1000, 1), "answer": res.answer or "",
            }
            rows.append(row)
            f.write(json.dumps({"id": q["id"], "question": q["question"], "check": q.get("check"),
                                "passed": passed, **res.to_dict()}, ensure_ascii=False) + "\n")
            f.flush()

    write_summary(out_dir / "summary.md", rows, agent.llm.model)
    print((out_dir / "summary.md").read_text(encoding="utf-8"))
    print(f"\nĐã lưu: {out_dir}")


def write_summary(path: Path, rows: list[dict], model: str) -> None:
    auto = [r for r in rows if r["passed"] is not None]
    n_pass = sum(1 for r in auto if r["passed"])
    multi = sum(1 for r in rows if len(r["tools"]) >= 2)
    lines = [
        f"# ReAct agent — kết quả test ({model})", "",
        f"- Đúng (chấm tự động): **{n_pass}/{len(auto)}**",
        f"- Dùng đủ loại tool cần thiết: {sum(r['coverage'] for r in rows)}/{len(rows)}",
        f"- Câu thực sự multi-step (≥ 2 lần gọi tool): {multi}/{len(rows)}",
        f"- Trung bình số bước: {sum(r['steps'] for r in rows) / max(1, len(rows)):.1f}",
        f"- Tổng tokens: in {sum(r['tokens_in'] for r in rows):,} / out {sum(r['tokens_out'] for r in rows):,}",
        "",
        "| ID | Status | Pass | Steps | Tools | Coverage | Tokens in/out | Latency (s) | Check |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        mark = {True: "✅", False: "❌", None: "—"}[r["passed"]]
        lines.append(f"| {r['id']} | {r['status']} | {mark} | {r['steps']} | {' → '.join(r['tools'])} | "
                     f"{'✓' if r['coverage'] else '✗'} | {r['tokens_in']}/{r['tokens_out']} | "
                     f"{r['latency_s']} | {r['why']} |")
    lines += ["", "## Câu trả lời", ""]
    for r in rows:
        lines.append(f"**{r['id']}**: {r['answer'].strip()}\n")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
