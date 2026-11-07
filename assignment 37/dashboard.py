"""Dashboard đơn giản từ log JSONL: latency trung bình, tổng cost, top failed queries.

    python dashboard.py                 # đọc logs/*.jsonl -> in tóm tắt + ghi dashboard.html
    python dashboard.py --run 20261028-101500     # chỉ 1 lần chạy
    python dashboard.py --since 2026-10-01

Langfuse cũng có dashboard sẵn (Home: latency, cost theo model, số trace; Traces: lọc theo tag
"rag"/"agent", level=WARNING/ERROR). File này cho thấy các con số đó được tính từ log thô như thế nào.
"""
from __future__ import annotations

import argparse
import html
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(log_dir: Path, run: str | None, since: str | None) -> list[dict]:
    rows = []
    for f in sorted(log_dir.glob("traces-*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if run and r.get("run_id") != run:
                continue
            if since and r.get("ts", "") < since:
                continue
            rows.append(r)
    return rows


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]


def stats(values: list[float]) -> dict:
    return {"n": len(values), "avg": statistics.fmean(values) if values else 0.0,
            "p50": pct(values, 0.5), "p95": pct(values, 0.95), "max": max(values) if values else 0.0}


def compute(rows: list[dict]) -> dict:
    traces = [r for r in rows if r["kind"] == "trace"]
    gens = [r for r in rows if r["kind"] == "generation"]
    retr = [r for r in rows if r["kind"] == "retrieval"]
    tools = [r for r in rows if r["kind"] == "tool"]

    by_trace = defaultdict(list)
    for t in traces:
        by_trace[t["name"]].append(t["latency_ms"])
    by_model = defaultdict(lambda: {"calls": 0, "in": 0, "out": 0, "cost": 0.0, "lat": [], "unknown_price": 0})
    for g in gens:
        m = by_model[g.get("model") or "?"]
        m["calls"] += 1
        m["in"] += g.get("input_tokens") or 0
        m["out"] += g.get("output_tokens") or 0
        if g.get("cost_usd") is None:
            m["unknown_price"] += 1
        m["cost"] += g.get("cost_usd") or 0.0
        m["lat"].append(g["latency_ms"])
    by_tool = defaultdict(lambda: {"calls": 0, "errors": 0, "lat": []})
    for t in tools:
        d = by_tool[t["name"]]
        d["calls"] += 1
        d["errors"] += int(bool(t.get("tool_error")) or t["status"] == "error")
        d["lat"].append(t["latency_ms"])

    # cost theo trace (cộng các generation con)
    cost_by_trace = defaultdict(float)
    for g in gens:
        cost_by_trace[g["trace_id"]] += g.get("cost_usd") or 0.0

    failed = [t for t in traces if t.get("failed") or t["status"] == "error"]
    fail_groups: dict[str, dict] = {}
    for t in failed:
        q = (t.get("query") or json.dumps(t.get("input"), ensure_ascii=False) or "?").strip()
        g = fail_groups.setdefault(q, {"query": q, "count": 0, "reasons": Counter(), "last": "", "type": t["name"]})
        g["count"] += 1
        g["reasons"][t.get("failure_reason") or t.get("error") or "error"] += 1
        g["last"] = max(g["last"], t["ts"])
    top_failed = sorted(fail_groups.values(), key=lambda g: (g["count"], g["last"]), reverse=True)[:15]

    top_scores = [r["retrieval"]["max_score"] for r in retr if r.get("retrieval")]
    return {
        "n_traces": len(traces), "n_failed": len(failed), "n_gen": len(gens),
        "total_cost": sum(g.get("cost_usd") or 0.0 for g in gens),
        "total_in": sum(g.get("input_tokens") or 0 for g in gens),
        "total_out": sum(g.get("output_tokens") or 0 for g in gens),
        "trace_latency": {k: stats(v) for k, v in by_trace.items()},
        "llm_latency": stats([g["latency_ms"] for g in gens]),
        "retrieval_latency": stats([r["latency_ms"] for r in retr]),
        "by_model": {k: {**v, "lat": stats(v["lat"])} for k, v in by_model.items()},
        "by_tool": {k: {**v, "lat": stats(v["lat"])} for k, v in by_tool.items()},
        "avg_cost_per_trace": (sum(cost_by_trace.values()) / len(traces)) if traces else 0.0,
        "retrieval_top1": stats(top_scores),
        "top_failed": top_failed,
        "runs": sorted({r["run_id"] for r in rows}),
        "prompt_versions": Counter(g.get("system_hash") for g in gens),
    }


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def print_summary(s: dict) -> None:
    print(f"Traces: {s['n_traces']} (failed {s['n_failed']}) | LLM calls: {s['n_gen']} | "
          f"tokens in/out: {s['total_in']:,}/{s['total_out']:,} | total cost: ${s['total_cost']:.4f} "
          f"(avg ${s['avg_cost_per_trace']:.5f}/trace)")
    for name, st in s["trace_latency"].items():
        print(f"  {name:12s} avg {st['avg']:8.0f} ms  p50 {st['p50']:8.0f}  p95 {st['p95']:8.0f}  (n={st['n']})")
    print(f"  LLM call     avg {s['llm_latency']['avg']:8.0f} ms  | retrieval avg {s['retrieval_latency']['avg']:.1f} ms")
    print("Top failed queries:")
    for g in s["top_failed"][:10]:
        reasons = ", ".join(f"{k}×{v}" for k, v in g["reasons"].items())
        print(f"  {g['count']}× [{g['type']}] {g['query'][:80]}  ({reasons})")


CSS = """
:root{--bg:#f7f7f5;--card:#fff;--fg:#1f2328;--muted:#6b7280;--line:#e5e7eb;--accent:#2563eb;--bad:#dc2626;--bar:#93c5fd}
@media (prefers-color-scheme:dark){:root{--bg:#111316;--card:#1b1e23;--fg:#e6e6e6;--muted:#9aa0a6;--line:#2c3036;--accent:#60a5fa;--bad:#f87171;--bar:#1d4ed8}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px}h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 10px}
.muted{color:var(--muted)}.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-top:16px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}.kpi b{display:block;font-size:22px;font-variant-numeric:tabular-nums}
.kpi span{color:var(--muted);font-size:12px}table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}
th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{font-size:12px;color:var(--muted);font-weight:600}
td.n{text-align:right;font-variant-numeric:tabular-nums}.bad{color:var(--bad)}.bar{height:8px;background:var(--bar);border-radius:4px}
.wrap{overflow-x:auto}
"""


def _row(cells, numeric=()):
    return "<tr>" + "".join(f"<td class='n'>{c}</td>" if i in numeric else f"<td>{c}</td>"
                            for i, c in enumerate(cells)) + "</tr>"


def write_html(s: dict, path: Path) -> None:
    e = html.escape
    kpis = [
        ("Traces", f"{s['n_traces']}"), ("Failed", f"<span class='bad'>{s['n_failed']}</span>"),
        ("LLM calls", f"{s['n_gen']}"), ("Total cost", f"${s['total_cost']:.4f}"),
        ("Avg cost / trace", f"${s['avg_cost_per_trace']:.5f}"),
        ("Avg LLM latency", f"{s['llm_latency']['avg']:.0f} ms"),
        ("Tokens in / out", f"{s['total_in']:,} / {s['total_out']:,}"),
        ("Avg top-1 similarity", f"{s['retrieval_top1']['avg']:.3f}"),
    ]
    max_avg = max([v["avg"] for v in s["trace_latency"].values()] or [1]) or 1
    lat_rows = "".join(_row([e(k), f"{v['n']}", f"{v['avg']:.0f}", f"{v['p50']:.0f}", f"{v['p95']:.0f}",
                             f"<div class='bar' style='width:{100 * v['avg'] / max_avg:.0f}%'></div>"], (1, 2, 3, 4))
                       for k, v in s["trace_latency"].items())
    model_rows = "".join(_row([e(k), v["calls"], f"{v['in']:,}", f"{v['out']:,}", f"{v['lat']['avg']:.0f}",
                               f"${v['cost']:.4f}" + (" <span class='muted'>(giá ?)</span>" if v["unknown_price"] else "")],
                              (1, 2, 3, 4, 5)) for k, v in s["by_model"].items())
    tool_rows = "".join(_row([e(k), v["calls"], v["errors"], f"{v['lat']['avg']:.0f}"], (1, 2, 3))
                        for k, v in s["by_tool"].items()) or "<tr><td colspan=4 class='muted'>(chưa có)</td></tr>"
    fail_rows = "".join(_row([g["count"], e(g["type"]), e(g["query"]),
                              e(", ".join(f"{k} ×{v}" for k, v in g["reasons"].items())), e(g["last"][:19])], (0,))
                        for g in s["top_failed"]) or "<tr><td colspan=5 class='muted'>Không có query lỗi 🎉</td></tr>"
    doc = f"""<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LLM Observability</title><style>{CSS}</style></head><body><main>
<h1>LLM Observability: RAG + Agent</h1>
<div class="muted">{len(s['runs'])} lần chạy: {e(', '.join(s['runs'][-5:]))}{' …' if len(s['runs']) > 5 else ''}</div>
<div class="kpis">{''.join(f"<div class='kpi'><span>{k}</span><b>{v}</b></div>" for k, v in kpis)}</div>
<h2>Latency theo loại trace (ms)</h2><div class="wrap"><table><tr><th>Trace</th><th>n</th><th>avg</th><th>p50</th><th>p95</th><th style="width:30%"></th></tr>{lat_rows}</table></div>
<h2>LLM calls theo model</h2><div class="wrap"><table><tr><th>Model</th><th>Calls</th><th>Tokens in</th><th>Tokens out</th><th>Avg latency (ms)</th><th>Cost</th></tr>{model_rows}</table></div>
<h2>Tools (agent)</h2><div class="wrap"><table><tr><th>Tool</th><th>Calls</th><th>Errors</th><th>Avg latency (ms)</th></tr>{tool_rows}</table></div>
<h2>Top failed queries</h2><div class="wrap"><table><tr><th>#</th><th>Loại</th><th>Query</th><th>Lý do</th><th>Lần cuối (UTC)</th></tr>{fail_rows}</table></div>
<p class="muted">Retrieval: {s['retrieval_latency']['n']} lần, avg {s['retrieval_latency']['avg']:.1f} ms, top-1 similarity avg {s['retrieval_top1']['avg']:.3f} / p50 {s['retrieval_top1']['p50']:.3f}.
Prompt versions (system hash): {e(', '.join(f'{k}×{v}' for k, v in s['prompt_versions'].items()))}</p>
</main></body></html>"""
    path.write_text(doc, encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", default=str(ROOT / "logs"))
    p.add_argument("--run", default=None)
    p.add_argument("--since", default=None, help="ISO date, vd 2026-10-01")
    p.add_argument("--out", default=str(ROOT / "dashboard.html"))
    args = p.parse_args()

    rows = load(Path(args.logs), args.run, args.since)
    if not rows:
        print("Chưa có log. Chạy run_rag.py hoặc run_agent.py trước.")
        return
    s = compute(rows)
    print_summary(s)
    write_html(s, Path(args.out))
    # Console Windows có thể đang dùng CP1252 và không in được tiếng Việt.
    # Dùng ASCII ở thông báo cuối để dashboard vẫn kết thúc thành công sau khi
    # đã tạo file HTML UTF-8.
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
