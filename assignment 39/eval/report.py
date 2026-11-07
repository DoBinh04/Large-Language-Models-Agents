"""Gộp kết quả thành bảng so sánh (Markdown) + dashboard HTML tĩnh.

    python -m eval.report
-> results/comparison.md   : bảng E2E (success, tool F1, routing, latency, cost) + bảng routing-only
-> results/dashboard.html  : biểu đồ cost vs success, latency, và waterfall trace của một run (span tree)
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

BASELINE = "api-opus-medium"


def load_e2e() -> list[dict]:
    out = []
    for p in sorted((config.RESULTS_DIR / "eval").glob("*/summary.json")):
        out.append(json.loads(p.read_text(encoding="utf-8")))
    return out


def load_routing() -> list[dict]:
    out = []
    for p in sorted((config.RESULTS_DIR / "routing").glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        out.append({"router": d["router"], **d["summary"]})
    return out


def _pct(x):
    return "-" if x is None else f"{x * 100:.1f}%"


def _delta(x, base):
    if x is None or base in (None, 0):
        return ""
    return f" ({(x / base - 1) * 100:+.0f}%)"


def markdown(e2e: list[dict], routing: list[dict]) -> str:
    base = next((s for s in e2e if s["config"] == BASELINE), None)
    lines = ["# So sánh cấu hình\n", f"Baseline: `{BASELINE}` (Claude Opus 5.5, effort medium — thay cho gpt-oss-120b medium).\n",
             "## End-to-end (data/eval_set.json)\n",
             "| Config | Router | Agents | Task success (Judge) | Tool F1 micro (P / R) | Tool F1 macro | Routing acc | "
             "Latency mean / p95 (s) | Router (ms) | Cost / kịch bản ($) | Cost / PASS ($) | LLM calls / lượt |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in sorted(e2e, key=lambda s: -(s["task_success_rate"] or 0)):
        f = s["tool_f1_micro"]
        router = s["router"] + (f" ({s['router_llm'][0]}@{s['router_llm'][1]})" if s.get("router_llm") else "")
        agents = f"{s['agents'][0]}@{s['agents'][1]}" + (f" + {s['overrides']}" if s.get("overrides") else "")
        bc = base["cost_per_scenario_usd"] if base else None
        bl = base["latency_ms_mean"] if base else None
        lines.append(
            f"| {s['config']} | {router} | {agents} | {_pct(s['task_success_rate'])} | {f['f1']:.3f} ({f['precision']:.2f} / {f['recall']:.2f}) "
            f"| {s['tool_f1_macro']:.3f} | {_pct(s['routing_acc'])} | {s['latency_ms_mean'] / 1000:.1f}{_delta(s['latency_ms_mean'], bl)} / "
            f"{s['latency_ms_p95'] / 1000:.1f} | {s['router_ms_mean']:.0f} | {s['cost_per_scenario_usd']:.4f}{_delta(s['cost_per_scenario_usd'], bc)} "
            f"| {s['cost_per_success_usd'] if s['cost_per_success_usd'] is not None else '-'} | {s['llm_calls_per_turn']} |")
    if e2e:
        cats = sorted({c for s in e2e for c in s["by_category"]})
        lines += ["\n### Task success theo nhóm kịch bản\n", "| Config | " + " | ".join(cats) + " |",
                  "|---|" + "---|" * len(cats)]
        for s in e2e:
            lines.append(f"| {s['config']} | " + " | ".join(_pct(s["by_category"].get(c, {}).get("success")) for c in cats) + " |")
    if routing:
        lines += ["\n## Routing-only (API LLM vs Small LLM vs SFT vs DPO)\n",
                  "| Router | Routing acc | Intent acc | Acc eval set | Acc test | JSON valid | Latency mean / p95 (ms) | $ / 1k quyết định |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in routing:
            lines.append(f"| {r['router']} | {_pct(r['routing_acc'])} | {_pct(r['intent_acc'])} | {_pct(r['routing_acc_eval_set'])} "
                         f"| {_pct(r['routing_acc_test'])} | {_pct(r['json_valid'])} | {r['latency_ms_mean']:.0f} / {r['latency_ms_p95']:.0f} "
                         f"| {r['cost_per_1k_usd']:.4f} |")
        kinds = sorted({k for r in routing for k in r["by_kind"]})
        lines += ["\n### Routing acc theo nhóm (state-aware: pending / in_state / follow_product / switch)\n",
                  "| Router | " + " | ".join(kinds) + " |", "|---|" + "---|" * len(kinds)]
        for r in routing:
            lines.append(f"| {r['router']} | " + " | ".join(_pct(r["by_kind"].get(k, {}).get("acc")) for k in kinds) + " |")
    return "\n".join(lines) + "\n"


def _bars(items: list[tuple[str, float]], fmt: str, color: str) -> str:
    mx = max((v for _, v in items), default=1) or 1
    rows = "".join(f'<div class="row"><span class="lbl">{html.escape(k)}</span><span class="bar" style="width:{v / mx * 100:.1f}%;'
                   f'background:{color}"></span><span class="val">{fmt.format(v)}</span></div>' for k, v in items)
    return f'<div class="bars">{rows}</div>'


def _waterfall(trace_file: Path, max_traces: int = 6) -> str:
    spans = [json.loads(l) for l in trace_file.open(encoding="utf-8")]
    by_trace: dict[str, list[dict]] = {}
    for s in spans:
        by_trace.setdefault(s["trace_id"], []).append(s)
    out = []
    for tid, ss in list(by_trace.items())[:max_traces]:
        root = next((s for s in ss if s["parent_id"] is None), ss[-1])
        t0, total = root["start"], max(root["latency_ms"], 1)
        children: dict[str | None, list[dict]] = {}
        for s in ss:
            children.setdefault(s["parent_id"], []).append(s)

        def walk(s, depth):
            left = (s["start"] - t0) * 1000 / total * 100
            width = max(s["latency_ms"] / total * 100, 0.5)
            a = s["attrs"]
            info = f"{s['latency_ms']:.0f}ms"
            if s["kind"] == "llm":
                info += f" · {a.get('model')}@{a.get('effort')} · in {a.get('input_tokens')} out {a.get('output_tokens')} cache {a.get('cache_read_input_tokens')} · ${a.get('cost_usd', 0):.4f}"
            if s["kind"] == "tool":
                info += f" · {html.escape(json.dumps(a.get('args'), ensure_ascii=False))[:120]}"
            rows = [f'<div class="span k-{s["kind"]}"><span class="name" style="padding-left:{depth * 14}px">{html.escape(s["name"])}</span>'
                    f'<span class="track"><span class="seg" style="left:{left:.1f}%;width:{width:.1f}%"></span></span>'
                    f'<span class="info">{info}</span></div>']
            for c in sorted(children.get(s["span_id"], []), key=lambda x: x["start"]):
                rows += walk(c, depth + 1)
            return rows

        title = html.escape(str(root["attrs"].get("user", root["name"])))[:140]
        out.append(f"<h3>{title}</h3>" + "".join(walk(root, 0)))
    return "".join(out)


def dashboard(e2e: list[dict], routing: list[dict]) -> str:
    sec = []
    if e2e:
        sec.append("<h2>Task success rate (LLM Judge)</h2>" + _bars([(s["config"], (s["task_success_rate"] or 0) * 100) for s in e2e], "{:.1f}%", "var(--good)"))
        sec.append("<h2>Tool-call F1 (micro)</h2>" + _bars([(s["config"], s["tool_f1_micro"]["f1"]) for s in e2e], "{:.3f}", "var(--accent)"))
        sec.append("<h2>Chi phí / kịch bản (USD)</h2>" + _bars([(s["config"], s["cost_per_scenario_usd"]) for s in e2e], "${:.4f}", "var(--warn)"))
        sec.append("<h2>Latency trung bình / lượt (s)</h2>" + _bars([(s["config"], s["latency_ms_mean"] / 1000) for s in e2e], "{:.1f}s", "var(--muted)"))
    if routing:
        sec.append("<h2>Routing accuracy (routing-only)</h2>" + _bars([(r["router"], r["routing_acc"] * 100) for r in routing], "{:.1f}%", "var(--accent)"))
    traces = sorted(config.TRACE_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime) if config.TRACE_DIR.exists() else []
    traces = [t for t in traces if not t.name.startswith("judge-")]
    if traces:
        sec.append(f"<h2>Trace waterfall: {html.escape(traces[-1].name)}</h2>" + _waterfall(traces[-1]))
    return f"""<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Support Agents Dashboard</title><style>
:root{{--bg:#fafaf9;--fg:#1c1917;--muted:#78716c;--line:#e7e5e4;--accent:#2563eb;--good:#16a34a;--warn:#d97706;--card:#fff}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1c1917;--fg:#f5f5f4;--muted:#a8a29e;--line:#44403c;--accent:#60a5fa;--good:#4ade80;--warn:#fbbf24;--card:#292524}}}}
body{{margin:0;padding:24px 16px;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;max-width:1100px;margin:auto}}
h2{{font-size:16px;margin:28px 0 8px}} h3{{font-size:13px;margin:18px 0 4px;color:var(--muted);font-weight:500}}
.row{{display:grid;grid-template-columns:180px 1fr 90px;gap:8px;align-items:center;margin:3px 0}}
.lbl{{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}} .bar{{height:14px;border-radius:3px;display:block}} .val{{text-align:right;font-variant-numeric:tabular-nums}}
.span{{display:grid;grid-template-columns:220px 1fr 420px;gap:8px;font-size:12px;align-items:center;border-bottom:1px solid var(--line)}}
.track{{position:relative;height:10px}} .seg{{position:absolute;top:0;height:10px;border-radius:2px;background:var(--muted)}}
.k-llm .seg{{background:var(--accent)}} .k-tool .seg{{background:var(--good)}} .k-router .seg{{background:var(--warn)}}
.info{{color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
@media (max-width:700px){{.span{{grid-template-columns:1fr}} .row{{grid-template-columns:110px 1fr 70px}}}}
</style></head><body><h1 style="font-size:20px">TechViet Support Agents: LLMOps</h1>{''.join(sec) or '<p>Chưa có kết quả.</p>'}</body></html>"""


def main():
    e2e, routing = load_e2e(), load_routing()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    md = markdown(e2e, routing)
    (config.RESULTS_DIR / "comparison.md").write_text(md, encoding="utf-8")
    (config.RESULTS_DIR / "dashboard.html").write_text(dashboard(e2e, routing), encoding="utf-8")
    print(md)
    print(f"-> {config.RESULTS_DIR / 'comparison.md'}\n-> {config.RESULTS_DIR / 'dashboard.html'}")


if __name__ == "__main__":
    main()
