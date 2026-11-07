"""End-to-end eval hệ thống multi-agent trên data/eval_set.json.

Mỗi cấu hình (router + model/effort của agent) chạy toàn bộ kịch bản nhiều lượt, đo:
  - Task success rate  : LLM Judge PASS / FAIL theo tiêu chí từng kịch bản
  - Tool-call F1       : micro / macro, kèm precision (gọi thừa) & recall (bỏ bước)
  - Routing accuracy   : từng lượt, so với nhãn agents + hỏi lại
  - Latency            : mỗi lượt (mean / p50 / p95) và riêng bước router
  - Cost               : USD theo token thực tế (có tính cache read/write), / kịch bản, / kịch bản PASS
Chi phí của Judge tính riêng, không cộng vào chi phí hệ thống.

    python -m eval.run_eval --configs api-opus-medium                 # baseline
    python -m eval.run_eval --configs api-opus-low api-sonnet-medium api-haiku-medium
    python -m eval.run_eval --configs small-dpo-opus --ids S07 S13    # cần GPU cho small router
    python -m eval.run_eval --configs api-opus-medium --tools mcp      # tool đi qua MCP server
Sau đó: python -m eval.report
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from eval.judge import Judge  # noqa: E402
from eval.metrics import micro_f1, percentile, routing_correct, tool_f1  # noqa: E402
from support.orchestrator import Orchestrator  # noqa: E402
from support.router import make_router  # noqa: E402
from support.tool_backend import make_backend  # noqa: E402
from support.tracing import Tracer  # noqa: E402

OPUS, SONNET, HAIKU = "claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"

# router: api | rule | small | sft | dpo. agents: (model, effort) mặc định cho cả 3 agent.
CONFIGS = {
    # Baseline của đề (gpt-oss-120b @ medium) -> Claude Opus 5.5 @ medium
    "api-opus-medium":   {"router": "api", "router_llm": (OPUS, "medium"), "agents": (OPUS, "medium")},
    "api-opus-low":      {"router": "api", "router_llm": (OPUS, "low"), "agents": (OPUS, "low")},
    "api-opus-high":     {"router": "api", "router_llm": (OPUS, "high"), "agents": (OPUS, "high")},
    "api-sonnet-medium": {"router": "api", "router_llm": (SONNET, "medium"), "agents": (SONNET, "medium")},
    "api-sonnet-low":    {"router": "api", "router_llm": (SONNET, "low"), "agents": (SONNET, "low")},
    "api-haiku-medium":  {"router": "api", "router_llm": (HAIKU, "medium"), "agents": (HAIKU, "medium")},
    "api-haiku-low":     {"router": "api", "router_llm": (HAIKU, "low"), "agents": (HAIKU, "low")},
    # Router rẻ (không gọi API) + agents giữ baseline -> cô lập ảnh hưởng của router
    "rule-opus":         {"router": "rule", "agents": (OPUS, "medium")},
    "small-base-opus":   {"router": "small", "agents": (OPUS, "medium")},
    "small-sft-opus":    {"router": "sft", "agents": (OPUS, "medium")},
    "small-dpo-opus":    {"router": "dpo", "agents": (OPUS, "medium")},
    # Cấu hình tối ưu chi phí: router DPO tự host, agent tra cứu đơn giản dùng Haiku,
    # Policy agent (suy luận điều kiện đổi trả) dùng Sonnet, composer Haiku.
    "opt-dpo-mixed":     {"router": "dpo", "agents": (HAIKU, "low"),
                          "overrides": {"policy": (SONNET, "low")}, "composer": (HAIKU, "low")},
    # Cùng bộ agent rẻ như opt-dpo-mixed, chỉ đổi router -> so Small base / SFT / DPO end-to-end với chi phí thấp
    "opt-small-mixed":   {"router": "small", "agents": (HAIKU, "low"),
                          "overrides": {"policy": (SONNET, "low")}, "composer": (HAIKU, "low")},
    "opt-sft-mixed":     {"router": "sft", "agents": (HAIKU, "low"),
                          "overrides": {"policy": (SONNET, "low")}, "composer": (HAIKU, "low")},
    "opt-api-mixed":     {"router": "api", "router_llm": (HAIKU, "low"), "agents": (HAIKU, "low"),
                          "overrides": {"policy": (SONNET, "low")}, "composer": (HAIKU, "low")},
}


def build(cfg_name: str, tracer: Tracer, tools_kind: str) -> Orchestrator:
    cfg = CONFIGS[cfg_name]
    rm, re_ = cfg.get("router_llm", (None, None))
    router = make_router(cfg["router"], tracer, model=rm, effort=re_)
    am, ae = cfg["agents"]
    return Orchestrator(router, make_backend(tools_kind), tracer, agent_model=am, agent_effort=ae,
                        composer=cfg.get("composer"), agent_overrides=cfg.get("overrides"))


def run_scenario(orch: Orchestrator, tracer: Tracer, scn: dict) -> dict:
    orch.reset()
    turns, calls, tool_trace = [], [], []
    for t in scn["turns"]:
        t0 = time.perf_counter()
        try:
            res = orch.chat(t["user"])
            reply, decision, err, trace_id = res.reply, res.decision, None, res.trace_id
            calls += res.tool_calls
        except Exception as e:
            reply, decision, err = f"[LỖI HỆ THỐNG] {type(e).__name__}: {e}", {"agents": [], "missing_info": []}, str(e)
            trace_id = tracer.spans[-1]["trace_id"] if tracer.spans else ""
        wall = (time.perf_counter() - t0) * 1000
        s = tracer.summary(trace_id) if trace_id else {}
        spans = [x for x in tracer.spans if x["trace_id"] == trace_id]
        tool_trace += [{"name": x["attrs"].get("tool"), "args": x["attrs"].get("args"), "result": x["attrs"].get("result")}
                       for x in spans if x["kind"] == "tool"]
        router_ms = sum(x["latency_ms"] for x in spans if x["kind"] == "router")
        turns.append({
            "user": t["user"], "assistant": reply, "decision": decision, "error": err,
            "gold": {"intent": t["intent"], "agents": t["agents"], "missing": t["missing"]},
            "routing_correct": routing_correct(decision, t["agents"], t["missing"]),
            "intent_ok": decision.get("intent") == t["intent"],
            "latency_ms": round(wall, 1), "router_ms": round(router_ms, 1),
            "cost_usd": s.get("cost_usd", 0.0), "llm_calls": s.get("llm_calls", 0),
            "input_tokens": s.get("input_tokens", 0), "output_tokens": s.get("output_tokens", 0),
            "cache_read_tokens": s.get("cache_read_tokens", 0),
        })
    f1 = tool_f1(calls, scn["expected_tools"])
    return {"id": scn["id"], "category": scn["category"], "turns": turns, "tool_calls": calls,
            "tool_trace": tool_trace, "tool_f1": f1, "cost_usd": round(sum(t["cost_usd"] for t in turns), 6)}


def summarize(name: str, rows: list[dict], judge_cost: float, cfg: dict) -> dict:
    turns = [t for r in rows for t in r["turns"]]
    lat = [t["latency_ms"] for t in turns]
    judged = [r for r in rows if "verdict" in r]
    passed = [r for r in judged if r["verdict"] == "PASS"]
    cost = sum(r["cost_usd"] for r in rows)
    cats = {}
    for c in sorted({r["category"] for r in rows}):
        rs = [r for r in judged if r["category"] == c]
        cats[c] = {"n": len(rs), "success": round(sum(r["verdict"] == "PASS" for r in rs) / len(rs), 3) if rs else None}
    return {
        "config": name, "router": cfg["router"], "router_llm": cfg.get("router_llm"), "agents": cfg["agents"],
        "overrides": cfg.get("overrides"), "n_scenarios": len(rows), "n_turns": len(turns),
        "task_success_rate": round(len(passed) / len(judged), 4) if judged else None,
        "routing_acc": round(sum(t["routing_correct"] for t in turns) / len(turns), 4),
        "intent_acc": round(sum(t["intent_ok"] for t in turns) / len(turns), 4),
        "tool_f1_micro": micro_f1([r["tool_f1"] for r in rows]),
        "tool_f1_macro": round(sum(r["tool_f1"]["f1"] for r in rows) / len(rows), 4),
        "latency_ms_mean": round(sum(lat) / len(lat), 1), "latency_ms_p50": round(percentile(lat, 0.5), 1),
        "latency_ms_p95": round(percentile(lat, 0.95), 1),
        "router_ms_mean": round(sum(t["router_ms"] for t in turns) / len(turns), 1),
        "cost_total_usd": round(cost, 5), "cost_per_scenario_usd": round(cost / len(rows), 5),
        "cost_per_success_usd": round(cost / len(passed), 5) if passed else None,
        "llm_calls_per_turn": round(sum(t["llm_calls"] for t in turns) / len(turns), 2),
        "input_tokens": sum(t["input_tokens"] for t in turns), "output_tokens": sum(t["output_tokens"] for t in turns),
        "cache_read_tokens": sum(t["cache_read_tokens"] for t in turns),
        "errors": sum(1 for t in turns if t["error"]), "judge_cost_usd": round(judge_cost, 5), "by_category": cats,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["api-opus-medium"], choices=list(CONFIGS))
    ap.add_argument("--ids", nargs="*", help="chỉ chạy các kịch bản này")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tools", choices=["local", "mcp"], default="local")
    ap.add_argument("--no-judge", action="store_true")
    a = ap.parse_args()

    scenarios = json.loads((config.DATA_DIR / "eval_set.json").read_text(encoding="utf-8"))
    if a.ids:
        scenarios = [s for s in scenarios if s["id"] in a.ids]
    if a.limit:
        scenarios = scenarios[: a.limit]

    judge_tracer = Tracer(run_name=f"judge-{time.strftime('%Y%m%d-%H%M%S')}")
    judge = None if a.no_judge else Judge(tracer=judge_tracer)
    for name in a.configs:
        out_dir = config.RESULTS_DIR / "eval" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        tracer = Tracer(run_name=f"{name}-{time.strftime('%Y%m%d-%H%M%S')}")
        orch = build(name, tracer, a.tools)
        rows = []
        jc0 = sum(s["attrs"].get("cost_usd", 0) for s in judge_tracer.spans)
        for scn in scenarios:
            r = run_scenario(orch, tracer, scn)
            if judge:
                try:
                    g = judge.grade(scn["criteria"], r["turns"], r["tool_trace"])
                    r.update(verdict=g["verdict"], judge=g)
                except Exception as e:
                    r.update(verdict="FAIL", judge={"error": str(e)})
            rows.append(r)
            print(f"[{name}] {scn['id']} {r.get('verdict', '-'):4} F1={r['tool_f1']['f1']:.2f} "
                  f"route={'/'.join('✓' if t['routing_correct'] else '✗' for t in r['turns'])} "
                  f"${r['cost_usd']:.4f} {sum(t['latency_ms'] for t in r['turns']) / 1000:.1f}s")
        orch.backend.close()
        judge_cost = sum(s["attrs"].get("cost_usd", 0) for s in judge_tracer.spans) - jc0
        summary = summarize(name, rows, judge_cost, CONFIGS[name])
        summary["tools_backend"] = a.tools
        with (out_dir / "details.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps({k: v for k, v in summary.items() if k != "by_category"}, ensure_ascii=False, indent=1))
        print(f"trace: {tracer.path}")


if __name__ == "__main__":
    main()
