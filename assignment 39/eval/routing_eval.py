"""Routing-only eval: so sánh API LLM vs Small LLM (base) vs SFT vs DPO (vs rule) ở bước điều phối.

Không chạy agent nên rẻ và nhanh: chỉ đo quyết định của Router trên
  - eval set (data/eval_set.json): mỗi lượt với STATE "chuẩn" (mô phỏng các lượt trước đúng như kỳ vọng)
  - tập test sinh sẵn (training/data/sft_test.jsonl), có đủ nhóm state-aware.

Metrics: routing accuracy (tập agents + có hỏi lại), intent accuracy, JSON valid rate (small LLM),
latency trung bình / p95, chi phí / 1000 quyết định (API: theo token; small: theo giờ GPU).

    python -m eval.routing_eval --routers rule api                    # không cần GPU
    python -m eval.routing_eval --routers small sft dpo --test-limit 300   # trên máy có GPU
    python -m eval.routing_eval --routers api --api-model claude-haiku-5-5 --api-effort low
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from eval.metrics import percentile, routing_correct  # noqa: E402
from support.router import make_router  # noqa: E402
from support.state import ASK_ORDER_ID, ConversationState, extract_slots  # noqa: E402
from support.tracing import Tracer  # noqa: E402


def gold_turns(scn: dict):
    """Sinh (state, user_msg, gold) cho từng lượt, giả định các lượt trước được xử lý đúng."""
    st = ConversationState()
    for turn in scn["turns"]:
        st.merge(extract_slots(turn["user"]))
        yield copy.deepcopy(st), turn["user"], {"intent": turn["intent"], "agents": turn["agents"], "missing": turn["missing"]}
        if turn["missing"]:
            st.pending_intent, st.pending_missing, reply = turn["intent"], ["order_id"], ASK_ORDER_ID
        else:
            st.pending_intent, st.pending_missing, reply = "", [], "(trợ lý đã trả lời yêu cầu)"
            st.last_agents = turn["agents"] or st.last_agents
        st.add_turn(turn["user"], reply)


def load_cases(test_limit: int) -> list[dict]:
    cases = []
    for scn in json.loads((config.DATA_DIR / "eval_set.json").read_text(encoding="utf-8")):
        for i, (st, msg, gold) in enumerate(gold_turns(scn)):
            cases.append({"src": "eval_set", "id": f"{scn['id']}.{i + 1}", "kind": scn["category"],
                          "state": st, "msg": msg, "gold": gold})
    test = config.TRAIN_DATA_DIR / "sft_test.jsonl"
    if test.exists() and test_limit:
        for j, line in enumerate(test.open(encoding="utf-8")):
            if j >= test_limit:
                break
            r = json.loads(line)
            t = r["target"]
            cases.append({"src": "test", "id": f"T{j}", "kind": r["kind"], "state": ConversationState.from_view(r["state"]),
                          "msg": r["user_message"],
                          "gold": {"intent": t["intent"], "agents": t["agents"], "missing": bool(t["missing_info"])}})
    return cases


def evaluate(router, tracer: Tracer, cases: list[dict]) -> dict:
    rows = []
    for c in cases:
        with tracer.span("routing_case", kind="eval", case=c["id"]) as root:
            try:
                d = router.decide(c["state"], c["msg"])
                err = None
            except Exception as e:
                d, err = {"intent": "", "agents": [], "missing_info": []}, f"{type(e).__name__}: {e}"
        s = tracer.summary(root.trace_id)
        rspan = next((x for x in tracer.spans if x["trace_id"] == root.trace_id and x["kind"] == "router"), {"attrs": {}})
        rows.append({"id": c["id"], "src": c["src"], "kind": c["kind"], "msg": c["msg"], "gold": c["gold"], "pred": d,
                     "correct": routing_correct(d, c["gold"]["agents"], c["gold"]["missing"]),
                     "intent_ok": d.get("intent") == c["gold"]["intent"],
                     "json_valid": rspan["attrs"].get("json_valid", err is None),
                     "latency_ms": s["latency_ms"], "cost_usd": s["cost_usd"], "error": err})
    return {"rows": rows, "summary": summarize(rows)}


def summarize(rows: list[dict]) -> dict:
    def acc(rs, key="correct"):
        return round(sum(r[key] for r in rs) / len(rs), 4) if rs else None

    lat = [r["latency_ms"] for r in rows]
    by_kind = {}
    for k in sorted({r["kind"] for r in rows}):
        rs = [r for r in rows if r["kind"] == k]
        by_kind[k] = {"n": len(rs), "acc": acc(rs)}
    return {
        "n": len(rows),
        "routing_acc": acc(rows), "intent_acc": acc(rows, "intent_ok"), "json_valid": acc(rows, "json_valid"),
        "routing_acc_eval_set": acc([r for r in rows if r["src"] == "eval_set"]),
        "routing_acc_test": acc([r for r in rows if r["src"] == "test"]),
        "latency_ms_mean": round(sum(lat) / len(lat), 1) if lat else 0,
        "latency_ms_p95": round(percentile(lat, 0.95), 1),
        "cost_per_1k_usd": round(sum(r["cost_usd"] for r in rows) / len(rows) * 1000, 4) if rows else 0,
        "errors": sum(1 for r in rows if r["error"]),
        "by_kind": by_kind,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routers", nargs="+", default=["rule", "api"], help="rule api small sft dpo")
    ap.add_argument("--api-model", default=config.BASE_MODEL)
    ap.add_argument("--api-effort", default=config.BASE_EFFORT)
    ap.add_argument("--test-limit", type=int, default=150, help="số mẫu lấy từ sft_test.jsonl (0 = bỏ)")
    a = ap.parse_args()

    cases = load_cases(a.test_limit)
    out_dir = config.RESULTS_DIR / "routing"
    out_dir.mkdir(parents=True, exist_ok=True)
    table = []
    for kind in a.routers:
        tracer = Tracer(enabled=False)
        router = make_router(kind, tracer, model=a.api_model, effort=a.api_effort)
        name = router.label if kind != "api" else f"api:{a.api_model}@{a.api_effort}"
        print(f"== {name}: {len(cases)} cases")
        res = evaluate(router, tracer, cases)
        safe = name.replace(":", "_").replace("@", "_").replace("/", "_")
        (out_dir / f"{safe}.json").write_text(json.dumps({"router": name, **res}, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
        s = res["summary"]
        table.append((name, s))
        print(json.dumps({k: v for k, v in s.items() if k != "by_kind"}, ensure_ascii=False))

    print("\n| Router | Routing acc | Intent acc | Acc (eval set) | Acc (test) | JSON valid | Latency mean (ms) | p95 | $/1k |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, s in table:
        print(f"| {name} | {s['routing_acc']} | {s['intent_acc']} | {s['routing_acc_eval_set']} | {s['routing_acc_test']} "
              f"| {s['json_valid']} | {s['latency_ms_mean']} | {s['latency_ms_p95']} | {s['cost_per_1k_usd']} |")


if __name__ == "__main__":
    main()
