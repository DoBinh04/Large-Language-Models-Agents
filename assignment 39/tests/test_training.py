"""Test nhẹ cho phần training (không GPU, không mạng, không API). Chỉ cần các file có trong notebook Colab.

    python -m tests.test_training
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from support.router import RuleRouter, normalize_decision  # noqa: E402


def test_training_data():
    from training.gen_data import sample, to_record
    from training.gen_dpo_pairs import perturb

    rng = random.Random(0)
    kinds = set()
    for _ in range(500):
        kind, st, msg, target = sample(rng)
        kinds.add(kind)
        rec = to_record(kind, st, msg, target)
        assert json.loads(rec["completion"][0]["content"]) == target
        assert target == normalize_decision(target), target
        if target["missing_info"]:
            assert not target["agents"] and not st.entities.get("order_id")
        if kind == "in_state":
            assert target["agents"] and not target["missing_info"], (msg, target)
        for src, rej in perturb(rec, rng):
            assert rej != rec["completion"][0]["content"], src
    assert {"single", "missing", "pending", "in_state", "multi"} <= kinds, kinds
    print("training data ok")


def test_hf_import():
    """Converter dataset HF (orchestrator từng bước) -> RoutingDecision cả lượt, không cần mạng."""
    from training.hf_import import convert_conversation, remap_order_ids

    def step(action, agent=None, task=None, message=None, order_id=None):
        return {"decision": {"check": {"last_ok": None, "reason": ""}, "action": action, "agent": agent, "task": task,
                             "message": message, "state_update": {"order_id": order_id, "product": None, "issue": None}}}

    conv = {"id": "X-001", "category": "return_create", "turns": [
        {"user": "Màn hình mới mua bị điểm chết, muốn đổi", "assistant": "Bạn cho mình mã đơn nhé",
         "steps": [step("ask_user", message="Bạn cho mình xin mã đơn hàng (DH + 4 số) nhé.")]},
        {"user": "dh1009", "assistant": "Đã tạo phiếu đổi trả",
         "steps": [step("call_agent", "policy", "Kiểm tra điều kiện đổi trả đơn DH1009", order_id="dh1009"),
                   step("call_agent", "policy", "Tạo phiếu đổi trả đơn DH1009"), step("finish")]},
        {"user": "đơn đó giao bằng hãng nào vậy", "assistant": "GHN",
         "steps": [json.dumps(step("call_agent", "order", "Tra hãng vận chuyển đơn DH1009")), step("finish")]},
        {"user": "cảm ơn shop", "assistant": "Không có gì ạ", "steps": [step("finish")]},
    ]}
    stats: dict = {}
    recs = convert_conversation(conv, stats)
    t = [r["target"] for r in recs]
    assert [r["kind"] for r in recs] == ["missing", "pending", "in_state", "chat"], [r["kind"] for r in recs]
    assert t[0]["missing_info"] == ["order_id"] and t[0]["intent"] == "return_request" and not t[0]["agents"]
    assert t[1]["intent"] == "return_request" and t[1]["agents"] == ["policy"] and len(t[1]["plan"]) == 1
    assert t[1]["entities"]["order_id"] == "DH1009"
    assert t[2]["agents"] == ["order"] and t[2]["intent"] == "order_status"
    assert '"order_id": "DH1009"' in recs[2]["state"] and '"pending_intent": ""' in recs[2]["state"]
    assert t[3]["intent"] == "chitchat" and t[3]["reply"] == "Không có gì ạ"
    assert '"pending_intent": "return_request"' in recs[1]["state"]

    mapped = remap_order_ids(conv, random.Random(0), set())
    s = json.dumps(mapped, ensure_ascii=False)
    assert "1009" not in s
    assert re.fullmatch(r"DH\d{4}", mapped["turns"][1]["user"])
    print("hf import ok")


def test_routing_eval_cases():
    """eval_set.json đọc được và router luật chạy hết các lượt (kiểm tra file data đã có đủ)."""
    from eval.routing_eval import evaluate, load_cases
    from support.tracing import Tracer

    cases = load_cases(0)
    assert len(cases) >= 30, len(cases)
    tracer = Tracer(enabled=False)
    res = evaluate(RuleRouter(tracer), tracer, cases)
    assert res["summary"]["errors"] == 0, res["summary"]
    print(f"routing eval ok: {len(cases)} lượt, rule acc = {res['summary']['routing_acc']}")


if __name__ == "__main__":
    test_training_data()
    test_hf_import()
    test_routing_eval_cases()
    print("TRAINING OK")
