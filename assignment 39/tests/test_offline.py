"""Test offline: không gọi API, không cần GPU / sentence-transformers.

    python -m tests.test_offline

Kiểm tra: logic tool khớp với tiêu chí eval, BM25 tìm đúng tài liệu, Tool-F1, A2A round-trip,
router rule + normalize, dữ liệu SFT/DPO sinh ra đúng format và nhãn nhất quán.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["RAG_LIGHT"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import routing_correct, tool_f1  # noqa: E402
from support import a2a, tools  # noqa: E402
from support.rag import HybridRetriever  # noqa: E402
from support.router import RuleRouter, normalize_decision  # noqa: E402
from support.state import ConversationState, extract_slots  # noqa: E402
from tests.test_training import test_hf_import, test_training_data  # noqa: E402,F401


def test_tools():
    tools.reset()
    r = tools.check_return_eligibility("DH1001", reason="change_of_mind", condition="sealed")
    assert r["eligible"] is False and r["days_since_delivery"] == 16, r                     # S14
    r = tools.check_return_eligibility("DH1005", "P002", "change_of_mind", "opened_unactivated")
    assert r["eligible"] is False and "true wireless" in r["rule"], r                       # S15
    r = tools.create_return_request("DH1005", "P012", "change_of_mind", "sealed")
    assert r["ok"] and r["refund_amount"] == 450000 and r["refund_time"] == "5–7 ngày làm việc", r   # S16
    r = tools.check_return_eligibility("DH1012", "P008", "change_of_mind", "activated")
    assert r["eligible"] is False, r                                                         # S17
    r = tools.check_return_eligibility("DH1008", reason="defect")
    assert r["eligible"] is True and r["days_since_delivery"] == 27, r                      # S20
    r = tools.check_warranty("DH1007")
    assert r["in_warranty"] and r["end"] == "2027-09-04", r                                 # S18
    r = tools.check_warranty("DH1011")
    assert r["end"] == "2027-12-12", r                                                      # S23
    r = tools.check_warranty("DH1010")
    assert r["in_warranty"] is False, r                                                     # S28
    r = tools.check_return_eligibility("DH1006")
    assert r["eligible"] is False, r                                                        # S27
    assert tools.cancel_order("DH1002")["ok"] is False                                      # S09
    assert tools.cancel_order("DH1003")["ok"] is True                                       # S08
    tools.reset()
    assert tools.lookup_order("DH1003")["status"] == "processing"
    assert {o["order_id"] for o in tools.list_orders_by_phone("0901234567")["orders"]} == {"DH1001", "DH1004"}
    hits = tools.search_products("laptop gaming", max_price=25_000_000, in_stock_only=True)["products"]
    assert hits and hits[0]["id"] == "P006", hits                                           # S31
    print("tools ok")


def test_rag():
    r = HybridRetriever(use_dense=False, use_rerank=False)
    assert r.mode == "bm25"
    top = r.search("trả góp 0% áp dụng cho đơn từ bao nhiêu", top_k=3)
    assert any("Trả góp" in h["text"] for h in top), top
    top = r.search("doi y tra hang trong bao nhieu ngay", top_k=3)                          # gõ không dấu
    assert any("15 ngày" in h["text"] for h in top), top
    top = r.search("Nova Air 14 nâng cấp RAM", top_k=2)
    assert any("hàn chết" in h["text"] for h in top), top
    print("rag ok")


def test_metrics():
    exp = [{"name": "check_return_eligibility", "args": {"order_id": "DH1005", "reason": "change_of_mind"}},
           {"name": "create_return_request", "args": {"order_id": "DH1005"}}]
    pred = [{"name": "lookup_order", "args": {"order_id": "DH1005"}},
            {"name": "check_return_eligibility", "args": {"order_id": "dh1005", "reason": "change_of_mind", "condition": "sealed"}}]
    f = tool_f1(pred, exp)
    assert (f["tp"], f["fp"], f["fn"]) == (1, 1, 1) and f["f1"] == 0.5, f
    assert tool_f1([], [])["f1"] == 1.0
    assert tool_f1([{"name": "compare_products", "args": {"product_ids": ["P006", "P005"]}}],
                   [{"name": "compare_products", "args": {"product_ids": ["P005", "P006"]}}])["f1"] == 1.0
    assert routing_correct({"agents": ["product", "order"], "missing_info": []}, ["order", "product"], False)
    assert not routing_correct({"agents": [], "missing_info": ["order_id"]}, ["policy"], False)
    print("metrics ok")


class EchoAgent:
    card = a2a.AgentCard(name="echo", description="echo", skills=[a2a.AgentSkill("echo", "Echo", "")])

    def handle(self, message):
        data = a2a.get_data(message)
        return a2a.make_task(message, "completed", "ok", [a2a.artifact("r", {"answer": a2a.get_text(message), **data})])


def test_a2a():
    client = a2a.A2AClient(a2a.InProcessTransport({"echo": EchoAgent()}))
    task = client.send("echo", "xin chào", {"order_id": "DH1001"})
    assert task["status"]["state"] == "completed"
    art = a2a.get_data(task["artifacts"][0])
    assert art == {"answer": "xin chào", "order_id": "DH1001"}, art
    print("a2a ok")


def test_router_and_state():
    assert extract_slots("đơn dh 1005 của sđt 0933333333") == {"order_id": "DH1005", "phone": "0933333333"}
    st = ConversationState()
    rr = RuleRouter()
    d = rr.decide(st, "Kiểm tra đơn hàng giúp mình với")
    assert d["missing_info"] == ["order_id"] and d["agents"] == []
    st.pending_intent = "order_status"
    st.merge(extract_slots("DH1003"))
    d = rr.decide(st, "DH1003")
    assert d["agents"] == ["order"], d
    d = normalize_decision({"intent": "multi", "plan": [{"agent": "order", "task": "x"}, {"agent": "bogus", "task": "y"}]})
    assert d["agents"] == ["order"] and d["entities"] == {"order_id": "", "phone": "", "product": ""}
    v = ConversationState.from_view(st.router_view())
    assert v.entities["order_id"] == "DH1003" and v.pending_intent == "order_status"
    print("router/state ok")


def test_eval_set():
    scns = json.loads((Path(__file__).resolve().parent.parent / "data" / "eval_set.json").read_text(encoding="utf-8"))
    assert len({s["id"] for s in scns}) == len(scns)
    for s in scns:
        for e in s["expected_tools"]:
            assert e["name"] in tools.TOOLS, e
    print(f"eval set ok ({len(scns)} kịch bản, {sum(len(s['turns']) for s in scns)} lượt)")


def test_query_rewrite():
    from support.query_rewriter import normalize, rewrite_query

    assert normalize("tai nghe tws bóc seal r đổi đc k") == "tai nghe true wireless đã mở hộp rồi đổi được không"
    assert normalize("bh mh bao lâu") == "bảo hành màn hình bao lâu"
    assert normalize("đơn 5tr trả góp dc ko") == "đơn 5 triệu trả góp được không"
    assert normalize("đơn dh 1005 của mình") == "đơn dh 1005 của mình"                 # mã đơn giữ nguyên
    assert normalize("Nova View 27 4K có loa không") == "Nova View 27 4K có loa không"  # 4K không thành "4 nghìn"
    assert normalize("Nova Tab S") == "Nova Tab S"

    rq = rewrite_query("đổi ý thì được mấy ngày? hoàn tiền mất bao lâu", mode="rules")
    assert rq.queries[0] == rq.normalized and len(rq.queries) == 3, rq
    assert rewrite_query("trả góp 0%", mode="off").queries == ["trả góp 0%"]

    r = HybridRetriever(use_dense=False, use_rerank=False)
    hits = r.search_multi(rq.queries, top_k=6, rerank_query=rq.rerank_query)
    assert any("15 ngày" in h["text"] for h in hits), hits
    assert any("Thời gian hoàn tiền" in h["section"] for h in hits), hits

    tools.reset()
    out = tools.search_knowledge_base("tai nghe tws bóc seal r đổi đc k")
    assert out["searched_as"][0].startswith("tai nghe true wireless"), out
    print("query rewrite ok")


def test_logging_and_sessions():
    import io
    import logging
    import types

    from api.sessions import SessionStore
    from support.logging_setup import ContextFilter, JsonFormatter
    from support.tracing import Tracer

    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.setFormatter(JsonFormatter())
    h.addFilter(ContextFilter())
    lg = logging.getLogger("test.json")
    lg.addHandler(h)
    lg.setLevel(logging.INFO)
    lg.propagate = False
    tracer = Tracer(enabled=False)
    with tracer.span("turn", kind="turn") as sp:
        lg.info("hello", extra={"intent": "chitchat", "cost_usd": 0.01})
    rec = json.loads(buf.getvalue().strip())
    assert rec["msg"] == "hello" and rec["intent"] == "chitchat" and rec["trace_id"] == sp.trace_id, rec

    base = types.SimpleNamespace(state=ConversationState())
    store = SessionStore(base, ttl_s=3600, max_sessions=2)
    a = store.get_or_create(None)
    a.orch.state.turn = 5
    assert store.get_or_create(a.id) is a and base.state.turn == 0          # state riêng từng session
    store.get_or_create("b"), store.get_or_create("c")                      # vượt max -> bỏ session cũ nhất
    assert store.get(a.id) is None and len(store) == 2
    assert store.reset("b") and store.get("b").orch.state.turn == 0
    print("logging/sessions ok")


if __name__ == "__main__":
    test_tools()
    test_rag()
    test_query_rewrite()
    test_logging_and_sessions()
    test_metrics()
    test_a2a()
    test_router_and_state()
    test_training_data()
    test_hf_import()
    test_eval_set()
    print("ALL OK")
