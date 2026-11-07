"""Business tools của cửa hàng: tìm sản phẩm, tra đơn, kiểm tra đổi trả / bảo hành, tạo phiếu.

Mỗi tool = hàm Python thuần (input dict -> output dict) + JSON Schema. Cùng một registry này được:
- agent gọi trực tiếp (in-process, nhanh, dùng khi eval), hoặc
- expose ra ngoài qua MCP server (support/mcp_server.py) để agent / Claude Desktop gọi qua MCP.

Dữ liệu là file JSON tĩnh; các tool "ghi" (tạo phiếu, huỷ đơn) chỉ ghi vào bộ nhớ, gọi reset()
trước mỗi kịch bản eval để kết quả tái lập được. TODAY cố định cho cùng lý do.
"""
from __future__ import annotations

import copy
import json
from datetime import date, timedelta
from typing import Callable

import config
from support import rag
from support.query_rewriter import rewrite_query

TODAY = date(2026, 10, 29)
ACTIVATABLE = {"điện thoại", "laptop", "máy tính bảng", "đồng hồ"}

PRODUCTS: dict[str, dict] = {p["id"]: p for p in json.loads((config.DATA_DIR / "products.json").read_text(encoding="utf-8"))}
_ORDERS0: dict[str, dict] = {o["order_id"]: o for o in json.loads((config.DATA_DIR / "orders.json").read_text(encoding="utf-8"))["orders"]}
ORDERS: dict[str, dict] = copy.deepcopy(_ORDERS0)
TICKETS: list[dict] = []


def reset() -> None:
    global ORDERS
    ORDERS = copy.deepcopy(_ORDERS0)
    TICKETS.clear()


def _d(s: str) -> date:
    return date.fromisoformat(s)


def _add_months(d: date, months: int) -> date:
    y, m = divmod(d.month - 1 + months, 12)
    day = min(d.day, [31, 29 if (d.year + y) % 4 == 0 else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m])
    return date(d.year + y, m + 1, day)


def _norm_order_id(s: str) -> str:
    return s.strip().upper().replace(" ", "")


def _brief(p: dict) -> dict:
    return {"id": p["id"], "name": p["name"], "category": p["category"], "price": p["price"],
            "in_stock": p["stock"] > 0, "warranty_months": p["warranty_months"]}


# --------------------------------------------------------------------------- #
# Product tools
# --------------------------------------------------------------------------- #
def search_products(query: str, category: str | None = None, max_price: int | None = None,
                    min_price: int | None = None, in_stock_only: bool = False, limit: int = 5) -> dict:
    q = set(rag.tokenize(query))
    hits = []
    for p in PRODUCTS.values():
        if category and rag.strip_accents(category.lower()) not in rag.strip_accents(p["category"]):
            continue
        if max_price and p["price"] > max_price:
            continue
        if min_price and p["price"] < min_price:
            continue
        if in_stock_only and p["stock"] <= 0:
            continue
        text = " ".join([p["name"], p["category"], p["brand"], " ".join(p["tags"]),
                         " ".join(f"{k} {v}" for k, v in p["specs"].items())])
        score = len(q & set(rag.tokenize(text)))
        if score or not query.strip():
            hits.append((score, p))
    hits.sort(key=lambda x: (-x[0], x[1]["price"]))
    return {"count": len(hits), "products": [_brief(p) for _, p in hits[:limit]]}


def get_product_details(product_id: str) -> dict:
    p = PRODUCTS.get(product_id.strip().upper())
    if not p:
        return {"error": f"Không tìm thấy sản phẩm {product_id}"}
    return {**p, "in_stock": p["stock"] > 0}


def compare_products(product_ids: list[str]) -> dict:
    found = [PRODUCTS[i.strip().upper()] for i in product_ids if i.strip().upper() in PRODUCTS]
    missing = [i for i in product_ids if i.strip().upper() not in PRODUCTS]
    keys = sorted({k for p in found for k in p["specs"]})
    table = {k: {p["id"]: p["specs"].get(k, "-") for p in found} for k in keys}
    table["giá"] = {p["id"]: p["price"] for p in found}
    table["bảo hành (tháng)"] = {p["id"]: p["warranty_months"] for p in found}
    return {"products": [_brief(p) for p in found], "comparison": table, "missing": missing}


def search_knowledge_base(query: str, top_k: int = config.TOP_K_FINAL) -> dict:
    """Query rewrite -> multi-query hybrid search trên tài liệu: chính sách + hướng dẫn sản phẩm."""
    rq = rewrite_query(query)
    hits = rag.get_retriever().search_multi(rq.queries, top_k=top_k, keywords=rq.keywords,
                                            rerank_query=rq.rerank_query)
    out = {"results": [{"source": f"{h['doc']} > {h['section']}", "text": h["text"],
                        "chunk_id": h["chunk_id"]} for h in hits]}
    if rq.changed:                     # cho agent (và trace) thấy truy vấn thực sự đã dùng
        out["searched_as"] = rq.queries
    return out


# --------------------------------------------------------------------------- #
# Order tools
# --------------------------------------------------------------------------- #
def _public_order(o: dict) -> dict:
    out = {k: v for k, v in o.items() if k != "phone"}
    out["phone_masked"] = o["phone"][:3] + "****" + o["phone"][-3:]
    out["items"] = [{**it, "name": PRODUCTS[it["product_id"]]["name"]} for it in o["items"]]
    out["total"] = sum(it["price"] * it["qty"] for it in o["items"])
    return out


def lookup_order(order_id: str, phone: str | None = None) -> dict:
    o = ORDERS.get(_norm_order_id(order_id))
    if not o:
        return {"error": f"Không tìm thấy đơn {order_id}. Mã đơn có dạng DHxxxx."}
    if phone and phone.strip() != o["phone"]:
        return {"error": "Số điện thoại không khớp với đơn hàng."}
    return _public_order(o)


def list_orders_by_phone(phone: str) -> dict:
    found = [o for o in ORDERS.values() if o["phone"] == phone.strip()]
    return {"count": len(found), "orders": [
        {"order_id": o["order_id"], "status": o["status"], "order_date": o["order_date"],
         "items": [PRODUCTS[i["product_id"]]["name"] for i in o["items"]]} for o in found]}


def cancel_order(order_id: str, reason: str = "") -> dict:
    o = ORDERS.get(_norm_order_id(order_id))
    if not o:
        return {"error": f"Không tìm thấy đơn {order_id}"}
    if o["status"] != "processing":
        return {"ok": False, "status": o["status"],
                "message": "Chỉ huỷ được đơn đang processing. Đơn shipping: từ chối nhận hàng khi shipper giao."}
    o["status"] = "cancelled"
    return {"ok": True, "order_id": o["order_id"], "status": "cancelled",
            "message": "Đã huỷ đơn. Tiền đã thanh toán (nếu có) hoàn trong 3–5 ngày làm việc."}


# --------------------------------------------------------------------------- #
# Policy tools: đổi trả & bảo hành
# --------------------------------------------------------------------------- #
def _order_item(order_id: str, product_id: str | None):
    o = ORDERS.get(_norm_order_id(order_id))
    if not o:
        return None, None, {"error": f"Không tìm thấy đơn {order_id}"}
    ids = [i["product_id"] for i in o["items"]]
    if not product_id:
        if len(ids) > 1:
            return o, None, {"error": "Đơn có nhiều sản phẩm, cần chỉ rõ product_id",
                             "items": [{"product_id": i, "name": PRODUCTS[i]["name"]} for i in ids]}
        product_id = ids[0]
    product_id = product_id.strip().upper()
    if product_id not in ids:
        return o, None, {"error": f"Sản phẩm {product_id} không thuộc đơn {o['order_id']}"}
    return o, PRODUCTS[product_id], None


def check_return_eligibility(order_id: str, product_id: str | None = None, reason: str = "change_of_mind",
                             condition: str | None = None) -> dict:
    """reason: defect | change_of_mind ; condition: sealed | opened_unactivated | activated"""
    o, p, err = _order_item(order_id, product_id)
    if err:
        return err
    base = {"order_id": o["order_id"], "product_id": p["id"], "product": p["name"], "reason": reason}
    if o["status"] in ("cancelled", "returned"):
        return {**base, "eligible": False, "rule": f"Đơn đã {o['status']}, không thể đổi trả."}
    if o["status"] in ("processing", "shipping"):
        alt = "Có thể huỷ miễn phí." if o["status"] == "processing" else "Từ chối nhận hàng khi shipper giao."
        return {**base, "eligible": False, "rule": f"Đơn chưa giao ({o['status']}). {alt}"}
    day_n = (TODAY - _d(o["delivered_date"])).days + 1          # tính cả ngày nhận là ngày 1
    base["days_since_delivery"] = day_n
    if reason == "defect":
        if day_n <= 30:
            return {**base, "eligible": True, "refund_percent": 100, "option": "đổi mới cùng model hoặc hoàn 100%",
                    "rule": "Lỗi NSX trong 30 ngày: đổi mới/hoàn 100%, miễn phí vận chuyển."}
        return {**base, "eligible": False, "rule": "Quá 30 ngày: xử lý theo chính sách bảo hành (check_warranty)."}
    if day_n > 15:
        return {**base, "eligible": False, "rule": f"Đổi ý chỉ áp dụng trong 15 ngày; đơn đã nhận được {day_n} ngày."}
    if condition is None:
        return {**base, "eligible": None, "need_info": "condition (sealed | opened_unactivated | activated)"}
    if condition == "sealed":
        return {**base, "eligible": True, "refund_percent": 100, "rule": "Còn seal: hoàn 100%. Khách chịu phí ship trả."}
    if "true wireless" in p["tags"]:
        return {**base, "eligible": False, "rule": "Tai nghe true wireless đã mở hộp không áp dụng đổi ý (vệ sinh)."}
    if condition == "activated" and p["category"] in ACTIVATABLE:
        return {**base, "eligible": False, "rule": f"{p['category']} đã kích hoạt không áp dụng đổi ý."}
    return {**base, "eligible": True, "refund_percent": 90, "rule": "Đã mở hộp, chưa kích hoạt: hoàn 90% (phí 10%)."}


def create_return_request(order_id: str, product_id: str, reason: str, condition: str | None = None,
                          description: str = "") -> dict:
    chk = check_return_eligibility(order_id, product_id, reason, condition)
    if not chk.get("eligible"):
        return {"ok": False, "eligibility": chk}
    o = ORDERS[_norm_order_id(order_id)]
    price = next(i["price"] for i in o["items"] if i["product_id"] == chk["product_id"])
    refund_days = "5–7 ngày làm việc" if o["payment"] == "COD" else "3–5 ngày làm việc"
    t = {"ticket_id": f"RMA-{2001 + len(TICKETS)}", "type": "return", "order_id": o["order_id"],
         "product_id": chk["product_id"], "reason": reason, "condition": condition, "description": description,
         "refund_amount": price * chk["refund_percent"] // 100, "refund_time": refund_days,
         "next_step": "GHN đến lấy hàng trong 1–2 ngày làm việc; xác nhận trong 48 giờ."}
    TICKETS.append(t)
    return {"ok": True, **t}


def check_warranty(order_id: str, product_id: str | None = None) -> dict:
    o, p, err = _order_item(order_id, product_id)
    if err:
        return err
    if o["status"] in ("cancelled", "returned"):
        return {"order_id": o["order_id"], "in_warranty": False, "status": o["status"],
                "rule": f"Đơn đã {o['status']} ({o.get('refund_status', '')}), sản phẩm không còn thuộc bảo hành của khách."}
    if not o.get("delivered_date"):
        return {"order_id": o["order_id"], "in_warranty": None, "rule": "Đơn chưa giao, chưa kích hoạt bảo hành."}
    start = _d(o["delivered_date"])
    end = _add_months(start, p["warranty_months"])
    return {"order_id": o["order_id"], "product_id": p["id"], "product": p["name"],
            "warranty_months": p["warranty_months"], "start": start.isoformat(), "end": end.isoformat(),
            "in_warranty": TODAY <= end, "days_left": max(0, (end - TODAY).days),
            "within_30_days_exchange": (TODAY - start).days + 1 <= 30}


def create_warranty_ticket(order_id: str, product_id: str, issue: str) -> dict:
    w = check_warranty(order_id, product_id)
    if "error" in w or not w.get("in_warranty"):
        return {"ok": False, "warranty": w, "message": "Hết hạn bảo hành: sửa chữa có tính phí (kiểm tra 100.000đ)."}
    t = {"ticket_id": f"BH-{3001 + len(TICKETS)}", "type": "warranty", "order_id": w["order_id"],
         "product_id": w["product_id"], "issue": issue, "processing_time": "7–14 ngày làm việc",
         "free_shipping": PRODUCTS[w["product_id"]]["price"] > 5_000_000}
    TICKETS.append(t)
    return {"ok": True, **t}


def search_policy(query: str, top_k: int = config.TOP_K_FINAL) -> dict:
    return search_knowledge_base(query, top_k)


# --------------------------------------------------------------------------- #
# Registry: schema + hàm + agent sở hữu
# --------------------------------------------------------------------------- #
def _schema(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


S, I, B = {"type": "string"}, {"type": "integer"}, {"type": "boolean"}
REASON = {"type": "string", "enum": ["defect", "change_of_mind"]}
CONDITION = {"type": "string", "enum": ["sealed", "opened_unactivated", "activated"]}

TOOLS: dict[str, dict] = {
    "search_products": {"agent": "product", "fn": search_products,
        "description": "Tìm sản phẩm trong catalog theo nhu cầu / từ khoá, lọc theo danh mục, khoảng giá (VND), còn hàng.",
        "schema": _schema({"query": S, "category": S, "max_price": I, "min_price": I, "in_stock_only": B}, ["query"])},
    "get_product_details": {"agent": "product", "fn": get_product_details,
        "description": "Lấy thông số, giá, tồn kho, thời hạn bảo hành của một sản phẩm theo mã (Pxxx).",
        "schema": _schema({"product_id": S}, ["product_id"])},
    "compare_products": {"agent": "product", "fn": compare_products,
        "description": "So sánh thông số / giá / bảo hành của 2–4 sản phẩm.",
        "schema": _schema({"product_ids": {"type": "array", "items": S}}, ["product_ids"])},
    "search_knowledge_base": {"agent": "product", "fn": search_knowledge_base,
        "description": "Tìm trong tài liệu hướng dẫn sử dụng / FAQ sản phẩm (hybrid search + rerank).",
        "schema": _schema({"query": S}, ["query"])},
    "lookup_order": {"agent": "order", "fn": lookup_order,
        "description": "Tra cứu đơn hàng theo mã (DHxxxx): trạng thái, sản phẩm, ngày giao, vận đơn, thanh toán.",
        "schema": _schema({"order_id": S, "phone": S}, ["order_id"])},
    "list_orders_by_phone": {"agent": "order", "fn": list_orders_by_phone,
        "description": "Liệt kê các đơn hàng của một số điện thoại (khi khách không nhớ mã đơn).",
        "schema": _schema({"phone": S}, ["phone"])},
    "cancel_order": {"agent": "order", "fn": cancel_order,
        "description": "Huỷ đơn hàng đang ở trạng thái processing. Chỉ gọi khi khách xác nhận muốn huỷ.",
        "schema": _schema({"order_id": S, "reason": S}, ["order_id"])},
    "search_policy": {"agent": "policy", "fn": search_policy,
        "description": "Tìm điều khoản trong chính sách đổi trả, bảo hành, vận chuyển, thanh toán (hybrid search + rerank).",
        "schema": _schema({"query": S}, ["query"])},
    "check_return_eligibility": {"agent": "policy", "fn": check_return_eligibility,
        "description": "Kiểm tra một sản phẩm trong đơn có được đổi trả không và hoàn bao nhiêu %. "
                       "reason: defect (lỗi NSX) | change_of_mind (đổi ý). condition: sealed | opened_unactivated | activated.",
        "schema": _schema({"order_id": S, "product_id": S, "reason": REASON, "condition": CONDITION}, ["order_id", "reason"])},
    "create_return_request": {"agent": "policy", "fn": create_return_request,
        "description": "Tạo phiếu đổi trả (RMA). Chỉ gọi sau khi đã kiểm tra đủ điều kiện và khách xác nhận muốn trả.",
        "schema": _schema({"order_id": S, "product_id": S, "reason": REASON, "condition": CONDITION, "description": S},
                          ["order_id", "product_id", "reason"])},
    "check_warranty": {"agent": "policy", "fn": check_warranty,
        "description": "Kiểm tra thời hạn bảo hành của sản phẩm trong đơn hàng.",
        "schema": _schema({"order_id": S, "product_id": S}, ["order_id"])},
    "create_warranty_ticket": {"agent": "policy", "fn": create_warranty_ticket,
        "description": "Tạo phiếu bảo hành (BH) cho sản phẩm còn hạn, mô tả lỗi trong issue.",
        "schema": _schema({"order_id": S, "product_id": S, "issue": S}, ["order_id", "product_id", "issue"])},
}

WRITE_TOOLS = {"cancel_order", "create_return_request", "create_warranty_ticket"}


def tools_for(agent: str) -> list[str]:
    return [n for n, t in TOOLS.items() if t["agent"] == agent]


def anthropic_schema(names: list[str]) -> list[dict]:
    return [{"name": n, "description": TOOLS[n]["description"], "input_schema": TOOLS[n]["schema"]} for n in names]


def call(name: str, args: dict) -> dict:
    if name not in TOOLS:
        return {"error": f"Tool không tồn tại: {name}"}
    try:
        return TOOLS[name]["fn"](**args)
    except TypeError as e:
        return {"error": f"Tham số không hợp lệ cho {name}: {e}"}


ToolFn = Callable[[str, dict], dict]
