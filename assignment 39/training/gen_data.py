"""Sinh dữ liệu SFT cho Router (Agent Routing + Planning + State-aware) — không tốn tiền API.

Mỗi mẫu = (STATE, tin nhắn khách) -> RoutingDecision JSON. Nhãn sinh bằng luật xác định từ template,
nên đúng tuyệt đối với quy tắc trong ROUTER_SYSTEM. Các nhóm mẫu:

  single     : 1 intent, có / không có mã đơn trong câu
  missing    : intent cần order_id nhưng STATE + câu đều không có -> missing_info
  pending    : STATE có pending_intent, khách chỉ đưa mã đơn -> tiếp tục intent cũ
  in_state   : mã đơn đã có trong STATE (lượt trước) -> KHÔNG được hỏi lại (state-aware)
  follow_up  : câu nối tiếp ngắn ("nó còn bảo hành không?") hiểu theo last_agents / entities
  switch     : đang pending nhưng khách hỏi chuyện khác -> bỏ pending
  multi      : 2 agent trong một câu
  chitchat / out_of_scope

Mã đơn trong dữ liệu train nằm ngoài DH1001–DH1012 (dùng cho eval) để tránh leak.
Tuỳ chọn --paraphrase N: dùng Claude (Haiku 5.5, rẻ) viết lại câu khách cho đa dạng, giữ nguyên nhãn.

    python -m training.gen_data --n 3000
    python -m training.gen_data --n 3000 --paraphrase 400
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from support.router import NEEDS_ORDER, ROUTER_SYSTEM, build_router_user, normalize_decision  # noqa: E402
from support.state import ConversationState, extract_slots, strip_accents  # noqa: E402

PRODUCTS = json.loads((config.DATA_DIR / "products.json").read_text(encoding="utf-8"))
PNAMES = [p["name"] for p in PRODUCTS] + ["tai nghe SoundMax", "laptop Nova", "Lumi X12", "Pulse Watch",
                                          "bàn phím K75", "chuột M3", "màn hình 27 inch", "Nova Tab"]
CATS = ["tai nghe", "laptop", "điện thoại", "đồng hồ thông minh", "máy tính bảng", "màn hình", "chuột", "bàn phím", "loa"]
NEEDS = ["học online", "chơi game", "làm đồ hoạ", "chạy bộ", "đi du lịch", "văn phòng", "dựng phim", "nghe nhạc"]
BUDGETS = ["2 triệu", "5 triệu", "10 triệu", "15tr", "20 triệu", "25 triệu", "30tr", "dưới 1 triệu"]
FEATURES = ["sạc nhanh", "chống nước", "NFC", "bluetooth 5.3", "kết nối 2 thiết bị", "nâng cấp RAM", "màn OLED", "GPS"]
DEFECTS = ["không lên nguồn", "mất tiếng một bên", "điểm chết", "liệt phím", "sạc không vào", "loạn cảm ứng", "tự tắt nguồn"]

T = {
    "product_inquiry": [
        "tư vấn giúp mình {cat} tầm {budget}", "{pname} giá bao nhiêu vậy?", "{pname} còn hàng không shop",
        "so sánh {pname} với {pname2} giúp mình", "có {cat} nào hợp {need} không?", "{pname} có {feature} không?",
        "mình nên mua {pname} hay {pname2}?", "cách reset {pname} thế nào", "{pname} pin dùng được bao lâu",
        "{cat} nào dưới {budget} đáng mua nhất?", "shop có bán {cat} cho {need} không", "thông số {pname} thế nào",
        "{pname} dùng với {pname2} được không?", "gợi ý {cat} cho {need}, ngân sách {budget}",
    ],
    "order_status": [
        "đơn {oid} tới đâu rồi?", "kiểm tra giúp mình đơn {oid}", "bao giờ mình nhận được đơn {oid}?",
        "mã vận đơn của đơn {oid} là gì", "đơn {oid} giao chưa shop", "sđt {phone} có đơn nào đang giao không?",
        "mình đặt bằng số {phone}, kiểm tra đơn giúp", "đơn {oid} đang ở trạng thái gì",
    ],
    "order_status_noid": [
        "đơn của mình bao giờ tới?", "kiểm tra đơn hàng giúp mình", "sao hàng mãi chưa tới vậy shop",
        "mình muốn biết đơn đang giao tới đâu", "check đơn giúp mình với",
    ],
    "cancel_order": [
        "huỷ đơn {oid} giúp mình", "mình không muốn mua nữa, huỷ đơn {oid}", "cho mình hủy đơn {oid} nhé",
        "đơn {oid} đặt nhầm, huỷ giúp", "mình muốn huỷ đơn {oid} được không",
    ],
    "cancel_order_noid": ["mình muốn huỷ đơn", "hủy đơn giúp mình với", "đặt nhầm rồi, huỷ đơn được không shop"],
    "return_request": [
        "mình muốn trả {pname} trong đơn {oid}", "{pname} đơn {oid} bị {defect}, đổi giúp mình cái mới",
        "đơn {oid} mình đổi ý, muốn trả hàng, còn nguyên seal", "đơn {oid} muốn trả lại vì không vừa ý, đã mở hộp",
        "làm phiếu trả hàng cho đơn {oid} giúp mình", "hàng đơn {oid} bị lỗi, mình muốn hoàn tiền",
        "đơn {oid} trả được không, máy đã kích hoạt rồi",
    ],
    "return_request_noid": [
        "mình muốn trả hàng", "{pname} mới mua bị {defect}, muốn đổi cái khác", "mình muốn hoàn tiền",
        "hàng mới nhận bị lỗi, đổi trả kiểu gì?", "mua về không ưng, trả lại được không shop", "muốn đổi sang màu khác",
    ],
    "warranty_claim": [
        "{pname} đơn {oid} bị {defect}, còn bảo hành không?", "tạo phiếu bảo hành cho đơn {oid} giúp mình",
        "máy đơn {oid} bị {defect}, bảo hành giúp mình", "đơn {oid} còn hạn bảo hành không vậy",
        "{pname} mua ở đơn {oid} hỏng rồi, mang đi bảo hành thế nào",
    ],
    "warranty_claim_noid": [
        "{pname} của mình bị {defect}, bảo hành sao?", "máy mình bị {defect}, còn bảo hành không",
        "mình muốn bảo hành {pname}", "tai nghe mình bị {defect}, gửi bảo hành được không",
    ],
    "policy_question": [
        "chính sách đổi trả thế nào?", "laptop bảo hành bao lâu?", "phí ship bao nhiêu vậy shop", "có trả góp 0% không?",
        "xuất hoá đơn VAT được không", "đơn COD hoàn tiền mất bao lâu", "có giao hàng chủ nhật không",
        "làm rơi vỡ có được bảo hành không", "đổi ý trong bao nhiêu ngày thì được trả", "giao hàng nhanh mất mấy ngày",
        "mua gói bảo hành mở rộng thế nào", "điểm chết màn hình thì chính sách sao", "được kiểm tra hàng trước khi trả tiền không",
        "thanh toán bằng những cách nào", "tai nghe true wireless mở hộp rồi trả được không",
    ],
    "chitchat": ["chào shop", "xin chào", "cảm ơn nhé", "ok cảm ơn bạn nhiều", "hello shop", "shop ơi", "tuyệt, cám ơn"],
    "out_of_scope": [
        "thời tiết hôm nay thế nào", "viết giúp mình đoạn code python sắp xếp mảng", "kể chuyện cười đi",
        "giá vàng hôm nay bao nhiêu", "dịch câu này sang tiếng Anh: tôi yêu Hà Nội", "làm thơ tặng mình đi",
        "ai vô địch world cup 2022", "công thức nấu phở bò",
    ],
}
MULTI = [
    ("đơn {oid} bao giờ tới? với lại {pname} có {feature} không?", ["order", "product"], "order_status"),
    ("{pname} có {feature} không, mua về không ưng thì trả được không?", ["product", "policy"], "product_inquiry"),
    ("đơn {oid} mình đổi địa chỉ nhận được không?", ["order", "policy"], "order_status"),
    ("kiểm tra đơn {oid} và tư vấn thêm {cat} cho {need}", ["order", "product"], "order_status"),
    ("{pname} giá bao nhiêu, có trả góp không?", ["product", "policy"], "product_inquiry"),
    ("đơn {oid} giao chưa, nếu chưa thì phí ship nhanh bao nhiêu?", ["order", "policy"], "order_status"),
]
FOLLOW_ORDER = [  # entities có order_id, last_agents=[order]
    ("nó còn bảo hành không?", "warranty_claim"), ("mình muốn trả lại đơn đó", "return_request"),
    ("huỷ luôn đơn đó giúp mình", "cancel_order"), ("khi nào giao tới vậy", "order_status"),
    ("sản phẩm đó bị {defect}, làm phiếu bảo hành giúp", "warranty_claim"), ("trả hàng được không?", "return_request"),
]
FOLLOW_PRODUCT = [  # entities có product, last_agents=[product]
    "máy đó có {feature} không?", "còn hàng không?", "nó nặng bao nhiêu?", "có màu khác không", "so với {pname2} thì sao?",
]
PENDING_REPLY = ["{oid}", "mã đơn {oid}", "đơn {oid} nhé", "của mình là {oid}", "mã là {oid} shop ơi", "{oid} ạ"]

ASK = {"order_status": "Bạn vui lòng cho mình mã đơn hàng (dạng DHxxxx) hoặc số điện thoại đặt hàng để mình kiểm tra nhé."}
ASK_DEFAULT = "Bạn vui lòng cho mình mã đơn hàng (dạng DHxxxx) để mình hỗ trợ nhé."
CHAT_REPLY = ["Chào bạn, TechViet có thể giúp gì cho bạn ạ?", "Dạ không có gì, bạn cần hỗ trợ thêm gì cứ nhắn mình nhé!"]
OOS_REPLY = "Xin lỗi, mình chỉ hỗ trợ các vấn đề về sản phẩm, đơn hàng, đổi trả và bảo hành của TechViet."
AGENT_OF = {"product_inquiry": "product", "order_status": "order", "cancel_order": "order",
            "return_request": "policy", "warranty_claim": "policy", "policy_question": "policy"}
TASK = {
    ("product_inquiry", "product"): "Tư vấn / trả lời câu hỏi sản phẩm{p}",
    ("order_status", "order"): "Tra cứu trạng thái đơn{o}",
    ("cancel_order", "order"): "Huỷ đơn{o} theo yêu cầu của khách",
    ("return_request", "policy"): "Kiểm tra điều kiện đổi trả đơn{o} và xử lý yêu cầu",
    ("warranty_claim", "policy"): "Kiểm tra bảo hành đơn{o} và xử lý yêu cầu",
    ("policy_question", "policy"): "Trả lời câu hỏi chính sách dựa trên tài liệu",
}


def rand_oid(rng) -> str:
    return f"DH{rng.randint(1013, 9999)}"


def rand_phone(rng) -> str:
    return "0" + rng.choice("35789") + "".join(rng.choice("0123456789") for _ in range(8))


def fill(tpl: str, rng, slots: dict) -> str:
    vals = {"oid": slots.get("order_id") or rand_oid(rng), "phone": slots.get("phone") or rand_phone(rng),
            "pname": slots.get("product") or rng.choice(PNAMES), "pname2": rng.choice(PNAMES), "cat": rng.choice(CATS),
            "need": rng.choice(NEEDS), "budget": rng.choice(BUDGETS), "feature": rng.choice(FEATURES),
            "defect": rng.choice(DEFECTS)}
    text = tpl.format(**vals)
    if "{oid}" in tpl:
        slots["order_id"] = vals["oid"]
    if "{phone}" in tpl:
        slots["phone"] = vals["phone"]
    if "{pname}" in tpl:
        slots["product"] = vals["pname"]
    return text


def noisy(text: str, rng) -> str:
    """Khách hay gõ không dấu / viết hoa đầu câu."""
    r = rng.random()
    if r < 0.15:
        return strip_accents(text)
    if r < 0.5:
        return text[:1].upper() + text[1:]
    return text


def task_for(intent: str, agent: str, ents: dict) -> str:
    o = f" {ents['order_id']}" if ents.get("order_id") else (f" (SĐT {ents['phone']})" if ents.get("phone") else "")
    p = f" {ents['product']}" if ents.get("product") else ""
    return TASK.get((intent, agent), "Xử lý yêu cầu của khách").format(o=o, p=p)


def label(intent: str, agents: list[str], ents: dict, plan_intents: list[str] | None = None,
          missing: bool = False, reply: str = "") -> dict:
    ents = {k: ents.get(k, "") or "" for k in ("order_id", "phone", "product")}
    if missing:
        return normalize_decision({"intent": intent, "missing_info": ["order_id"], "entities": ents,
                                   "reply": ASK.get(intent, ASK_DEFAULT)})
    plan_intents = plan_intents or [intent] * len(agents)
    plan = [{"agent": a, "task": task_for(pi, a, ents)} for a, pi in zip(agents, plan_intents)]
    return normalize_decision({"intent": intent, "agents": agents, "plan": plan, "entities": ents, "reply": reply})


def _state(**kw) -> ConversationState:
    st = ConversationState()
    for k, v in kw.items():
        setattr(st, k, v)
    return st


def sample(rng) -> tuple[str, ConversationState, str, dict]:
    kind = rng.choices(["single", "missing", "pending", "in_state", "follow_product", "switch", "multi", "chat", "oos"],
                       weights=[30, 10, 12, 12, 6, 4, 14, 6, 6])[0]
    st = ConversationState()
    slots: dict = {}

    if kind == "single":
        intent = rng.choice(list(AGENT_OF))
        msg = fill(rng.choice(T[intent]), rng, slots)
        st.merge(extract_slots(msg))
        return kind, st, msg, label(intent, [AGENT_OF[intent]], {**st.entities, **slots})

    if kind == "missing":
        intent = rng.choice(sorted(NEEDS_ORDER))
        msg = fill(rng.choice(T[f"{intent}_noid"]), rng, slots)
        return kind, st, msg, label(intent, [], {**slots, "order_id": ""}, missing=True)

    if kind == "pending":
        intent = rng.choice(sorted(NEEDS_ORDER))
        prev = fill(rng.choice(T[f"{intent}_noid"]), rng, slots)
        st = _state(pending_intent=intent, pending_missing=["order_id"],
                    history=[{"user": prev, "assistant": ASK.get(intent, ASK_DEFAULT)}], turn=1)
        st.merge({"product": slots.get("product", "")})
        if rng.random() < 0.15:                      # khách vẫn chưa đưa mã đơn
            msg = rng.choice(["mình không nhớ mã đơn", "để mình tìm đã", "mã đơn ở đâu nhỉ"])
            return kind, st, msg, label(intent, [], st.entities, missing=True)
        msg = fill(rng.choice(PENDING_REPLY), rng, slots)
        st.merge(extract_slots(msg))
        return kind, st, msg, label(intent, [AGENT_OF[intent]], st.entities)

    if kind == "in_state":
        oid = rand_oid(rng)
        prev = f"đơn {oid} của mình gồm những gì?"
        st = _state(last_agents=["order"], history=[{"user": prev, "assistant": f"Đơn {oid} gồm {rng.choice(PNAMES)}, đã giao."}], turn=1)
        st.merge({"order_id": oid})
        tpl, intent = rng.choice(FOLLOW_ORDER)
        msg = fill(tpl, rng, slots)
        return kind, st, msg, label(intent, [AGENT_OF[intent]], st.entities)

    if kind == "follow_product":
        pname = rng.choice(PNAMES)
        st = _state(last_agents=["product"], turn=1,
                    history=[{"user": f"tư vấn {pname} giúp mình", "assistant": f"{pname} là lựa chọn tốt..."}])
        st.merge({"product": pname})
        msg = fill(rng.choice(FOLLOW_PRODUCT), rng, {})
        return kind, st, msg, label("product_inquiry", ["product"], st.entities)

    if kind == "switch":
        intent = rng.choice(sorted(NEEDS_ORDER))
        prev = fill(rng.choice(T[f"{intent}_noid"]), rng, {})
        st = _state(pending_intent=intent, pending_missing=["order_id"], turn=1,
                    history=[{"user": prev, "assistant": ASK.get(intent, ASK_DEFAULT)}])
        new = rng.choice(["product_inquiry", "policy_question"])
        msg = fill(rng.choice(T[new]), rng, slots)
        return kind, st, msg, label(new, [AGENT_OF[new]], {**st.entities, **slots})

    if kind == "multi":
        tpl, agents, first = rng.choice(MULTI)
        msg = fill(tpl, rng, slots)
        st.merge(extract_slots(msg))
        second = {"product": "product_inquiry", "policy": "policy_question", "order": "order_status"}[agents[1]]
        return kind, st, msg, label("multi", agents, {**st.entities, **slots}, plan_intents=[first, second])

    if kind == "chat":
        msg = rng.choice(T["chitchat"])
        thanks = any(w in msg for w in ("cảm ơn", "cám ơn"))
        return kind, st, msg, label("chitchat", [], {}, reply=CHAT_REPLY[1 if thanks else 0])

    msg = rng.choice(T["out_of_scope"])
    return kind, st, msg, label("out_of_scope", [], {}, reply=OOS_REPLY)


def to_record(kind: str, st: ConversationState, msg: str, target: dict) -> dict:
    user = build_router_user(st.router_view(), msg)
    return {"kind": kind, "user_message": msg, "state": st.router_view(),
            "prompt": [{"role": "system", "content": ROUTER_SYSTEM}, {"role": "user", "content": user}],
            "completion": [{"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}],
            "target": target}


def paraphrase(records: list[dict], n: int, rng) -> list[dict]:
    """Viết lại tin nhắn khách bằng Claude Haiku 5.5, giữ nguyên mã đơn / SĐT / tên sản phẩm."""
    from support.llm import Claude

    llm = Claude(model="claude-haiku-5-5", effort="low", max_tokens=2000)
    schema = {"type": "object", "properties": {"paraphrases": {"type": "array", "items": {"type": "string"}}},
              "required": ["paraphrases"], "additionalProperties": False}
    out = []
    for r in rng.sample(records, min(n, len(records))):
        res = llm.structured("Viết lại câu của khách hàng theo 2 cách khác, văn nói tự nhiên của người Việt, "
                             "giữ nguyên ý định, mã đơn, số điện thoại, tên sản phẩm. Không thêm thông tin.",
                             r["user_message"], schema)
        for p in res["paraphrases"][:2]:
            if extract_slots(p) != extract_slots(r["user_message"]):
                continue                     # paraphrase làm mất / đổi slot -> bỏ
            out.append({**r, "kind": r["kind"] + "+para", "user_message": p,
                        "prompt": [r["prompt"][0], {"role": "user", "content": build_router_user(r["state"], p)}]})
    print(f"paraphrase: +{len(out)} mẫu")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000, help="số mẫu train")
    ap.add_argument("--val", type=int, default=300)
    ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--paraphrase", type=int, default=0, help="số mẫu train đem paraphrase bằng Claude Haiku")
    ap.add_argument("--seed", type=int, default=39)
    a = ap.parse_args()

    config.TRAIN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    splits = {"train": (a.n, a.seed), "val": (a.val, a.seed + 1), "test": (a.test, a.seed + 2)}
    seen: set[str] = set()
    for split, (n, seed) in splits.items():
        rng = random.Random(seed)
        recs = []
        while len(recs) < n:
            kind, st, msg, target = sample(rng)
            msg = noisy(msg, rng)
            key = st.router_view() + msg
            if key in seen:                         # không trùng giữa các split
                continue
            seen.add(key)
            recs.append(to_record(kind, st, msg, target))
        if split == "train" and a.paraphrase:
            recs += paraphrase(recs, a.paraphrase, rng)
        path = config.TRAIN_DATA_DIR / f"sft_{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        kinds: dict[str, int] = {}
        for r in recs:
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        print(f"{split}: {len(recs)} mẫu -> {path}  {kinds}")


if __name__ == "__main__":
    main()
