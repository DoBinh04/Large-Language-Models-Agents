"""Import dataset Hugging Face `Tson29/techviet-cs-orchestrator-data` -> định dạng Router của project.

Vì sao cần convert?
  Dataset HF train một orchestrator kiểu *từng bước* (mỗi bước 1 JSON:
  {check, action: call_agent|ask_user|finish, agent, task, message, state_update}),
  còn Router của project quyết định *cả lượt* trong 1 JSON
  {intent, agents, plan, missing_info, entities, reply} với prompt ROUTER_SYSTEM + STATE.
  Train thẳng file `sft/train.jsonl` của HF sẽ ra model in sai schema -> normalize_decision() coi là out_of_scope.

Cách làm: đọc config `conversations` (gen/conv_*.jsonl: 300 hội thoại, mỗi lượt có các bước quyết định),
mô phỏng lại ConversationState đúng như Orchestrator (regex slot -> router -> cập nhật pending / last_agents / history),
rồi gộp các bước của một lượt thành 1 RoutingDecision:
  - các bước call_agent         -> agents + plan (mỗi agent 1 bước, task gộp theo thứ tự)
  - chỉ ask_user hỏi mã đơn      -> missing_info ["order_id"] (+ pending_intent cho lượt sau)
  - finish không gọi agent       -> chitchat (agents [], reply = câu trả lời của trợ lý)
  - ask_user hỏi thứ khác / hỏi mã đơn khi STATE đã có mã -> bỏ (không khớp quy tắc ROUTER_SYSTEM)
Output cùng format với training/gen_data.py (prompt / completion / target / kind / state ...), nên
gen_dpo_pairs.py, sft_train.py, dpo_train.py, eval/routing_eval.py dùng được nguyên trạng.

`dpo/dpo.jsonl` của HF là cặp ở mức *bước* (schema khác) nên không dùng; gen_dpo_pairs.py tự sinh lại các
kiểu lỗi tương đương (ask_known_info ~ state_unaware, guess_missing_info ~ no_ask, wrong_agent ...) + on-policy.

Mã đơn: dataset HF dùng DH1001–DH1016, trùng dải mã của data/eval_set.json -> mặc định đổi sang mã ngẫu nhiên
ngoài dải eval (nhất quán trong từng hội thoại). Tắt bằng --keep-order-ids.

    python -m training.hf_import                         # chỉ dữ liệu HF, ghi đè sft_{train,val,test}.jsonl
    python -m training.gen_data --n 1500 && python -m training.hf_import --mode append   # trộn template + HF
    python -m training.hf_import --local-dir path/to/snapshot   # đã tải sẵn (có thư mục gen/)
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from support.router import NEEDS_ORDER, normalize_decision  # noqa: E402
from support.state import ORDER_RE, ConversationState, extract_slots  # noqa: E402
from training.gen_data import AGENT_OF, to_record  # noqa: E402

REPO = "Tson29/techviet-cs-orchestrator-data"
CONV_FILE_RE = re.compile(r"(?:^|/)gen/conv_[^/]+\.jsonl$")
ORDER_TEXT_RE = re.compile(r"\b[Dd]\s?[Hh]\s?-?(\d{4})\b")
ASK_ORDER_RE = re.compile(r"mã đơn|ma don|DHxxxx|DH\s?\d|\bDH\b|số điện thoại|so dien thoai", re.I)

CHAT_KW = ("chào", "chao", "hello", "hi ", "cảm ơn", "cám ơn", "cam on", "thanks", "ok", "oke", "vâng", "dạ", "tạm biệt")
CANCEL_KW = ("huỷ", "hủy", "huy don", "cancel")
WARRANTY_KW = ("bảo hành", "bao hanh")
RETURN_KW = ("đổi trả", "trả hàng", "trả lại", "hoàn tiền", "đổi sang", "đổi mới", "đổi cái", "đổi hàng", "đổi ý",
             "muốn đổi", "muốn trả", "đổi được", "trả được", "doi tra", "tra hang", "hoan tien")
DEFECT_KW = ("hỏng", "lỗi", "không lên", "điểm chết", "bị ", "hong", "loi")


def _has(text: str, kws) -> bool:
    low = text.lower()
    return any(k in low for k in kws)


def _obj(x):
    """Cột kiểu Json trên HF đôi khi bị lưu thành chuỗi JSON."""
    if isinstance(x, str):
        try:
            return json.loads(x)
        except json.JSONDecodeError:
            return x
    return x


# --------------------------------------------------------------------------- #
def load_conversations(repo: str = REPO, local_dir: str | None = None, revision: str | None = None) -> list[dict]:
    if local_dir:
        root = Path(local_dir)
        files = sorted(root.glob("gen/conv_*.jsonl")) or sorted(root.glob("conv_*.jsonl"))
    else:
        from huggingface_hub import HfApi, hf_hub_download

        names = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset", revision=revision)
                       if CONV_FILE_RE.search(f))
        files = [Path(hf_hub_download(repo, f, repo_type="dataset", revision=revision)) for f in names]
    if not files:
        raise SystemExit(f"Không tìm thấy gen/conv_*.jsonl trong {local_dir or repo}")
    convs = []
    for p in files:
        for line in p.open(encoding="utf-8"):
            if line.strip():
                convs.append(json.loads(line))
    print(f"Đọc {len(convs)} hội thoại từ {len(files)} file: {[p.name for p in files]}")
    return convs


def remap_order_ids(conv: dict, rng: random.Random, used: set[str]) -> dict:
    """Đổi mọi mã DHxxxx trong hội thoại sang mã ngoài dải eval (DH1001–DH1099), nhất quán trong hội thoại."""
    mapping: dict[str, str] = {}

    def sub(m: re.Match) -> str:
        num = m.group(1)
        if not ("1000" <= num <= "1099"):
            return f"DH{num}"
        if num not in mapping:
            new = f"DH{rng.randint(1100, 9999)}"
            while new in used:
                new = f"DH{rng.randint(1100, 9999)}"
            used.add(new)
            mapping[num] = new
        return mapping[num]

    return json.loads(ORDER_TEXT_RE.sub(sub, json.dumps(conv, ensure_ascii=False)))


# --------------------------------------------------------------------------- #
def category_intent(cat: str) -> str:
    c = (cat or "").lower()
    for key, intent in (("cancel", "cancel_order"), ("warranty", "warranty_claim"), ("return", "return_request"),
                        ("order", "order_status"), ("product", "product_inquiry"), ("policy", "policy_question"),
                        ("chit", "chitchat"), ("greet", "chitchat"), ("small", "chitchat"),
                        ("oos", "out_of_scope"), ("out_of_scope", "out_of_scope")):
        if key in c:
            return intent
    return ""


def infer_intent(agent: str, text: str, has_order: bool) -> str:
    if agent == "product":
        return "product_inquiry"
    if agent == "order":
        return "cancel_order" if _has(text, CANCEL_KW) else "order_status"
    if _has(text, WARRANTY_KW):
        return "warranty_claim" if has_order else "policy_question"
    if _has(text, RETURN_KW):
        return "return_request" if has_order else "policy_question"
    return "policy_question"


def infer_need_order(text: str) -> str:
    """Intent cần mã đơn khi khách chưa đưa mã (lượt ask_user)."""
    if _has(text, CANCEL_KW):
        return "cancel_order"
    if _has(text, RETURN_KW):
        return "return_request"
    if _has(text, WARRANTY_KW) or _has(text, DEFECT_KW):
        return "warranty_claim"
    return "order_status"


def turn_entities(st: ConversationState, decisions: list[dict]) -> dict:
    ents = dict(st.entities)
    for d in decisions:
        su = _obj(d.get("state_update")) or {}
        if not isinstance(su, dict):
            continue
        if su.get("order_id") and (m := ORDER_RE.search(str(su["order_id"]))):
            ents["order_id"] = f"DH{m.group(1)}"
        if su.get("product"):
            ents["product"] = str(su["product"])
    return ents


def kind_of(view: ConversationState, user: str, t: dict) -> str:
    if t["missing_info"]:
        return "pending" if view.pending_intent else "missing"
    if not t["agents"]:
        return "chat" if t["intent"] == "chitchat" else "oos"
    if len(t["agents"]) > 1:
        return "multi"
    if view.pending_intent:
        return "pending" if t["intent"] == view.pending_intent else "switch"
    if t["intent"] in NEEDS_ORDER and view.entities.get("order_id") and not ORDER_RE.search(user):
        return "in_state"
    if t["agents"] == ["product"] and view.last_agents == ["product"] and view.history:
        return "follow_product"
    return "single"


def convert_turn(view: ConversationState, user: str, decisions: list[dict], assistant: str,
                 cat_intent: str) -> tuple[dict | None, str, bool]:
    """-> (target | None nếu bỏ, lý do, có ask_user sau khi đã gọi agent)."""
    ents = turn_entities(view, decisions)
    has_order = bool(ents.get("order_id") or ents.get("phone"))
    calls = [d for d in decisions if d.get("action") == "call_agent" and d.get("agent") in config.AGENTS]
    asks = [d for d in decisions if d.get("action") == "ask_user"]

    if calls:
        tasks: dict[str, list[str]] = {}
        for d in calls:
            tasks.setdefault(d["agent"], [])
            if d.get("task") and d["task"] not in tasks[d["agent"]]:
                tasks[d["agent"]].append(d["task"])
        agents = list(tasks)
        text = user + " " + " ".join(" ".join(v) for v in tasks.values())
        if len(agents) > 1:
            intent = "multi"
        elif view.pending_intent and AGENT_OF.get(view.pending_intent) == agents[0]:
            intent = view.pending_intent                                   # quy tắc 2: tiếp tục pending
        elif AGENT_OF.get(cat_intent) == agents[0] and (cat_intent not in NEEDS_ORDER or has_order):
            intent = cat_intent
        else:
            intent = infer_intent(agents[0], text, has_order)
        plan = [{"agent": a, "task": "; ".join(ts)} for a, ts in tasks.items()]
        target = normalize_decision({"intent": intent, "agents": agents, "plan": plan, "entities": ents})
        return target, "ok", bool(asks)

    if asks:
        msg = asks[0].get("message") or ""
        if not ASK_ORDER_RE.search(msg):
            return None, "ask_other", False                   # schema router chỉ có missing order_id
        if has_order:
            return None, "ask_with_order", False              # vd. mã sai -> ngoài quy tắc 1, để agent xử lý
        if view.pending_intent:
            intent = view.pending_intent
        elif cat_intent in NEEDS_ORDER:
            intent = cat_intent
        else:
            intent = infer_need_order(user)
        target = normalize_decision({"intent": intent, "missing_info": ["order_id"], "entities": ents,
                                     "reply": msg})
        return target, "ok", False

    if _has(user, CHAT_KW) or cat_intent == "chitchat":
        intent = "chitchat"
    elif cat_intent == "out_of_scope":
        intent = "out_of_scope"
    else:
        return None, "finish_no_agent", False                # trả lời từ lịch sử: không có nhãn tương ứng
    target = normalize_decision({"intent": intent, "entities": ents, "reply": assistant})
    return target, "ok", False


def convert_conversation(conv: dict, stats: dict) -> list[dict]:
    st = ConversationState()
    cat_intent = category_intent(conv.get("category", ""))
    recs = []
    for i, turn in enumerate(_obj(conv.get("turns")) or []):
        turn = _obj(turn)
        user = str(turn.get("user") or "").strip()
        if not user:
            continue
        steps = [_obj(s) for s in (_obj(turn.get("steps")) or [])]
        decisions = [d for d in (_obj(s.get("decision")) if isinstance(s, dict) else None for s in steps)
                     if isinstance(d, dict)]
        assistant = str(turn.get("assistant") or "")

        st.merge(extract_slots(user))                    # giống Orchestrator.chat(): regex trước router
        view = copy.deepcopy(st)
        target, why, asked_after = convert_turn(view, user, decisions, assistant, cat_intent)
        stats[why] = stats.get(why, 0) + 1
        if target is not None:
            rec = to_record(kind_of(view, user, target), view, user, target)
            rec.update(source="hf", conv_id=conv.get("id", ""), turn_index=i)
            recs.append(rec)
            st.merge(target["entities"])
            if target["missing_info"]:
                st.pending_intent, st.pending_missing = target["intent"], ["order_id"]
            elif target["agents"] and asked_after:
                st.pending_intent, st.pending_missing = target["intent"], []   # agent trả input-required
            else:
                st.pending_intent, st.pending_missing = "", []
            st.last_agents = target["agents"] or st.last_agents
        else:
            st.merge(turn_entities(st, decisions))
        st.add_turn(user, assistant)
    return recs


# --------------------------------------------------------------------------- #
def write_split(path: Path, recs: list[dict], append: bool) -> int:
    seen = set()
    old = []
    if append and path.exists():
        old = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
        seen = {r["state"] + r["user_message"] for r in old}
    new = [r for r in recs if (r["state"] + r["user_message"]) not in seen]
    with path.open("w", encoding="utf-8") as f:
        for r in old + new:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(old) + len(new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--revision", default=None, help="commit / branch trên HF (để tái lập)")
    ap.add_argument("--local-dir", default=None, help="thư mục snapshot đã tải sẵn (chứa gen/conv_*.jsonl)")
    ap.add_argument("--mode", choices=["replace", "append"], default="replace",
                    help="replace: chỉ dữ liệu HF; append: nối vào sft_*.jsonl do gen_data.py sinh")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--keep-order-ids", action="store_true", help="giữ mã DH10xx gốc (có thể trùng eval set)")
    ap.add_argument("--seed", type=int, default=39)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    convs = load_conversations(a.repo, a.local_dir, a.revision)
    if not a.keep_order_ids:
        used: set[str] = set()
        convs = [remap_order_ids(c, rng, used) for c in convs]

    ids = sorted({c.get("id", str(i)) for i, c in enumerate(convs)})
    rng.shuffle(ids)                                       # chia theo hội thoại -> không lộ lượt giữa các split
    n_val, n_test = int(len(ids) * a.val_frac), int(len(ids) * a.test_frac)
    split_of = {cid: "val" for cid in ids[:n_val]}
    split_of.update({cid: "test" for cid in ids[n_val:n_val + n_test]})

    stats: dict[str, int] = {}
    by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for i, c in enumerate(convs):
        by_split[split_of.get(c.get("id", str(i)), "train")] += convert_conversation(c, stats)

    config.TRAIN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Lượt: {stats}  (chỉ 'ok' được giữ)")
    for split, recs in by_split.items():
        raw = config.TRAIN_DATA_DIR / f"hf_sft_{split}.jsonl"          # bản riêng để xem lại
        write_split(raw, recs, append=False)
        total = write_split(config.TRAIN_DATA_DIR / f"sft_{split}.jsonl", recs, append=a.mode == "append")
        kinds: dict[str, int] = {}
        for r in recs:
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        print(f"{split}: +{len(recs)} mẫu HF -> sft_{split}.jsonl (tổng {total})  {kinds}")


if __name__ == "__main__":
    main()
