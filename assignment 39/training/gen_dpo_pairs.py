"""Sinh cặp preference (prompt, chosen, rejected) cho DPO, nhắm đúng các lỗi điều phối hay gặp.

chosen  = quyết định đúng (nhãn SFT).
rejected lấy từ 2 nguồn:
  1. Nhiễu có chủ đích theo loại mẫu (luôn có, không cần GPU):
     - state_unaware   : mã đơn đã có trong STATE mà vẫn hỏi lại khách
     - no_ask          : thiếu mã đơn mà vẫn route thẳng tới agent (đoán / bịa)
     - ignore_pending  : khách đưa mã đơn để tiếp tục đổi trả, router lại coi là tra đơn
     - drop_agent      : câu multi chỉ route 1 agent
     - extra_agent     : gọi thêm agent không cần (tốn chi phí)
     - wrong_agent     : nhầm product <-> policy <-> order
     - chat_routed     : chào hỏi / ngoài phạm vi mà vẫn gọi agent
     - bad_format      : trả lời văn xuôi thay vì JSON
  2. On-policy (tuỳ chọn --sft-adapter, cần GPU): sample từ model SFT, output nào sai so với nhãn
     thì làm rejected. Đây là lỗi model thực sự mắc, DPO sửa trúng nhất.

    python -m training.gen_dpo_pairs
    python -m training.gen_dpo_pairs --sft-adapter training/checkpoints/sft --onpolicy-n 800
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from eval.metrics import routing_correct  # noqa: E402
from support.router import NEEDS_ORDER, normalize_decision, parse_json_obj  # noqa: E402

CONFUSE = {"policy": ["product", "order"], "product": ["policy"], "order": ["policy"]}
ASK = "Bạn vui lòng cho mình mã đơn hàng (dạng DHxxxx) để mình hỗ trợ nhé."


def _dump(d: dict) -> str:
    return json.dumps(d, ensure_ascii=False)


def _with_agents(t: dict, agents: list[str]) -> dict:
    r = copy.deepcopy(t)
    r["agents"] = agents
    r["plan"] = [{"agent": a, "task": next((s["task"] for s in t["plan"] if s["agent"] == a), "Xử lý yêu cầu của khách")}
                 for a in agents]
    r["missing_info"], r["reply"] = [], ""
    return r


def perturb(rec: dict, rng) -> list[tuple[str, str]]:
    t, kind = rec["target"], rec["kind"].replace("+para", "")
    out: list[tuple[str, str]] = []
    if t["missing_info"]:
        agent = {"order_status": "order", "cancel_order": "order"}.get(t["intent"], "policy")
        out.append(("no_ask", _dump(_with_agents(t, [agent]))))
    elif t["intent"] in ("chitchat", "out_of_scope"):
        out.append(("chat_routed", _dump(_with_agents({**t, "intent": "product_inquiry"}, ["product"]))))
    else:
        if kind in ("in_state", "pending") and t["intent"] in NEEDS_ORDER:
            r = copy.deepcopy(t)
            r.update(agents=[], plan=[], missing_info=["order_id"], reply=ASK)
            r["entities"]["order_id"] = ""
            out.append(("state_unaware", _dump(r)))
        if kind == "pending" and t["intent"] != "order_status":
            r = _with_agents({**t, "intent": "order_status"}, ["order"])
            out.append(("ignore_pending", _dump(r)))
        if len(t["agents"]) > 1:
            out.append(("drop_agent", _dump(_with_agents(t, t["agents"][:1]))))
        else:
            a = t["agents"][0]
            wrong = rng.choice(CONFUSE[a])
            out.append(("wrong_agent", _dump(_with_agents(t, [wrong]))))
            if rng.random() < 0.4:
                out.append(("extra_agent", _dump(_with_agents({**t, "intent": "multi"}, [a, wrong]))))
    if rng.random() < 0.08:
        agents = ", ".join(t["agents"]) or "không agent nào"
        out.append(("bad_format", f"Mình sẽ chuyển yêu cầu này cho {agents} xử lý nhé."))
    return out


def onpolicy_rejections(records: list[dict], adapter: str, n: int, k: int, rng,
                        base_model: str = config.SMALL_BASE_MODEL) -> list[dict]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(adapter)
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, device_map="auto")
    model = PeftModel.from_pretrained(base, adapter).eval()
    pairs = []
    for rec in rng.sample(records, min(n, len(records))):
        enc = tok.apply_chat_template(rec["prompt"], add_generation_prompt=True, return_tensors="pt", return_dict=True)
        ids = enc["input_ids"].to(model.device)
        with torch.no_grad():
            outs = model.generate(ids, max_new_tokens=320, do_sample=True, temperature=0.9, top_p=0.95,
                                  num_return_sequences=k, pad_token_id=tok.eos_token_id)
        seen = set()
        for o in outs:
            text = tok.decode(o[ids.shape[1]:], skip_special_tokens=True).strip()
            raw = parse_json_obj(text)
            t = rec["target"]
            ok = raw is not None and routing_correct(normalize_decision(raw), t["agents"], bool(t["missing_info"]),
                                                     ) and normalize_decision(raw)["intent"] == t["intent"]
            if not ok and text not in seen:
                seen.add(text)
                pairs.append({"prompt": rec["prompt"], "chosen": rec["completion"],
                              "rejected": [{"role": "assistant", "content": text}], "source": "onpolicy"})
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-prompt", type=int, default=2)
    ap.add_argument("--sft-adapter", default=None, help="bật on-policy negatives từ model SFT (cần GPU)")
    ap.add_argument("--base", default=config.SMALL_BASE_MODEL, help="base model của adapter SFT (phải khớp lúc train)")
    ap.add_argument("--onpolicy-n", type=int, default=600)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    for split in ("train", "val"):
        recs = [json.loads(l) for l in (config.TRAIN_DATA_DIR / f"sft_{split}.jsonl").open(encoding="utf-8")]
        pairs = []
        for rec in recs:
            cands = perturb(rec, rng)
            rng.shuffle(cands)
            for src, rej in cands[: a.max_per_prompt]:
                pairs.append({"prompt": rec["prompt"], "chosen": rec["completion"],
                              "rejected": [{"role": "assistant", "content": rej}], "source": src})
        if split == "train" and a.sft_adapter:
            pairs += onpolicy_rejections(recs, a.sft_adapter, a.onpolicy_n, a.k, rng, a.base)
        rng.shuffle(pairs)
        path = config.TRAIN_DATA_DIR / f"dpo_{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        by_src: dict[str, int] = {}
        for p in pairs:
            by_src[p["source"]] = by_src.get(p["source"], 0) + 1
        print(f"{split}: {len(pairs)} cặp -> {path}  {by_src}")


if __name__ == "__main__":
    main()
