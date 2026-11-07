"""Router / Planner của Orchestrator: quyết định intent, agent nào xử lý, theo thứ tự nào, thiếu gì.

Output (RoutingDecision, JSON):
    {"intent": "...", "agents": ["order","policy"], "plan": [{"agent":"order","task":"..."}, ...],
     "missing_info": ["order_id"], "entities": {"order_id":"", "phone":"", "product":""}, "reply": ""}

Ba cài đặt cùng interface `decide(state, user_msg) -> dict`:
- APIRouter      : Claude + structured output (JSON Schema), baseline "API LLM".
- SmallLLMRouter : model 1B–3B (HF transformers), base / + LoRA SFT / + LoRA DPO.
- RuleRouter     : từ khoá, baseline rẻ nhất, dùng để so sánh và làm fallback khi JSON hỏng.

ROUTER_SYSTEM + build_router_user() là prompt DUY NHẤT, dùng cho cả train lẫn inference.
"""
from __future__ import annotations

import json
import re
import time

import config
from support.state import ConversationState
from support.tracing import NULL_TRACER, Tracer, gpu_cost

INTENTS = ["product_inquiry", "order_status", "cancel_order", "return_request", "warranty_claim",
           "policy_question", "multi", "chitchat", "out_of_scope"]
NEEDS_ORDER = {"order_status", "cancel_order", "return_request", "warranty_claim"}

ROUTER_SYSTEM = """Bạn là Router của hệ thống chăm sóc khách hàng TechViet. Đọc STATE và tin nhắn khách, trả về JSON quyết định điều phối.

Agents:
- product: tư vấn, tìm, so sánh sản phẩm, thông số, giá, tồn kho, hướng dẫn sử dụng.
- order: tra cứu trạng thái đơn, vận chuyển, huỷ đơn.
- policy: chính sách đổi trả/bảo hành/vận chuyển/thanh toán; kiểm tra điều kiện và tạo phiếu đổi trả, bảo hành.

Intent -> agents:
- product_inquiry -> [product]
- order_status, cancel_order -> [order]
- return_request, warranty_claim, policy_question -> [policy]
- multi: câu hỏi cần nhiều agent -> liệt kê đủ, theo thứ tự cần chạy
- chitchat (chào hỏi, cảm ơn) / out_of_scope (không liên quan cửa hàng) -> agents [] và viết reply ngắn

Quy tắc:
1. order_status, cancel_order, return_request, warranty_claim cần order_id. Nếu STATE và tin nhắn đều không có order_id (order_status chấp nhận phone thay order_id) -> missing_info ["order_id"], agents [], plan [], reply hỏi khách mã đơn.
2. Nếu STATE có pending_intent và tin nhắn bổ sung thông tin còn thiếu -> tiếp tục pending_intent.
3. Câu nối tiếp ngắn ("còn bảo hành thì sao?", "nó còn hàng không?") hiểu theo recent_turns và entities.
4. plan: mỗi bước một agent, task là chỉ dẫn cụ thể (kèm mã đơn / sản phẩm nếu có).
5. entities: điền order_id, phone, product (tên hoặc mã) biết được từ STATE + tin nhắn; không biết thì "".
6. reply chỉ dùng khi agents rỗng; ngược lại để "".
Chỉ trả về JSON."""

ROUTER_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": INTENTS},
        "agents": {"type": "array", "items": {"type": "string", "enum": list(config.AGENTS)}},
        "plan": {"type": "array", "items": {
            "type": "object",
            "properties": {"agent": {"type": "string", "enum": list(config.AGENTS)}, "task": {"type": "string"}},
            "required": ["agent", "task"], "additionalProperties": False}},
        "missing_info": {"type": "array", "items": {"type": "string", "enum": ["order_id"]}},
        "entities": {"type": "object",
                     "properties": {"order_id": {"type": "string"}, "phone": {"type": "string"},
                                    "product": {"type": "string"}},
                     "required": ["order_id", "phone", "product"], "additionalProperties": False},
        "reply": {"type": "string"},
    },
    "required": ["intent", "agents", "plan", "missing_info", "entities", "reply"],
    "additionalProperties": False,
}


def build_router_user(state_view: str, user_msg: str) -> str:
    return f"STATE: {state_view}\nKHÁCH: {user_msg}"


def normalize_decision(d: dict) -> dict:
    """Đưa output (nhất là của small LLM) về đúng hình dạng; agents suy ra từ plan nếu thiếu."""
    d = dict(d) if isinstance(d, dict) else {}
    intent = d.get("intent") if d.get("intent") in INTENTS else "out_of_scope"
    plan = [s for s in d.get("plan") or [] if isinstance(s, dict) and s.get("agent") in config.AGENTS]
    agents = [a for a in d.get("agents") or [] if a in config.AGENTS]
    for s in plan:
        if s["agent"] not in agents:
            agents.append(s["agent"])
    if agents and not plan:
        plan = [{"agent": a, "task": ""} for a in agents]
    ents = d.get("entities") if isinstance(d.get("entities"), dict) else {}
    missing = [m for m in d.get("missing_info") or [] if m == "order_id"]
    if missing:
        agents, plan = [], []
    return {"intent": intent, "agents": agents, "plan": plan, "missing_info": missing,
            "entities": {k: str(ents.get(k) or "") for k in ("order_id", "phone", "product")},
            "reply": str(d.get("reply") or "")}


def parse_json_obj(text: str) -> dict | None:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


# --------------------------------------------------------------------------- #
class APIRouter:
    def __init__(self, model: str = config.BASE_MODEL, effort: str = config.BASE_EFFORT, tracer: Tracer = NULL_TRACER):
        from support.llm import Claude

        self.llm = Claude(model=model, effort=effort, max_tokens=config.ROUTER_MAX_TOKENS, tracer=tracer, name="router")
        self.tracer = tracer
        self.label = f"api:{model}@{effort}"

    def decide(self, state: ConversationState, user_msg: str) -> dict:
        user = build_router_user(state.router_view(), user_msg)
        with self.tracer.span("router", kind="router", router=self.label) as sp:
            d = normalize_decision(self.llm.structured(ROUTER_SYSTEM, user, ROUTER_SCHEMA))
            sp.set(decision=d)
            return d


class SmallLLMRouter:
    """Qwen2.5-1.5B-Instruct (mặc định) + LoRA adapter (SFT hoặc DPO). Cần GPU để chạy nhanh;
    trên CPU vẫn chạy được nhưng chậm (vài giây / quyết định)."""

    def __init__(self, base_model: str = config.SMALL_BASE_MODEL, adapter: str | None = None,
                 tracer: Tracer = NULL_TRACER, load_4bit: bool = False, label: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(adapter or base_model)
        # T4 / P100 (Colab, Kaggle) không có bf16 native -> fp16; CPU -> fp32
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
        kw = {"torch_dtype": dtype}
        if load_4bit:
            from transformers import BitsAndBytesConfig

            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                            bnb_4bit_compute_dtype=dtype)
        if torch.cuda.is_available():
            kw["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(base_model, **kw)
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
        self.model = model.eval()
        self.tracer = tracer
        self.label = label or f"small:{base_model.split('/')[-1]}" + (f"+{adapter.rstrip('/').split('/')[-1]}" if adapter else "")
        self.fallback = RuleRouter()

    def generate(self, user: str, max_new_tokens: int = 320) -> tuple[str, dict]:
        msgs = [{"role": "system", "content": ROUTER_SYSTEM}, {"role": "user", "content": user}]
        enc = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        ids = enc["input_ids"].to(self.model.device)
        t0 = time.perf_counter()
        with self.torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        secs = time.perf_counter() - t0
        new = out[0, ids.shape[1]:]
        return self.tok.decode(new, skip_special_tokens=True), {
            "input_tokens": int(ids.shape[1]), "output_tokens": int(new.shape[0]),
            "gen_seconds": round(secs, 3), "cost_usd": gpu_cost(secs)}

    def decide(self, state: ConversationState, user_msg: str) -> dict:
        user = build_router_user(state.router_view(), user_msg)
        with self.tracer.span("router", kind="router", router=self.label) as sp:
            text, usage = self.generate(user)
            raw = parse_json_obj(text)
            sp.set(raw_output=text, **usage, json_valid=raw is not None)
            d = normalize_decision(raw) if raw is not None else self.fallback.decide(state, user_msg)
            sp.set(decision=d)
            return d


class RuleRouter:
    """Baseline từ khoá. Cố tình đơn giản: cho thấy vì sao cần router học được ngữ cảnh."""

    label = "rule"
    KW = {
        "cancel_order": ["huỷ đơn", "hủy đơn", "huỷ", "hủy", "cancel"],
        "warranty_claim": ["bảo hành", "hỏng", "lỗi", "không lên", "điểm chết", "sửa"],
        "return_request": ["đổi trả", "trả hàng", "hoàn tiền", "đổi ý", "trả lại", "đổi sang"],
        "order_status": ["đơn hàng", "giao", "vận đơn", "đơn của", "ship", "khi nào nhận", "tới chưa"],
        "policy_question": ["chính sách", "trả góp", "phí vận chuyển", "thanh toán", "vat", "cod"],
        "product_inquiry": ["tư vấn", "laptop", "tai nghe", "điện thoại", "giá", "còn hàng", "so sánh", "pin",
                            "đồng hồ", "màn hình", "chuột", "bàn phím", "loa", "máy tính bảng", "sạc", "gợi ý"],
        "chitchat": ["xin chào", "chào", "cảm ơn", "cám ơn", "hello", "hi "],
    }
    AGENT = {"product_inquiry": "product", "order_status": "order", "cancel_order": "order",
             "return_request": "policy", "warranty_claim": "policy", "policy_question": "policy"}

    def __init__(self, tracer: Tracer = NULL_TRACER):
        self.tracer = tracer

    def decide(self, state: ConversationState, user_msg: str) -> dict:
        low = user_msg.lower()
        hits = [i for i, kws in self.KW.items() if any(k in low for k in kws)]
        has_order = bool(state.entities.get("order_id") or state.entities.get("phone"))
        if not hits and state.pending_intent:
            hits = [state.pending_intent]
        task_hits = [h for h in hits if h in self.AGENT]
        if not task_hits:
            intent = "chitchat" if "chitchat" in hits else "out_of_scope"
            reply = "Chào bạn, TechViet có thể giúp gì cho bạn?" if intent == "chitchat" else \
                "Xin lỗi, mình chỉ hỗ trợ các vấn đề về sản phẩm, đơn hàng, đổi trả và bảo hành của TechViet."
            return normalize_decision({"intent": intent, "reply": reply, "entities": state.entities})
        agents = list(dict.fromkeys(self.AGENT[h] for h in task_hits))
        intent = task_hits[0] if len(agents) == 1 else "multi"
        if any(h in NEEDS_ORDER for h in task_hits) and not has_order:
            return normalize_decision({"intent": intent, "missing_info": ["order_id"], "entities": state.entities,
                                       "reply": "Bạn vui lòng cho mình mã đơn hàng (dạng DHxxxx) nhé."})
        return normalize_decision({"intent": intent, "agents": agents, "entities": state.entities,
                                   "plan": [{"agent": a, "task": user_msg} for a in agents]})


def make_router(kind: str, tracer: Tracer = NULL_TRACER, model: str | None = None, effort: str | None = None):
    """kind: api | rule | small | sft | dpo"""
    if kind == "api":
        return APIRouter(model or config.BASE_MODEL, effort or config.BASE_EFFORT, tracer)
    if kind == "rule":
        return RuleRouter(tracer)
    adapter = {"small": None, "sft": str(config.SFT_ADAPTER), "dpo": str(config.DPO_ADAPTER)}[kind]
    return SmallLLMRouter(adapter=adapter, tracer=tracer, label=f"small-{kind}")
