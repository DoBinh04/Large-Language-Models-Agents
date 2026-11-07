"""LLM Judge: chấm hệ thống Agents hoàn thành task hay không (PASS / FAIL).

Judge thấy: tiêu chí thành công của kịch bản, toàn bộ hội thoại, các tool call kèm kết quả (rút gọn)
— để phân biệt "trả lời đúng nhờ dữ liệu thật" với "đoán trúng". Structured output (JSON Schema)
buộc judge ghi lý do TRƯỚC verdict (reasoning-first), và đánh giá từng tiêu chí con.

Quy tắc chấm (trong prompt): FAIL nếu sai bất kỳ con số / ngày / mã nào, bịa thông tin, thực hiện hành động
ghi (huỷ đơn, tạo phiếu) khi khách không yêu cầu hoặc không đủ điều kiện, hoặc bỏ sót ý bắt buộc.
Văn phong, độ dài không ảnh hưởng verdict.
"""
from __future__ import annotations

import json

import config
from support.llm import Claude
from support.tracing import NULL_TRACER, Tracer

JUDGE_SYSTEM = """Bạn là giám khảo chấm hệ thống chăm sóc khách hàng TechViet. Hôm nay là 2026-10-29.
Cho: TIÊU CHÍ thành công, HỘI THOẠI (khách & trợ lý), và TOOL CALLS hệ thống đã thực hiện kèm kết quả.

Chấm PASS chỉ khi câu trả lời cuối cùng của mỗi lượt đáp ứng TẤT CẢ ý bắt buộc trong tiêu chí.
Chấm FAIL nếu có một trong các lỗi:
- sai hoặc thiếu con số, ngày, mã đơn, mã phiếu, % hoàn tiền mà tiêu chí yêu cầu;
- khẳng định thông tin trái với kết quả tool hoặc không có trong kết quả tool (bịa);
- thực hiện hành động ghi (huỷ đơn, tạo phiếu) khi khách không yêu cầu hoặc không đủ điều kiện, hoặc nói đã làm mà không có tool call tương ứng;
- không hỏi lại khi tiêu chí yêu cầu hỏi lại.
Không trừ điểm vì văn phong, độ dài, hay thông tin bổ sung đúng."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "criteria_checks": {"type": "array", "items": {
            "type": "object",
            "properties": {"criterion": {"type": "string"}, "met": {"type": "boolean"}, "evidence": {"type": "string"}},
            "required": ["criterion", "met", "evidence"], "additionalProperties": False}},
        "reasoning": {"type": "string"},
        "verdict": {"type": "string", "enum": ["PASS", "FAIL"]},
    },
    "required": ["criteria_checks", "reasoning", "verdict"],
    "additionalProperties": False,
}


class Judge:
    def __init__(self, model: str = config.JUDGE_MODEL, effort: str = config.JUDGE_EFFORT, tracer: Tracer = NULL_TRACER):
        self.llm = Claude(model=model, effort=effort, tracer=tracer, name="judge", max_tokens=6000)
        self.label = f"{model}@{effort}"

    def grade(self, criteria: str, transcript: list[dict], tool_trace: list[dict]) -> dict:
        convo = "\n".join(f"[Lượt {i + 1}] KHÁCH: {t['user']}\n[Lượt {i + 1}] TRỢ LÝ: {t['assistant']}"
                          for i, t in enumerate(transcript))
        tools = "\n".join(f"- {c['name']}({json.dumps(c.get('args', {}), ensure_ascii=False)}) -> "
                          f"{json.dumps(c.get('result', ''), ensure_ascii=False)[:800]}" for c in tool_trace) or "(không có)"
        user = f"TIÊU CHÍ:\n{criteria}\n\nHỘI THOẠI:\n{convo}\n\nTOOL CALLS:\n{tools}"
        return self.llm.structured(JUDGE_SYSTEM, user, JUDGE_SCHEMA)
