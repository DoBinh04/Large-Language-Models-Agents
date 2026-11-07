"""3 agent chuyên trách: Product / Order / Policy. Mỗi agent:

- công bố AgentCard (A2A) với skills của mình,
- nhận A2A Message: text = chỉ dẫn từ Orchestrator, data = {entities, user_message, history, previous_results},
- chạy vòng tool use với đúng nhóm tool của mình (qua LocalTools hoặc MCP),
- trả về A2A Task: completed (artifact chứa answer + tool_calls) hoặc input-required (cần khách bổ sung).
"""
from __future__ import annotations

import json

import config
from support import a2a
from support.llm import Claude
from support.tools import TODAY, WRITE_TOOLS, tools_for
from support.tracing import NULL_TRACER, Tracer

NEED_INPUT = "[CẦN THÔNG TIN]"

COMMON_RULES = f"""Hôm nay là {TODAY.isoformat()}. Bạn trả lời khách bằng tiếng Việt, thân thiện, ngắn gọn.
- Chỉ dùng thông tin lấy từ tool; không bịa giá, ngày, chính sách. Không có dữ liệu thì nói rõ.
- Giá tiền viết dạng 1.290.000đ.
- Tool ghi dữ liệu (huỷ đơn, tạo phiếu) chỉ gọi khi khách đã nói rõ muốn làm việc đó.
- Nếu thiếu thông tin bắt buộc mà tool không tự suy ra được, bắt đầu câu trả lời bằng "{NEED_INPUT}" rồi hỏi đúng thông tin cần.
- Không nhắc tới tên tool, agent hay hệ thống nội bộ trong câu trả lời."""

PROMPTS = {
    "product": """Bạn là Product Agent của TechViet: tư vấn sản phẩm.
- Tìm bằng search_products, xem chi tiết bằng get_product_details, so sánh bằng compare_products.
- Câu hỏi cách dùng / thông số kỹ thuật chi tiết / tương thích: tra search_knowledge_base.
- Gợi ý tối đa 3 sản phẩm phù hợp ngân sách và nhu cầu; ưu tiên còn hàng, nói rõ nếu hết hàng.""",
    "order": """Bạn là Order Agent của TechViet: tra cứu và xử lý đơn hàng.
- Tra đơn bằng lookup_order (mã DHxxxx); không có mã nhưng có SĐT thì list_orders_by_phone.
- Báo trạng thái, ngày giao / dự kiến, mã vận đơn, sản phẩm trong đơn.
- Huỷ đơn chỉ bằng cancel_order khi khách yêu cầu huỷ.""",
    "policy": """Bạn là Policy Agent của TechViet: chính sách đổi trả, bảo hành, vận chuyển, thanh toán.
- Câu hỏi chung về chính sách: search_policy, trả lời dựa trên đoạn tài liệu tìm được.
- Đổi trả một đơn cụ thể: check_return_eligibility (reason defect nếu sản phẩm lỗi, change_of_mind nếu đổi ý;
  condition sealed / opened_unactivated / activated theo lời khách). Nếu khách muốn trả và đủ điều kiện -> create_return_request.
- Sản phẩm hỏng: check_warranty; còn trong 30 ngày đầu thì lỗi NSX được đổi mới (check_return_eligibility reason defect);
  quá 30 ngày mà còn bảo hành -> create_warranty_ticket khi khách muốn bảo hành.
- Luôn nêu số tiền / % hoàn, thời gian hoàn tiền, mã phiếu và bước tiếp theo nếu đã tạo phiếu.""",
}

SKILLS = {
    "product": [a2a.AgentSkill("product_search", "Tìm & tư vấn sản phẩm", "Gợi ý theo nhu cầu, ngân sách"),
                a2a.AgentSkill("product_compare", "So sánh sản phẩm", "So sánh thông số, giá, bảo hành"),
                a2a.AgentSkill("product_howto", "Hướng dẫn sử dụng", "Tra tài liệu hướng dẫn / FAQ")],
    "order": [a2a.AgentSkill("order_lookup", "Tra cứu đơn", "Trạng thái, vận chuyển, sản phẩm trong đơn"),
              a2a.AgentSkill("order_cancel", "Huỷ đơn", "Huỷ đơn processing")],
    "policy": [a2a.AgentSkill("policy_qa", "Hỏi đáp chính sách", "Đổi trả, bảo hành, vận chuyển, thanh toán (RAG)"),
               a2a.AgentSkill("return_process", "Xử lý đổi trả", "Kiểm tra điều kiện, tạo phiếu RMA"),
               a2a.AgentSkill("warranty_process", "Xử lý bảo hành", "Kiểm tra hạn, tạo phiếu BH")],
}


class SpecialistAgent:
    def __init__(self, name: str, backend, model: str = config.BASE_MODEL, effort: str = config.BASE_EFFORT,
                 tracer: Tracer = NULL_TRACER):
        self.name = name
        self.backend = backend
        self.tracer = tracer
        self.tool_names = tools_for(name)
        self.llm = Claude(model=model, effort=effort, tracer=tracer, name=f"agent.{name}")
        self.system = PROMPTS[name] + "\n\n" + COMMON_RULES
        self.card = a2a.AgentCard(name=f"{name}_agent", description=PROMPTS[name].splitlines()[0],
                                  skills=SKILLS[name], url=f"inproc://{name}")

    def _execute(self, tool: str, args: dict) -> dict:
        with self.tracer.span(f"tool.{tool}", kind="tool", tool=tool, args=args, agent=self.name,
                              backend=self.backend.backend, write=tool in WRITE_TOOLS) as sp:
            if tool not in self.tool_names:                     # agent chỉ được dùng tool của mình
                out = {"error": f"Tool {tool} không thuộc {self.name} agent"}
            else:
                out = self.backend.call(tool, args)
            sp.set(result=out, is_error=isinstance(out, dict) and "error" in out)
            return out

    def handle(self, message: dict) -> dict:
        instruction = a2a.get_text(message)
        data = a2a.get_data(message)
        with self.tracer.span(f"agent.{self.name}", kind="agent", agent=self.name, instruction=instruction) as sp:
            context = {k: data.get(k) for k in ("entities", "history", "previous_results") if data.get(k)}
            user = (f"Tin nhắn của khách: {data.get('user_message', '')}\n"
                    f"Chỉ dẫn từ điều phối viên: {instruction}\n"
                    f"Ngữ cảnh: {json.dumps(context, ensure_ascii=False)}")
            answer, calls = self.llm.tool_loop(self.system, [{"role": "user", "content": user}],
                                               self.backend.schemas(self.tool_names), self._execute)
            state = "input-required" if answer.strip().startswith(NEED_INPUT) else "completed"
            answer = answer.replace(NEED_INPUT, "").strip()
            sp.set(answer=answer, task_state=state, n_tool_calls=len(calls))
            art = a2a.artifact(f"{self.name}_result", {"answer": answer, "tool_calls": calls}, answer)
            return a2a.make_task(message, state, answer, [art])


def build_agents(backend, model: str = config.BASE_MODEL, effort: str = config.BASE_EFFORT,
                 tracer: Tracer = NULL_TRACER, overrides: dict | None = None) -> dict[str, SpecialistAgent]:
    """overrides: {"product": ("claude-haiku-5-5", "low"), ...} để thử cấu hình model khác nhau từng agent."""
    overrides = overrides or {}
    return {n: SpecialistAgent(n, backend, *overrides.get(n, (model, effort)), tracer=tracer) for n in config.AGENTS}
