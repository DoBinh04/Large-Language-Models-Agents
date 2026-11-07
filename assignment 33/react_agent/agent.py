"""Vòng lặp ReAct.

    messages = [user: Question]
    lặp tối đa max_steps lần:
        LLM sinh "Thought + Action + Action Input"   (dừng ở stop sequence "Observation:")
        parse -> nếu Final Answer: trả về
        chạy tool -> thêm  user: "Observation: <kết quả>"  vào hội thoại

Hội thoại chỉ được NỐI THÊM (append-only): lượt assistant được gửi lại nguyên vẹn
(cả thinking blocks) nên prompt cache và preserved thinking của model không bị phá.

_call_llm và _run_tool là 2 điểm móc (hook) có thể override, ví dụ ở assignment 3.7
để thêm tracing mà không phải sửa file này.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List

from .llm import ClaudeClient, LLMResponse
from .parser import ParseError, parse
from .prompt import build_system_prompt
from .tools import TOOLS, Tool

STOP_SEQUENCES = ["Observation:"]


@dataclass
class Step:
    index: int
    thought: str = ""
    action: str | None = None
    action_input: dict = field(default_factory=dict)
    observation: str | None = None
    final_answer: str | None = None
    raw_output: str = ""
    error: str | None = None
    llm_latency_ms: float = 0.0
    tool_latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class AgentResult:
    question: str
    answer: str | None
    status: str                         # success | max_steps | refusal | error
    steps: List[Step]
    model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int

    @property
    def tools_used(self) -> List[str]:
        return [s.action for s in self.steps if s.action]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tools_used"] = self.tools_used
        return d


class ReActAgent:
    def __init__(self, llm: ClaudeClient | None = None, tools: Dict[str, Tool] | None = None,
                 max_steps: int = 10, verbose: bool = True,
                 printer: Callable[[str], None] = print):
        self.llm = llm or ClaudeClient()
        self.tools = tools or TOOLS
        self.max_steps = max_steps
        self.verbose = verbose
        self.printer = printer
        self.system = build_system_prompt(self.tools)
        # tham số đầu tiên của mỗi tool: dùng khi model viết Action Input là chuỗi trần
        self._first_arg = {name: next(iter(t.args), None) for name, t in self.tools.items()}

    # ---- hooks ---------------------------------------------------------------
    def _call_llm(self, messages: List[dict], step_index: int) -> LLMResponse:
        return self.llm.complete(self.system, messages, stop=STOP_SEQUENCES)

    def _run_tool(self, name: str, args: dict, step_index: int) -> str:
        tool = self.tools.get(name)
        if tool is None:
            return f"ERROR: không có tool '{name}'. Các tool hợp lệ: {list(self.tools)}"
        return tool.run(**args)

    # ---- main loop -------------------------------------------------------------
    def _log(self, text: str) -> None:
        if self.verbose:
            self.printer(text)

    def run(self, question: str) -> AgentResult:
        t_start = time.perf_counter()
        messages: List[dict] = [{"role": "user", "content": f"Question: {question}"}]
        steps: List[Step] = []
        answer, status = None, "max_steps"
        self._log(f"\n=== Question: {question}")

        for i in range(1, self.max_steps + 1):
            step = Step(index=i)
            steps.append(step)

            resp = self._call_llm(messages, i)
            step.raw_output = resp.text
            step.llm_latency_ms = resp.latency_ms
            step.input_tokens, step.output_tokens = resp.input_tokens, resp.output_tokens
            messages.append({"role": "assistant", "content": resp.content})

            if resp.stop_reason == "refusal":
                status, step.error = "refusal", "model refused"
                break

            try:
                parsed = parse(resp.text, self._first_arg)
            except ParseError as e:
                step.error = str(e)
                step.observation = (f"ERROR: {e}. Hãy trả lời đúng format: "
                                    "'Thought: ...' rồi 'Action: ...' + 'Action Input: {...}', "
                                    "hoặc 'Thought: ...' rồi 'Final Answer: ...'.")
                self._log(f"[{i}] PARSE ERROR: {e}\n    raw: {resp.text[:300]!r}")
                messages.append({"role": "user", "content": f"Observation: {step.observation}"})
                continue

            step.thought = parsed.thought
            self._log(f"[{i}] Thought: {parsed.thought}")
            if parsed.is_final:
                step.final_answer = answer = parsed.final_answer
                status = "success"
                self._log(f"[{i}] Final Answer: {answer}")
                break

            step.action, step.action_input = parsed.action, parsed.action_input
            self._log(f"[{i}] Action: {parsed.action} {parsed.action_input}")
            t0 = time.perf_counter()
            step.observation = self._run_tool(parsed.action, parsed.action_input, i)
            step.tool_latency_ms = (time.perf_counter() - t0) * 1000
            self._log(f"[{i}] Observation: {step.observation[:500]}")
            messages.append({"role": "user", "content": f"Observation: {step.observation}"})

        return AgentResult(
            question=question, answer=answer, status=status, steps=steps, model=self.llm.model,
            latency_ms=(time.perf_counter() - t_start) * 1000,
            input_tokens=sum(s.input_tokens for s in steps),
            output_tokens=sum(s.output_tokens for s in steps),
        )
