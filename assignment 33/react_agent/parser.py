"""Parse output của LLM theo format ReAct.

Thought: ...
Action: tool_name
Action Input: {"arg": "value"}
hoặc
Thought: ...
Final Answer: ...

Parser chịu được các lỗi format hay gặp: Action Input không phải JSON (một chuỗi trần),
có ```json fence, có text thừa sau JSON, hoặc model viết cả Action lẫn Final Answer.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

_THOUGHT = re.compile(r"Thought\s*:\s*(.*?)(?=\n\s*(?:Action\s*:|Final Answer\s*:)|\Z)", re.S | re.I)
_ACTION = re.compile(r"Action\s*:\s*([A-Za-z_][\w]*)", re.I)
_ACTION_INPUT = re.compile(r"Action Input\s*:\s*(.*)", re.S | re.I)
_FINAL = re.compile(r"Final Answer\s*:\s*(.*)", re.S | re.I)


class ParseError(ValueError):
    pass


@dataclass
class ParsedStep:
    thought: str = ""
    action: str | None = None
    action_input: dict = field(default_factory=dict)
    final_answer: str | None = None

    @property
    def is_final(self) -> bool:
        return self.final_answer is not None


def _first_json_object(text: str) -> str | None:
    """Lấy object JSON đầu tiên (đếm ngoặc, bỏ qua ngoặc nằm trong chuỗi)."""
    start = text.find("{")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_action_input(raw: str, first_arg: str | None) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw).strip()
    obj_text = _first_json_object(raw)
    if obj_text is not None:
        try:
            obj = json.loads(obj_text)
        except json.JSONDecodeError as e:
            raise ParseError(f"Action Input không phải JSON hợp lệ: {e}") from e
        if not isinstance(obj, dict):
            raise ParseError("Action Input phải là một JSON object")
        return obj
    # Không có JSON: coi cả dòng đầu là giá trị của tham số đầu tiên
    value = raw.splitlines()[0].strip().strip('"').strip("'") if raw else ""
    if not value:
        return {}
    if first_arg is None:
        raise ParseError("Action Input rỗng hoặc không phải JSON")
    return {first_arg: value}


def parse(text: str, tool_first_arg: dict[str, str | None] | None = None) -> ParsedStep:
    tool_first_arg = tool_first_arg or {}
    step = ParsedStep()
    m = _THOUGHT.search(text)
    if m:
        step.thought = m.group(1).strip()

    action = _ACTION.search(text)
    final = _FINAL.search(text)
    # Nếu có cả hai, cái nào xuất hiện trước thì thắng (model hay viết Action rồi "bịa" tiếp)
    if final and (not action or final.start() < action.start()):
        step.final_answer = final.group(1).strip()
        return step
    if action:
        step.action = action.group(1)
        ai = _ACTION_INPUT.search(text, action.end())
        step.action_input = parse_action_input(ai.group(1), tool_first_arg.get(step.action)) if ai else {}
        return step
    raise ParseError("Không tìm thấy 'Action:' hoặc 'Final Answer:' trong output")
