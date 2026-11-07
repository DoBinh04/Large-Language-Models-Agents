"""Test không cần API key / internet:  python -m tests.test_offline

Dùng một FakeLLM trả về các lượt ReAct dựng sẵn để kiểm tra vòng lặp agent,
parser, tool calculate/get_current_time và bộ chấm điểm.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from react_agent.agent import ReActAgent
from react_agent.checks import check_answer, extract_numbers
from react_agent.llm import LLMResponse
from react_agent.parser import ParseError, parse
from react_agent.tools import TOOLS, calculate, get_current_time


class FakeLLM:
    model = "fake"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.seen = []

    def complete(self, system, messages, stop=None):
        self.seen.append([dict(m) for m in messages])
        text = self.outputs.pop(0)
        return LLMResponse(text=text, content=[{"type": "text", "text": text}],
                           stop_reason="stop_sequence", model="fake", input_tokens=10, output_tokens=5)


def test_calculate():
    assert calculate("2 + 3 * 4") == "14"
    assert calculate("10000000 * 1.065**5").startswith("13700866.")
    assert calculate("sqrt(16) + max(1, 2)") == "6"
    assert calculate("1,000,000 / 4") == "250000"
    assert calculate("2^10") == "1024"
    for bad in ["__import__('os')", "open('x')", "2**100000"]:
        assert TOOLS["calculate"].run(expr=bad).startswith("ERROR")


def test_time():
    d = json.loads(get_current_time("Asia/Tokyo"))
    assert d["utc_offset"] == "+0900"
    assert get_current_time("Mars/Olympus").startswith("ERROR")


def test_parser():
    s = parse('Thought: cần tính\nAction: calculate\nAction Input: {"expr": "1+1"}')
    assert s.action == "calculate" and s.action_input == {"expr": "1+1"}
    s = parse("Thought: x\nAction: search_web\nAction Input: population of Japan", {"search_web": "query"})
    assert s.action_input == {"query": "population of Japan"}
    s = parse('Thought: done\nFinal Answer: 42')
    assert s.is_final and s.final_answer == "42"
    s = parse('Action: calculate\nAction Input: ```json\n{"expr": "{1}"}\n``` thừa')
    assert s.action_input == {"expr": "{1}"}
    try:
        parse("blah")
        raise AssertionError("phải lỗi")
    except ParseError:
        pass


def test_agent_loop():
    llm = FakeLLM([
        'Thought: tính trước\nAction: calculate\nAction Input: {"expr": "6*7"}\n',
        "chữ linh tinh không đúng format",
        "Thought: đã có kết quả\nFinal Answer: 6*7 = 42",
    ])
    res = ReActAgent(llm, verbose=False).run("6 nhân 7?")
    assert res.status == "success" and res.answer == "6*7 = 42"
    assert res.tools_used == ["calculate"]
    assert res.steps[0].observation == "42"
    assert res.steps[1].error                                     # lỗi format được báo lại cho model
    last = llm.seen[-1]
    assert [m["role"] for m in last] == ["user", "assistant", "user", "assistant", "user"]
    assert last[2]["content"] == "Observation: 42"


def test_checks():
    assert 13700866.6 in extract_numbers("13.700.866,6 VND")
    assert 13700866.6 in extract_numbers("13,700,866.6 VND")
    assert check_answer("Argentina vô địch", {"type": "contains", "value": "argentina"})[0]
    assert check_answer("khoảng 8,3 phút", {"type": "number_close", "value": 8.32, "tolerance": 0.05})[0]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
