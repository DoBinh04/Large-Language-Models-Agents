"""3 tool của agent: search_web(query), calculate(expr), get_current_time().

Mỗi tool là một hàm Python thường nhận keyword args và trả về str (= Observation).
Tool KHÔNG được raise ra ngoài: lỗi được trả về dưới dạng text "ERROR: ..." để
agent đọc được và tự sửa (ví dụ viết lại biểu thức).
"""
from __future__ import annotations

import ast
import json
import math
import operator
import os
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

USER_AGENT = "phase3-react-agent/1.0 (educational project)"
MAX_OBSERVATION_CHARS = 3000


@dataclass
class Tool:
    name: str
    description: str
    args: Dict[str, str]          # tên tham số -> mô tả (đưa vào prompt)
    func: Callable[..., str]

    def signature(self) -> str:
        return f"{self.name}({', '.join(self.args)})"

    def run(self, **kwargs) -> str:
        unknown = set(kwargs) - set(self.args)
        if unknown:
            return f"ERROR: {self.name} không có tham số {sorted(unknown)}. Tham số hợp lệ: {list(self.args)}"
        try:
            out = self.func(**kwargs)
        except TypeError as e:
            return f"ERROR: sai tham số cho {self.name}: {e}"
        except Exception as e:  # tool lỗi -> agent vẫn đọc được
            return f"ERROR: {type(e).__name__}: {e}"
        out = str(out)
        if len(out) > MAX_OBSERVATION_CHARS:
            out = out[:MAX_OBSERVATION_CHARS] + " ...[truncated]"
        return out


# --------------------------------------------------------------------------- #
# 1. search_web
# --------------------------------------------------------------------------- #
def _http_json(url: str, data: dict | None = None, timeout: float = 15.0):
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, headers={
        "User-Agent": USER_AGENT, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _search_tavily(query: str, n: int) -> list[dict]:
    data = _http_json("https://api.tavily.com/search", {
        "api_key": os.environ["TAVILY_API_KEY"], "query": query, "max_results": n,
        "include_answer": False})
    return [{"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
            for r in data.get("results", [])]


def _search_ddg(query: str, n: int) -> list[dict]:
    from ddgs import DDGS  # pip install ddgs (không cần API key)

    return [{"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
            for r in DDGS().text(query, max_results=n)]


def _search_wikipedia(query: str, n: int) -> list[dict]:
    """Fallback cuối: Wikipedia search API (không cần key, không cần thư viện)."""
    params = urllib.parse.urlencode({"action": "query", "list": "search", "srsearch": query,
                                     "srlimit": n, "format": "json"})
    data = _http_json(f"https://en.wikipedia.org/w/api.php?{params}")
    out = []
    for r in data.get("query", {}).get("search", []):
        snippet = r.get("snippet", "").replace('<span class="searchmatch">', "").replace("</span>", "")
        title = r.get("title", "")
        out.append({"title": title, "snippet": snippet,
                    "url": "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))})
    return out


def search_web(query: str, max_results: int = 5) -> str:
    """Thử lần lượt: Tavily (nếu có TAVILY_API_KEY) -> DuckDuckGo (ddgs) -> Wikipedia."""
    if not query or not str(query).strip():
        return "ERROR: query rỗng"
    n = max(1, min(int(max_results), 8))
    backends = []
    if os.environ.get("TAVILY_API_KEY"):
        backends.append(("tavily", _search_tavily))
    backends += [("duckduckgo", _search_ddg), ("wikipedia", _search_wikipedia)]

    errors = []
    for name, fn in backends:
        try:
            results = fn(query, n)
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")
            continue
        if results:
            lines = [f"[{i}] {r['title']}\n    {r['snippet'].strip()}\n    ({r['url']})"
                     for i, r in enumerate(results, 1)]
            return f"Kết quả tìm kiếm ({name}) cho '{query}':\n" + "\n".join(lines)
    return "Không tìm thấy kết quả. " + ("; ".join(errors) if errors else "")


# --------------------------------------------------------------------------- #
# 2. calculate — evaluator an toàn dựa trên AST (KHÔNG dùng eval())
# --------------------------------------------------------------------------- #
_BIN_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
            ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {name: getattr(math, name) for name in (
    "sqrt", "log", "log10", "log2", "exp", "sin", "cos", "tan", "asin", "acos", "atan",
    "floor", "ceil", "factorial", "radians", "degrees")}
_FUNCS.update({"abs": abs, "round": round, "min": min, "max": max})
_CONSTS = {"pi": math.pi, "e": math.e}
MAX_POWER_EXPONENT = 1000


def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_POWER_EXPONENT:
            raise ValueError("số mũ quá lớn")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
        return _FUNCS[node.func.id](*[_eval_node(a) for a in node.args])
    raise ValueError(f"không hỗ trợ: {ast.dump(node)[:80]}")


def calculate(expr: str) -> str:
    """Tính biểu thức số học. Hỗ trợ + - * / // % **, sqrt, log, sin..., pi, e."""
    expr = str(expr).replace("^", "**")
    expr = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", expr)       # "1,000,000" -> "1000000", giữ dấu phẩy giữa tham số
    tree = ast.parse(expr, mode="eval")
    value = _eval_node(tree)
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        return f"{value:.10g}"
    return str(value)


# --------------------------------------------------------------------------- #
# 3. get_current_time
# --------------------------------------------------------------------------- #
DEFAULT_TIMEZONE = "Asia/Ho_Chi_Minh"


def get_current_time(timezone: str = DEFAULT_TIMEZONE) -> str:
    """Thời gian hiện tại theo múi giờ IANA (vd 'Asia/Tokyo', 'America/New_York', 'UTC')."""
    try:
        tz = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError):
        return f"ERROR: múi giờ không hợp lệ '{timezone}'. Dùng tên IANA, ví dụ 'Asia/Tokyo'."
    now = datetime.now(tz)
    return json.dumps({
        "timezone": timezone,
        "iso": now.isoformat(timespec="seconds"),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "weekday": now.strftime("%A"),
        "utc_offset": now.strftime("%z"),
        "day_of_year": now.timetuple().tm_yday,
    }, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
TOOLS: Dict[str, Tool] = {t.name: t for t in [
    Tool("search_web",
         "Tìm kiếm thông tin trên Internet. Dùng cho mọi sự kiện/số liệu bạn không chắc chắn hoặc có thể đã thay đổi.",
         {"query": "câu truy vấn (nên viết bằng tiếng Anh, ngắn gọn)",
          "max_results": "(tuỳ chọn) số kết quả, mặc định 5"},
         search_web),
    Tool("calculate",
         "Tính một biểu thức toán học. LUÔN dùng tool này thay vì tự tính nhẩm.",
         {"expr": "biểu thức, vd '(1.065**5) * 10000000' hoặc 'sqrt(2) * pi'"},
         calculate),
    Tool("get_current_time",
         "Lấy ngày giờ hiện tại (kèm thứ trong tuần, số thứ tự ngày trong năm).",
         {"timezone": f"(tuỳ chọn) múi giờ IANA, mặc định '{DEFAULT_TIMEZONE}'"},
         get_current_time),
]}
