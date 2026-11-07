"""Chấm tự động câu trả lời của agent cho bộ 10 câu test.

Câu trả lời có thể viết số theo kiểu Việt ("13.700.866,6") hoặc kiểu Anh ("13,700,866.6"),
nên extract_numbers sinh cả hai cách hiểu cho mỗi con số và chấp nhận nếu một cách khớp.
Các check phụ thuộc thời gian (days_until, weekday_after, year_progress) tính đáp án
lúc chạy, theo múi giờ mặc định của tool get_current_time.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from .tools import DEFAULT_TIMEZONE

_NUM = re.compile(r"-?\d[\d.,]*\d|-?\d")

WEEKDAYS = {
    0: ["monday", "thứ hai", "thứ 2"],
    1: ["tuesday", "thứ ba", "thứ 3"],
    2: ["wednesday", "thứ tư", "thứ 4"],
    3: ["thursday", "thứ năm", "thứ 5"],
    4: ["friday", "thứ sáu", "thứ 6"],
    5: ["saturday", "thứ bảy", "thứ 7"],
    6: ["sunday", "chủ nhật"],
}


def _today() -> date:
    return datetime.now(ZoneInfo(DEFAULT_TIMEZONE)).date()


def extract_numbers(text: str) -> list[float]:
    out: list[float] = []
    for tok in _NUM.findall(text or ""):
        candidates = {
            tok.replace(",", ""),                          # kiểu Anh: 13,700,866.6
            tok.replace(".", "").replace(",", "."),        # kiểu Việt: 13.700.866,6
        }
        for c in candidates:
            try:
                out.append(float(c))
            except ValueError:
                pass
    return out


def _any_close(nums: list[float], target: float, tol: float) -> bool:
    return any(abs(n - target) <= tol for n in nums)


def check_answer(answer: str | None, check: dict) -> tuple[bool | None, str]:
    """Trả về (passed, giải thích). passed = None nghĩa là không chấm tự động được."""
    if not answer:
        return False, "không có câu trả lời"
    kind = check.get("type", "manual")
    nums = extract_numbers(answer)
    low = answer.lower()

    if kind == "contains":
        ok = check["value"].lower() in low
        return ok, f"cần chứa '{check['value']}'"
    if kind == "number_close":
        return _any_close(nums, check["value"], check["tolerance"]), f"cần ≈ {check['value']} ± {check['tolerance']}"
    if kind == "number_range":
        ok = any(check["min"] <= n <= check["max"] for n in nums)
        return ok, f"cần một số trong [{check['min']}, {check['max']}]"
    if kind == "number_in":
        return any(n in check["values"] for n in nums), f"cần một trong {check['values']}"
    if kind == "days_until":
        expected = (date.fromisoformat(check["date"]) - _today()).days
        # chấp nhận ±1 vì có người đếm cả ngày hôm nay
        return _any_close(nums, expected, 1), f"cần ≈ {expected} ngày (±1)"
    if kind == "weekday_after":
        wd = (_today() + timedelta(days=check["days"])).weekday()
        names = WEEKDAYS[wd]
        return any(n in low for n in names), f"cần '{names[0]}' / '{names[1]}'"
    if kind == "year_progress":
        t = _today()
        doy = t.timetuple().tm_yday
        days_in_year = (date(t.year + 1, 1, 1) - date(t.year, 1, 1)).days
        pct = round(doy / days_in_year * 100, 1)
        ok = _any_close(nums, doy, 1) and _any_close(nums, pct, 0.3)
        return ok, f"cần ngày thứ {doy} và ≈ {pct}%"
    return None, "chấm tay"
