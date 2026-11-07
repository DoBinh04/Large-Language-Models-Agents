"""Metrics không cần LLM: Routing accuracy và Tool-call F1.

Routing đúng (mỗi lượt) khi: tập agents dự đoán == tập agents kỳ vọng VÀ (có hỏi lại thiếu thông tin) khớp.
Thứ tự agents không tính (thứ tự chỉ ảnh hưởng chất lượng câu trả lời, đã có Judge đo).

Tool-call F1 (mỗi kịch bản, gộp mọi lượt):
- Một call dự đoán khớp call kỳ vọng khi cùng tên tool VÀ mọi tham số được nêu trong expected đều bằng nhau
  (so sánh không phân biệt hoa thường; list so như tập). Tham số không nêu -> không xét
  (vd. query của search_* là câu tự do nên expected để {}).
- Ghép tham lam 1-1: TP = số cặp khớp; FP = call thừa (gọi sai / gọi lặp / gọi không cần);
  FN = call kỳ vọng bị thiếu. Precision phạt agent "gọi lung tung" (tốn latency + tiền), recall phạt agent bỏ bước.
- Kịch bản không cần tool và agent không gọi tool -> P = R = F1 = 1.
Báo cáo micro-F1 (cộng TP/FP/FN toàn bộ) và macro-F1 (trung bình theo kịch bản).
"""
from __future__ import annotations


def routing_correct(pred: dict, gold_agents: list[str], gold_missing: bool) -> bool:
    return set(pred.get("agents", [])) == set(gold_agents) and bool(pred.get("missing_info")) == gold_missing


def _norm(v):
    if isinstance(v, (list, tuple)):
        return tuple(sorted(_norm(x) for x in v))
    if isinstance(v, str):
        return v.strip().upper()
    return v


def call_matches(pred: dict, exp: dict) -> bool:
    if pred.get("name") != exp["name"]:
        return False
    args = pred.get("args") or {}
    return all(_norm(args.get(k)) == _norm(v) for k, v in exp.get("args", {}).items())


def tool_f1(pred_calls: list[dict], exp_calls: list[dict]) -> dict:
    used = [False] * len(pred_calls)
    tp = 0
    for e in exp_calls:
        for i, p in enumerate(pred_calls):
            if not used[i] and call_matches(p, e):
                used[i] = True
                tp += 1
                break
    fp, fn = len(pred_calls) - tp, len(exp_calls) - tp
    if not pred_calls and not exp_calls:
        return {"tp": 0, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0}
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


def micro_f1(rows: list[dict]) -> dict:
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r) if p + r else 0.0, 4)}


def percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)
