"""Faithfulness: câu trả lời có DỰA TRÊN context đã retrieve không? (LLM-as-judge)

Cách làm (giống RAGAS, nhưng gộp 2 bước vào 1 lời gọi có structured output):
  1. Tách câu trả lời thành các claim nguyên tử (mỗi claim là một mệnh đề kiểm chứng được).
  2. Với từng claim: supported / unsupported / contradicted, so với CONTEXT
     (không so với kiến thức chung và không so với đáp án chuẩn).
  score = #supported / #claims

Faithfulness KHÁC correctness: trả lời sai nhưng đúng như context nói thì vẫn faithful.
Câu trả lời từ chối ("I don't know...") không có claim nào -> score = 1.0 (không bịa), gắn cờ abstained.
"""
from __future__ import annotations

from .judge import Judge, format_chunks

SCHEMA = {
    "type": "object",
    "properties": {
        "abstained": {"type": "boolean", "description": "true if the answer declines to answer / says the info is missing"},
        "claims": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "claim": {"type": "string"},
                "verdict": {"type": "string", "enum": ["supported", "unsupported", "contradicted"]},
                "chunk_ids": {"type": "array", "items": {"type": "integer"}},
                "reason": {"type": "string"},
            },
            "required": ["claim", "verdict", "chunk_ids", "reason"],
            "additionalProperties": False,
        }},
    },
    "required": ["abstained", "claims"],
    "additionalProperties": False,
}

SYSTEM = """You evaluate the FAITHFULNESS of an answer produced by a retrieval-augmented system.

Step 1. Break the answer into atomic factual claims: short, self-contained statements that can each be checked.
- Ignore citation markers like [1], filler, and restatements of the question.
- A statement that information is missing ("the document does not say X") is not a factual claim about the world;
  if the whole answer declines to answer, set abstained=true and return an empty claims list.
Step 2. Judge every claim ONLY against the context chunks, never against your own knowledge:
- supported: the context states it or it follows directly (simple arithmetic or paraphrase is fine)
- unsupported: the context does not contain this information
- contradicted: the context says something different
List the chunk ids you relied on, and give a one-sentence reason."""


def faithfulness(judge: Judge, question: str, answer: str, contexts: list[str]) -> dict:
    user = (f"<context>\n{format_chunks(contexts)}\n</context>\n\n"
            f"<question>{question}</question>\n\n<answer>\n{answer}\n</answer>")
    out = judge.ask(SYSTEM, user, SCHEMA)
    claims = out["claims"]
    n_sup = sum(c["verdict"] == "supported" for c in claims)
    if not claims:
        score = 1.0 if out["abstained"] else None
    else:
        score = n_sup / len(claims)
    return {
        "score": score,
        "abstained": out["abstained"],
        "n_claims": len(claims),
        "n_supported": n_sup,
        "n_contradicted": sum(c["verdict"] == "contradicted" for c in claims),
        "claims": claims,
    }
