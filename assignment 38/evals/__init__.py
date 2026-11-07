"""3 automated evals cho RAG.

faithfulness.py      : LLM-as-judge, câu trả lời có dựa trên context không (claim-level)
answer_relevance.py  : cosine similarity giữa embedding của answer và question
context_precision.py : % chunk được retrieve thực sự liên quan (ground-truth evidence + LLM judge)
"""
