# Phase 3: Assignments 3.3 – 3.9

| Thư mục | Bài | Nội dung chính | Chạy thử không tốn tiền |
|---|---|---|---|
| `assignment 32` | 3.2 | RAG from scratch (đã có sẵn) | `python -m tests.test_pipeline` |
| `assignment 33` | 3.3 | ReAct agent from scratch: parse text `Thought/Action/Observation`, stop sequence, 3 tools, 10 câu multi-step | `python -m tests.test_offline` |
| `assignment 34` | 3.4 | Agent trên LangGraph `StateGraph` + `interrupt()` trước `write_file`/`send_email`; `WHY_STATEGRAPH.md` | `python -m tests.test_graph_offline` |
| `assignment 35` | 3.5 | MCP server tự viết (JSON-RPC over stdio + Streamable HTTP): `read_file`, `list_directory`, `execute_python`; client, agent, cấu hình Claude Desktop; `PROTOCOL.md` | `python -m tests.test_server`, `python mcp_client_demo.py` |
| `assignment 36` | 3.6 | Researcher / Writer / Reviewer + Orchestrator, A2A-style message passing, mỗi agent một node LangGraph | `python -m tests.test_flow_offline` |
| `assignment 37` | 3.7 | Observability cho RAG 3.2 + agent 3.3: Langfuse + log JSONL + `dashboard.html` | `python -m tests.test_tracer_offline` |
| `assignment 38` | 3.8 | Golden set 20 câu + Faithfulness (LLM judge), Answer Relevance (cosine), Context Precision | `python -m tests.test_offline` |
| `assignment 39` | 3.9 | Customer Support multi-agent: Orchestrator (router API / Small LLM SFT+DPO) + Product/Order/Policy agent, A2A, tools qua MCP, Hybrid RAG + rerank, tracing/cost, eval Judge + Tool F1 | `python -m tests.test_offline` |

Mỗi thư mục có `README.md` riêng (cách chạy + gợi ý phân tích cho báo cáo) và `requirements.txt`.
Các bài sau dùng lại code bài trước qua `_bootstrap.py` (thêm thư mục kia vào `sys.path`), không copy code.

## Cài đặt một lần

```bat
cd C:\phase3
python -m venv .venv
.venv\Scripts\activate
pip install anthropic langgraph langfuse sentence-transformers ddgs tzdata numpy
```

Điền `ANTHROPIC_API_KEY` vào `assignment 33\.env` (copy từ `.env.example`). Các bài 3.5, 3.6 tự đọc file này; 3.7 và 3.8 đọc thêm `assignment 32\.env`.

## Model mặc định

- Agent (3.3–3.6) và judge (3.8): `claude-opus-5-5`. Đổi bằng `ANTHROPIC_MODEL` / `REACT_MODEL` / `EVAL_JUDGE_MODEL`, ví dụ `claude-haiku-5-5` cho rẻ.
- Generator của RAG (3.2, 3.7, 3.8): giữ như `assignment 32/config.py` (`claude-haiku-5-5`).

## Thay đổi ở bài 3.2

`assignment 32/rag/llm.py`: bỏ `temperature` khỏi lời gọi API. SDK `anthropic` 1.x / API version 2026-10-21 không còn nhận `temperature`/`top_p`/`top_k` (gửi lên sẽ lỗi 400). Claude Haiku 5.5 vốn chỉ chấp nhận giá trị mặc định, nên kết quả không đổi.
