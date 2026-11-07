"""ReAct agent from scratch (không dùng LangChain / LangGraph / tool runner).

llm.py    : gọi Claude Messages API (text thuần, dùng stop sequence)
tools.py  : search_web, calculate, get_current_time + registry
parser.py : tách Thought / Action / Action Input / Final Answer từ text của LLM
prompt.py : system prompt mô tả format ReAct + danh sách tool
agent.py  : vòng lặp Thought -> Action -> Observation -> ... -> Final Answer
"""
