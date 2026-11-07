"""RAG pipeline from scratch (no LangChain / LlamaIndex).

Bước 1  loader.py + chunking.py   : đọc PDF/text, chia chunk
Bước 2  embedder.py               : embed bằng sentence-transformers
Bước 3  vector_store.py           : lưu in-memory (NumPy array)
Bước 4  vector_store.search       : cosine similarity -> top-k
Bước 5  prompt.py                 : [system] + [chunks] + [user query]
Bước 6  llm.py                    : gọi Claude API
pipeline.py ghép tất cả lại, evaluation.py đo Recall@k.
"""
