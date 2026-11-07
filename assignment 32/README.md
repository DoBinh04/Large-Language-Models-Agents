# Báo cáo Assignment 3.2 — RAG Pipeline from Scratch

Không dùng LangChain / LlamaIndex. Pipeline chỉ dùng `numpy`, `sentence-transformers`, `pypdf` và SDK `anthropic`.

## 1. Mục tiêu và thiết lập

Mục tiêu là xây một RAG pipeline từ đầu: đọc tài liệu, chunk, semantic embedding, lưu vector trong bộ nhớ, cosine retrieval top-5, ghép prompt và gọi LLM. Tài liệu thử nghiệm là `data/sample_document.md` (23.040 ký tự, 4.638 BGE tokens), một handbook hư cấu để tránh việc LLM biết sẵn đáp án. Bộ đánh giá gồm 10 câu hỏi và 13 evidence trích nguyên văn. Evidence được đổi thành offset ký tự trong tài liệu nên đánh giá công bằng, không phụ thuộc vào chunk ID của từng strategy.

Embedding dùng `BAAI/bge-small-en-v1.5` (384 chiều); query có BGE search prefix. Vector store là ma trận NumPy `float32 (N,d)`; truy xuất dùng cosine similarity và lấy top-5. Metric chính là Recall@5: evidence được tính là tìm thấy khi ít nhất 50% ký tự của evidence nằm trong hợp top-5 chunks. Báo cáo thêm Strict Recall@5 (phủ 100%), Recall@1/@3, Hit@5 và MRR.

| Strategy | Cấu hình |
|---|---|
| fixed_256 | 256 token, overlap 32 |
| fixed_512 | 512 token, overlap 64 |
| recursive | tối đa 300 token; paragraph → line → sentence → clause → word |

## 2. Kết quả chạy thực tế

Lệnh đã chạy: `python run_experiment.py` với BGE embedding thật.

| Strategy | # chunks | TB token | Recall@1 | Recall@3 | Recall@5 | Strict R@5 | MRR | Evidence bị cắt | Chunk bị encoder cắt | Context top-5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_256 | 21 | 251 | 0.80 | 0.95 | 1.00 | 1.00 | 0.90 | 0/13 | 0 | 1.257 |
| fixed_512 | 11 | 480 | 0.55 | 0.95 | 1.00 | 1.00 | 0.80 | 0/13 | 10 | 2.399 |
| recursive | 18 | 258 | 0.80 | 1.00 | 1.00 | 1.00 | 0.90 | 0/13 | 0 | 1.288 |

Artifact có thể kiểm tra lại ở `results/20261031_143438/`: `summary.md`, `results.json`, `per_question.csv`, `retrieved.jsonl`, và JSON chunks của ba strategy.

## 3. Quan sát và kết luận

1. Recall@5 đều bằng 1.00 vì tài liệu ngắn và top-5 đã bao phủ phần lớn nội dung. Do đó không thể chỉ dùng Recall@5 để chọn strategy.
2. `recursive` là lựa chọn mặc định tốt nhất: Recall@3 = 1.00, MRR = 0.90, giữ ranh giới ngữ nghĩa và chỉ gửi khoảng 1.288 token context.
3. `fixed_256` cũng có MRR = 0.90 và context vừa phải, phù hợp khi cần cửa sổ token đều.
4. `fixed_512` có Recall@1 thấp nhất (0.55), MRR thấp nhất (0.80), và context gần gấp đôi. Chunk quá lớn trộn nhiều chủ đề, khiến vector biểu diễn bị loãng và chunk đúng tụt hạng.
5. Có 10/11 fixed_512 chunk bị encoder cắt: 512 là token nội dung, trong khi BGE còn thêm special tokens vào giới hạn input 512. Vì vậy không nên dùng fixed_512 nguyên trạng với encoder này.
6. Strict Recall@5 đều đạt 1.00. Overlap của fixed-size và vị trí evidence đã tránh được trường hợp top-5 chỉ lấy nửa evidence; nên cần thử thêm `--no-overlap` hoặc tài liệu dài hơn trước khi kết luận overlap không cần thiết.

## 4. Kiểm chứng generation với Claude

Bước 6 đã được kiểm chứng lại ngày 06-11-2026 bằng API key mới trong `.env`. Claude API phản hồi thành công. Với question `What temperature is the data centre kept at?` và context top-k chứa evidence liên quan, model trả lời: **“The data centre is kept at a constant 21 degrees Celsius all year round.”** Kết quả khớp ground truth và chỉ dùng thông tin trong context.

Luồng generation là: `RAGPipeline.answer()` gọi `retrieve()` để lấy top-5 → `build_prompt()` đóng gói system instructions, chunks và question → `ClaudeLLM.generate()` gọi Anthropic Messages API. Khi chạy trên máy cá nhân, dùng `python ask.py "..."` để hỏi một câu hoặc `python run_experiment.py --generate` để lưu toàn bộ đáp án vào `answers.jsonl`.

## 5. Hạn chế và hướng phát triển

Thí nghiệm chỉ dùng một tài liệu và 10 câu hỏi, nên chênh 0.1 chưa có ý nghĩa thống kê. Ground truth hiện đánh giá retrieval bằng evidence nguyên văn, chưa chấm chất lượng generation. Hướng phát triển: thêm tài liệu/câu hỏi, hybrid BM25+dense, cross-encoder reranker, metadata section, và query decomposition cho multi-hop questions như Q06.

---

## Hướng dẫn chạy và cấu trúc mã

## Cấu trúc

```
assignment 32/
├── config.py              # MỌI tham số: tài liệu, model, 3 strategy, top-k, LLM
├── ask.py                 # Hỏi đáp end-to-end (đủ 6 bước)
├── run_experiment.py      # So sánh 3 chiến lược chunking, đo Recall@5
├── rag/
│   ├── loader.py          # Bước 1a: đọc PDF / TXT / MD, làm sạch text
│   ├── tokenizer.py       # Đếm token + offset ký tự (HF tokenizer hoặc regex)
│   ├── chunking.py        # Bước 1b: fixed_256, fixed_512, recursive
│   ├── embedder.py        # Bước 2: sentence-transformers (+ embedder offline để test)
│   ├── vector_store.py    # Bước 3+4: NumPy array, cosine similarity, top-k
│   ├── prompt.py          # Bước 5: [system] + [chunks] + [user query]
│   ├── llm.py             # Bước 6: Claude API (tự DRY RUN khi chưa có key)
│   ├── pipeline.py        # Ghép 6 bước
│   └── evaluation.py      # Recall@k, strict recall, Hit@k, MRR
├── data/
│   ├── sample_document.md # Tài liệu mẫu (~6k tokens, hư cấu → LLM không "học thuộc" được)
│   └── questions.json     # 10 câu hỏi + đáp án + evidence (ground truth)
├── tests/test_pipeline.py # Test offline, không cần model
└── results/               # Sinh ra khi chạy run_experiment.py
```

## Cài đặt (Windows)

```bat
cd "C:\phase3\assignment 32"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Lần chạy đầu sẽ tự tải embedding model `BAAI/bge-small-en-v1.5` (~130 MB) từ HuggingFace.

## Chạy

```bat
:: 0) Kiểm tra code (không cần internet, không cần model)
python -m tests.test_pipeline

:: 1) Thí nghiệm chính: 3 strategy x 10 câu hỏi -> Recall@5  (KHÔNG cần API key)
python run_experiment.py

:: 1b) Ablation: bỏ overlap ở fixed-size để thấy rõ hiệu ứng cắt ngang câu
python run_experiment.py --no-overlap

:: 2) Hỏi đáp end-to-end. Chưa có key -> in ra prompt (dry run)
python ask.py "When can the dome reopen after a humidity closure?" --show-chunks
python ask.py -i --strategy fixed_256

:: 3) Khi đã có key: copy .env.example -> .env, điền ANTHROPIC_API_KEY, rồi
python ask.py "What temperature is the data centre kept at?"
python run_experiment.py --generate        :: lưu câu trả lời LLM vào answers.jsonl
```

Mỗi lần chạy `run_experiment.py` tạo thư mục `results/<thời gian>/`:

| File | Nội dung |
|---|---|
| `summary.md` | Bảng tổng hợp + Recall theo từng câu để đối chiếu với báo cáo ở đầu README |
| `results.json` | Số liệu đầy đủ |
| `per_question.csv` | Recall, rank, coverage từng câu × strategy (mở bằng Excel) |
| `retrieved.jsonl` | Nội dung top-5 chunk của từng câu → dùng để phân tích lỗi |
| `chunks_<strategy>.json` | Toàn bộ chunk của mỗi strategy |
| `answers.jsonl` | (khi `--generate`) câu trả lời của LLM vs ground truth |

## Thiết kế chính

**Chunking dựa trên offset.** Mọi chunk lưu `start/end` (vị trí ký tự trong tài liệu gốc). Token được đếm bằng *chính tokenizer của embedding model*, nên các target 256/512 được đo nhất quán. Lưu ý giới hạn input model còn gồm special tokens: fixed_512 có thể bị truncation với BGE max 512.

- `fixed_256` / `fixed_512`: cửa sổ trượt theo token, overlap 32 / 64 (12.5%), cắt bất kể câu.
- `recursive` (~300 tokens, đúng yêu cầu Bước 1): thử tách theo `\n\n` → `\n` → `. ` → `? ` `! ` `; ` `, ` → ` ` → cắt cứng; chỉ xuống separator mịn hơn khi mảnh vẫn > 300 token, rồi gộp tham lam các mảnh liền kề cho tới gần 300.

**Vector store**: ma trận `float32 (N, d)` + list chunk. Search = `E·q / (‖E‖‖q‖)`, chọn top-k bằng `np.argpartition` (O(N)) rồi sort k phần tử.

**Ground truth không phụ thuộc chunking.** Vì mỗi strategy tạo ra bộ chunk khác nhau, ground truth không thể là "chunk #7". Mỗi câu hỏi có 1–2 đoạn `evidence` trích nguyên văn từ tài liệu. Evaluator tìm vị trí evidence, rồi tính:

- `coverage(e)` = % ký tự của evidence nằm trong top-k chunk
- **Recall@5** = trung bình (số evidence có coverage ≥ 50%) / (tổng số evidence của câu)
- **Strict Recall@5**: như trên nhưng yêu cầu coverage = 100% → phát hiện evidence bị cắt ngang biên chunk
- **MRR**, **Hit@5**, **Recall@1/@3** (tài liệu ngắn → Recall@5 dễ bão hoà, k nhỏ phân biệt tốt hơn)
- Các chỉ số phụ: số evidence bị cắt ở biên chunk, số chunk dài hơn `max_seq_length` của embedder (bị cắt bớt khi embed)

10 câu hỏi được thiết kế có chủ đích: fact đơn + số gây nhiễu (Q01, Q05), multi-evidence (Q02, Q08), paraphrase ít trùng từ (Q03, Q09), evidence dài dạng list (Q04), multi-hop 2 section xa nhau (Q06), câu hỏi "vì sao" (Q10).

## Dùng tài liệu của riêng bạn

1. Đặt file vào `data/`, sửa `DOCUMENT_PATH` trong `config.py` (hoặc `--document path`).
2. Viết lại `data/questions.json`: mỗi `evidence` phải **trích nguyên văn** từ tài liệu (khác hoa/thường hay xuống dòng thì không sao). Script báo lỗi nếu không tìm thấy.
3. Tài liệu tiếng Việt → đặt `EMBEDDING_PRESET = "e5-multilingual"`.
4. PDF: `pypdf` đôi khi làm mất ranh giới đoạn (`\n\n`). Khi đó recursive chunking sẽ kém lợi thế hơn, và đây cũng là một quan sát đáng ghi vào báo cáo.

## Thí nghiệm thêm (tuỳ chọn, cho điểm cộng)

- `--preset minilm`: `all-MiniLM-L6-v2` có `max_seq_length = 256` nên chunk 512 bị **cắt mất nửa sau** khi embed. Cột "Chunk bị embedder cắt" sẽ cho thấy điều này.
- `--no-overlap`: so sánh strict recall có/không overlap.
- Sửa `chunk_size` của recursive trong `config.py` (200 / 300 / 500).
