# Báo cáo Assignment 3.7: Observability cho RAG pipeline và ReAct Agent

| | |
|---|---|
| **Môn học** | Phase 3: LLM Engineering |
| **Ngày chạy thực nghiệm** | 07/11/2026 |
| **Công cụ** | Langfuse Cloud (vùng JP), Python SDK `langfuse 5.2.0`, Anthropic SDK |
| **Hệ thống được quan sát** | RAG pipeline (Assignment 3.2) và ReAct agent (Assignment 3.3) |

---

## 1. Đề bài

> Thêm full observability vào RAG pipeline từ Section 2 và Agent từ Section 4:
> - Log mọi LLM call với: prompt hash, model, tokens in/out, latency, cost estimate
> - Log retrieval: query, top-k chunks returned, similarity scores
> - Dashboard đơn giản: trung bình latency, total cost, top failed queries
> - Dùng Langfuse (open source, self-host được).

### Đối chiếu yêu cầu và nơi thực hiện

| Yêu cầu | Trạng thái | Thực hiện ở đâu |
|---|---|---|
| Prompt hash cho mỗi LLM call | ✅ | `tracer.prompt_hash()`: sha256 của `{model, system, messages}`, lấy 16 ký tự đầu |
| Model, tokens in/out | ✅ | `Observation.set_usage()` đọc `response.usage` (gồm cả cache read/write tokens) |
| Latency | ✅ | Đo bằng `time.perf_counter()` trong `Observation.__exit__` |
| Cost estimate | ✅ | `pricing.estimate_cost()` theo bảng giá Anthropic. Kết quả trùng với giá Langfuse tự tính (xem §5.4). |
| Log retrieval: query, top-k chunks, similarity scores | ✅ | `TracedRAGPipeline.retrieve()`: rank, chunk_id, **score**, vị trí ký tự, preview, max/min/mean score |
| Dashboard: latency TB, total cost, top failed queries | ✅ | `dashboard.py` in ra terminal và ghi `dashboard.html`. Langfuse cũng có dashboard riêng. |
| Dùng Langfuse | ✅ | `observability/langfuse_sink.py`. Chạy trên Langfuse Cloud, code dùng được cả với bản self-host. |

---

## 2. Kiến trúc

### 2.1 Nguyên tắc thiết kế

1. **Không sửa code của 3.2 và 3.3.** Mọi instrument đều làm bằng *subclass*:
   - `TracedClaudeLLM` ghi đè `ClaudeLLM.generate`
   - `TracedRAGPipeline` ghi đè `RAGPipeline.retrieve` và `answer`
   - `TracedReActAgent` ghi đè hai hook `_call_llm` / `_run_tool` đã chuẩn bị sẵn ở 3.3, cùng với `run`
2. **Observability không được làm hỏng ứng dụng.** Mọi lời gọi Langfuse đều bọc `try/except`. Langfuse lỗi mạng hoặc sai key thì app vẫn chạy và vẫn ghi log cục bộ.
3. **Ghi ra hai nơi:**
   - `logs/traces-YYYYMMDD.jsonl`: luôn có, không phụ thuộc dịch vụ ngoài, là nguồn dữ liệu của `dashboard.py`
   - **Langfuse**: dùng để xem cây trace, lọc, so sánh qua giao diện web
4. **Quan hệ cha/con lấy từ `contextvars`.** Code được instrument không phải truyền `parent` qua tham số. Langfuse SDK (chạy trên OpenTelemetry) cũng dùng context hiện hành, và tracer mở/đóng observation Langfuse đúng thứ tự lồng nhau, nên cây trên Langfuse giống hệt cây trong JSONL.

### 2.2 Cây trace

```
rag.query  (trace, tags: rag, recursive)        react.run  (trace, tags: agent)
├── rag.retrieve   (retriever)                  ├── react.step1      (generation)
│     query, top-5 chunks + scores              ├── tool.search_web  (tool)
└── rag.generate   (generation)                 ├── react.step2      (generation)
      tokens, latency, cost, prompt_hash        ├── tool.calculate   (tool)
                                                └── react.step3      (generation → Final Answer)
```

Mỗi observation dùng đúng *loại* của Langfuse (`as_type` = `generation` / `retriever` / `tool` / `span`), nên giao diện Langfuse hiển thị icon và trường riêng cho từng loại.

### 2.3 Cấu trúc thư mục

```
assignment 37/
├── observability/
│   ├── tracer.py            # Tracer / Observation, ghi JSONL + điều phối Langfuse
│   ├── langfuse_sink.py     # adapter Langfuse SDK (v3 → v5), tự tắt khi lỗi
│   ├── pricing.py           # bảng giá → ước tính cost
│   ├── instrument_rag.py    # TracedClaudeLLM, TracedRAGPipeline
│   └── instrument_agent.py  # TracedReActAgent
├── run_rag.py               # chạy RAG 3.2 có trace (10 câu + 3 câu ngoài phạm vi)
├── run_agent.py             # chạy agent 3.3 có trace (10 câu test)
├── dashboard.py             # đọc JSONL → tóm tắt + dashboard.html
├── tests/test_tracer_offline.py
└── logs/                    # JSONL sinh ra khi chạy
```

---

## 3. Dữ liệu được log

| Observation | Trường |
|---|---|
| **generation** (mọi LLM call) | `model`, `prompt_hash`, `system_hash` (phiên bản system prompt), `input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_write_tokens`, `latency_ms`, `cost_usd`, `cost_details`, `stop_reason`, `request_id`, `model_parameters` |
| **retrieval** (RAG) | `query`, `top_k`, `chunks[]` (rank, chunk_id, score, start/end, preview 160 ký tự), `max_score` / `min_score` / `mean_score`, `embed_ms`, `search_ms` |
| **tool** (agent) | tên tool, input, output, `tool_error`, latency |
| **trace** (một câu hỏi) | `query`, output, latency tổng, `failed`, `failure_reason`, tags `rag`/`agent`, `session_id`, link Langfuse |

**Vì sao cần cả `prompt_hash` lẫn `system_hash`?**
- `prompt_hash` thay đổi theo từng câu hỏi. Hai request có cùng hash là request giống hệt nhau, nên là ứng viên cho response cache.
- `system_hash` chỉ đổi khi system prompt đổi. Có trường này thì so sánh được chất lượng và cost giữa các *phiên bản prompt*.

### 3.1 Định nghĩa "failed query"

| Hệ thống | Điều kiện |
|---|---|
| RAG | có exception; câu trả lời chứa *"I don't know based on the provided document"* (`no_answer`); top-1 similarity < 0.55 (`low_retrieval_score`) |
| Agent | exception; `max_steps` (hết số bước mà chưa có Final Answer); `refusal` |

Trên Langfuse, trace failed có `level = WARNING`, hoặc `ERROR` nếu là exception, kèm score `failed = 1`. Trace thành công có `failed = 0`. Lọc theo hai trường này sẽ ra danh sách query lỗi.

### 3.2 Ước tính cost

```
cost = input_tokens  × giá_input
     + output_tokens × giá_output
     + cache_read    × giá_đọc_cache     (0.1 × input; riêng Opus 5.5 / Sonnet 5.5 = $0.20, Fable 5.1 = $0.25 / MTok)
     + cache_write   × 1.25 × giá_input
```

| Model dùng trong thực nghiệm | Input $/MTok | Output $/MTok |
|---|---|---|
| `claude-haiku-4-5` (RAG) | 1.00 | 5.00 |
| `claude-opus-5-5` (Agent) | 4.00 | 20.00 |

Model id có hậu tố snapshot (vd `claude-haiku-4-5-20251001`) được khớp theo tiền tố dài nhất. Model không có trong bảng giá thì `cost_usd = null`, và dashboard đánh dấu "(giá ?)" thay vì ghi 0 sai.

---

## 4. Cài đặt Langfuse

Báo cáo này dùng **Langfuse Cloud** (gói Hobby miễn phí, vùng JP) vì máy cá nhân ít RAM. Bản self-host cần Postgres, ClickHouse, Redis và MinIO, tốn khoảng 4 GB RAM. Code không phụ thuộc vào việc chọn Cloud hay self-host: chỉ cần đổi `LANGFUSE_HOST` / `LANGFUSE_BASE_URL`.

```ini
# .env
ANTHROPIC_API_KEY=sk-ant-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://jp.cloud.langfuse.com     # self-host: http://localhost:3000
TAVILY_API_KEY=tvly-...                              # không bắt buộc (agent tự chuyển sang DuckDuckGo / Wikipedia)
```

Muốn self-host thì làm như sau:

```bat
git clone https://github.com/langfuse/langfuse.git
cd langfuse
docker compose up -d
:: mở http://localhost:3000 → tạo project → Settings → API Keys
```

### 4.1 Tương thích Langfuse SDK v5

Lúc chạy, phiên bản mới nhất là `langfuse 5.2.0`. Bản này **bỏ `update_trace()`** của SDK v3. Thuộc tính cấp trace (tên, tags, session) giờ phải gắn qua context manager `propagate_attributes()`, và phải mở *trước* observation gốc để mọi span con đều mang theo các thuộc tính đó. Adapter `langfuse_sink.py` xử lý cả hai trường hợp:

- SDK có `propagate_attributes` (v4+): mở nó trước observation gốc, đóng sau observation gốc.
- SDK v3: lùi về `update_trace()`.

Key của `cost_details` cũng được đặt trùng key của `usage_details` (`input`, `output`, `cache_read_input_tokens`, `cache_creation_input_tokens`) để Langfuse hiển thị đúng phần cost chia theo từng loại token.

---

## 5. Kết quả thực nghiệm

**Cấu hình:**
- RAG: tài liệu mẫu của 3.2, chunking `recursive` (18 chunks), embedding `BAAI/bge-small-en-v1.5`, top-k = 5, LLM `claude-haiku-4-5`
- Agent: `claude-opus-5-5`, effort `medium`, tối đa 10 bước, các tool `search_web` / `calculate` / `get_current_time`

**Bộ câu hỏi:**
- RAG: 10 câu của 3.2 (Q01–Q10) và 3 câu cố ý nằm ngoài phạm vi tài liệu (OOS1–OOS3) để tạo failed query
- Agent: 10 câu test của 3.3 (T01–T10)

### 5.1 Tổng quan (từ `python dashboard.py`)

```
Traces: 23 (failed 3) | LLM calls: 52 | tokens in/out: 96,245/8,358 | total cost: $0.4692 (avg $0.02040/trace)
  rag.query    avg     1772 ms  p50     1878  p95     2265  (n=13)
  react.run    avg    16311 ms  p50    13709  p95    28925  (n=10)
  LLM call     avg     3109 ms  | retrieval avg 48.9 ms
Top failed queries:
  1× [rag.query] How many exoplanets has HERON discovered?  (no_answer×1)
  1× [rag.query] What is the ticket price for the night-time public observing session?  (no_answer×1)
  1× [rag.query] Who is the current president of France?  (low_retrieval_score(0.43), no_answer×1)
```

| Chỉ số | RAG (13 trace) | Agent (10 trace) |
|---|---|---|
| Latency trung bình / trace | **1.77 s** | **16.31 s** (≈ 9.2×) |
| LLM call / trace | 1 | 3–5 (TB 3.9) |
| Cost trung bình / trace | **$0.0021** | **$0.0442** (≈ 21×) |
| Tổng cost | $0.0276 | $0.4416 |
| Tỉ lệ failed | 3/13 (đều là câu OOS, đúng như thiết kế) | 0/10 |

### 5.2 Chi tiết RAG: retrieval và generation

| ID | Latency (ms) | Retrieve (ms) | Generate (ms) | Tokens in / out | Cost ($) | Score top-1 / top-5 | Failed |
|---|---|---|---|---|---|---|---|
| Q01 | 2265 | 36 | 2224 | 1738 / 106 | 0.00227 | 0.826 / 0.638 | – |
| Q02 | 2004 | 50 | 1950 | 1506 / 115 | 0.00208 | 0.795 / 0.647 | – |
| Q03 | 1964 | 46 | 1913 | 1706 / 92 | 0.00217 | 0.628 / 0.578 | – |
| Q04 | 3167 | 52 | 3111 | 1657 / 194 | 0.00263 | 0.786 / 0.590 | – |
| Q05 | 966 | 51 | 910 | 1681 / 30 | 0.00183 | 0.690 / 0.594 | – |
| Q06 | 1323 | 52 | 1260 | 1784 / 76 | 0.00216 | 0.615 / 0.575 | – |
| Q07 | 1707 | 54 | 1649 | 1660 / 69 | 0.00200 | 0.741 / 0.588 | – |
| Q08 | 1231 | 53 | 1173 | 1714 / 48 | 0.00195 | 0.686 / 0.629 | – |
| Q09 | 2045 | 55 | 1986 | 1766 / 113 | 0.00233 | 0.673 / 0.625 | – |
| Q10 | 2045 | 57 | 1984 | 1648 / 116 | 0.00223 | 0.739 / 0.596 | – |
| OOS1 | 1193 | 40 | 1149 | 1685 / 46 | 0.00191 | **0.428** / 0.389 | low_score, no_answer |
| OOS2 | 1878 | 51 | 1816 | 1604 / 75 | 0.00198 | 0.658 / 0.585 | no_answer |
| OOS3 | 1244 | 40 | 1200 | 1799 / 60 | 0.00210 | 0.601 / 0.552 | no_answer |

### 5.3 Chi tiết Agent: số bước, token, cost

| ID | Latency (ms) | LLM calls | Tool calls | Tokens in / out | Cost ($) | Input tokens theo từng bước |
|---|---|---|---|---|---|---|
| T01 | 16145 | 4 | 3 | 11168 / 936 | 0.0634 | 837 → 2385 → 3879 → 4067 |
| T02 | 12084 | 3 | 2 | 3007 / 481 | 0.0216 | 822 → 989 → 1196 |
| T03 | 8922 | 4 | 3 | 4383 / 618 | 0.0299 | 843 → 1005 → 1191 → 1344 |
| T04 | 11774 | 3 | 2 | 2966 / 583 | 0.0235 | 875 → 969 → 1122 |
| T05 | 13709 | 4 | 3 | 8210 / 634 | 0.0455 | 837 → 2337 → 2490 → 2546 |
| T06 | 28925 | 5 | 4 | 13279 / 875 | 0.0706 | 867 → 2132 → 3345 → 3421 → 3514 |
| T07 | 16146 | 4 | 3 | 4340 / 703 | 0.0314 | 817 → 986 → 1121 → 1416 |
| T08 | 23277 | 5 | 4 | 13809 / 855 | 0.0723 | 849 → 2091 → 3486 → 3605 → 3778 |
| T09 | 24222 | 4 | 3 | 10023 / 986 | 0.0598 | 835 → 2113 → 3478 → 3597 |
| T10 | 7907 | 3 | 2 | 3112 / 547 | 0.0234 | 857 → 1035 → 1220 |

Cả 10 câu đều `success`. `cache_read_tokens = 0` ở mọi bước (phân tích ở §6.3).

### 5.4 Kiểm tra dữ liệu trên Langfuse

Sau khi chạy, giao diện Langfuse hiển thị các trace và observation của agent. Ảnh dưới đây là bằng chứng trực quan cho việc các generation và tool call đã được gửi lên Langfuse.

![Giao diện Langfuse hiển thị trace, input/output, metadata và latency của ReAct agent](./Screenshot%20%28150%29.png)

*Hình 1. Trang Tracing của Langfuse sau khi chạy ReAct agent. Các dòng generation, tool và trace gốc được hiển thị trong cùng một phiên làm việc.*

Đối chiếu với JSONL của lần chạy, dữ liệu quan sát có các đặc điểm sau:

| Kiểm tra | Kết quả |
|---|---|
| Số trace gốc | 23 (10 `rag.query` DEFAULT, 3 `rag.query` **WARNING**, 10 `react.run` DEFAULT) |
| Loại observation | `GENERATION`, `RETRIEVER`, `TOOL`, `SPAN`, đúng loại cho từng bước |
| Trạng thái failed trong log cục bộ | 3 trace RAG được đánh dấu `failed = true`; 20 trace còn lại là `false` |
| Session | `rag-<run_id>` và `agent-<run_id>` gom các trace của cùng một lần chạy |
| Generation `react.step3` (ví dụ) | `model = claude-opus-5-5`, usage 1220 in / 303 out, cost $0.01094 |
| Giá ước tính | `pricing.py` tính cost theo số token và bảng giá cấu hình trong source code |

Mỗi trace gốc trong JSONL có trường `langfuse_url` trỏ thẳng tới trace trên Langfuse.

---

## 6. Phân tích

### 6.1 Agent tốn hơn RAG khoảng 20 lần mỗi câu hỏi

- **Cost:** agent $0.044/câu, RAG $0.0021/câu, chênh khoảng 21 lần. Có hai nguyên nhân cộng lại: (1) agent gọi LLM 3–5 lần thay vì 1 lần; (2) agent dùng Opus 5.5 ($4/$20), còn RAG dùng Haiku 4.5 ($1/$5). Nếu chỉ do số lần gọi thì mức chênh khoảng 4 lần. Phần còn lại là do chọn model.
- **Latency:** 16.3 s so với 1.8 s. Trong RAG, **97% thời gian là generation** (retrieval chỉ khoảng 50 ms vì vector store nằm trong RAM với 18 chunks). Với agent, latency tăng theo số bước và theo các tool gọi mạng (`search_web`). Câu chậm nhất (T06, 28.9 s) gọi `search_web` 2 lần.
- Trace có tool `search_web` (T01, T05, T06, T08, T09) đắt hơn hẳn các trace chỉ dùng `calculate` / `get_current_time` (khoảng $0.06 so với $0.02–0.03). Lý do là kết quả search được đưa vào hội thoại làm tăng mạnh input tokens.

### 6.2 Input tokens của agent tăng dần theo từng bước

Cột "Input tokens theo từng bước" cho thấy input tăng đơn điệu (vd T08: 849 → 2091 → 3486 → 3605 → 3778). Lý do là ReAct gửi lại *toàn bộ* hội thoại ở mỗi bước. Bước có bước nhảy lớn nhất luôn là bước ngay sau một lần `search_web` (khoảng +1200 đến +1550 token mỗi lần), vì observation của search dài hơn nhiều so với kết quả của `calculate`. Đây chính là loại insight mà observability giúp nhìn ra: muốn giảm cost agent thì **rút gọn output của tool search** sẽ hiệu quả hơn đổi prompt.

### 6.3 Vì sao `cache_read_tokens = 0`?

Agent 3.3 chưa đặt `cache_control` cho prompt. Mặt khác, prefix chung giữa các bước (system prompt và câu hỏi, khoảng 800 token) còn nhỏ hơn mức tối thiểu cache được của Opus. Vì log có trường `cache_read_tokens`, việc này hiện ra ngay. Muốn tận dụng prompt caching thì cần bật `cache_control` và chấp nhận rằng chỉ những câu có nhiều bước, hội thoại dài mới được lợi.

### 6.4 Ngưỡng similarity 0.55 chưa đủ để phát hiện câu ngoài phạm vi

| | Score top-1 |
|---|---|
| OOS1 (tổng thống Pháp, hoàn toàn ngoài chủ đề) | 0.428 |
| OOS3 (số exoplanet HERON đã tìm ra) | 0.601 |
| OOS2 (giá vé buổi quan sát đêm) | 0.658 |
| Q06 (câu hỏi hợp lệ, trả lời đúng) | 0.615 |
| Q03 (câu hỏi hợp lệ, trả lời đúng) | 0.628 |

- Chỉ OOS1 bị ngưỡng 0.55 bắt được. OOS2 và OOS3 *cùng chủ đề* với tài liệu (vẫn nói về HERON, Visitor Centre) nên similarity cao. Riêng OOS2 (0.658) còn cao hơn cả hai câu hợp lệ Q03 và Q06.
- Vì vậy **không có ngưỡng similarity nào tách được hai nhóm này**. Đặt ngưỡng 0.66 thì bắt được OOS2/OOS3 nhưng cũng đánh dấu sai hai câu hợp lệ Q03 và Q06 (false positive).
- Tín hiệu đáng tin hơn là **câu trả lời của chính LLM** (`no_answer`): cả 3 câu OOS đều bị phát hiện nhờ điều kiện này, và không có câu hợp lệ nào bị đánh dấu nhầm. Kết luận: nên kết hợp nhiều tín hiệu thay vì chỉ dựa vào retrieval score. Assignment 3.8 (Faithfulness, Context Precision) đi tiếp theo hướng này.

### 6.5 Prompt hash

13 câu RAG có 13 `prompt_hash` khác nhau nhưng chung một `system_hash`, nghĩa là cùng một phiên bản prompt. Chạy lại cùng câu hỏi trên cùng index sẽ ra cùng `prompt_hash`. Nhờ vậy có thể (1) phát hiện request trùng lặp để cache response và (2) gom các lần chạy theo phiên bản prompt khi thử prompt mới.

---

## 7. Dashboard

`python dashboard.py` đọc toàn bộ `logs/*.jsonl` và ghi ra `dashboard.html`, một file HTML tĩnh hỗ trợ cả light và dark mode, gồm:

- **KPI:** số trace, số failed, số LLM call, **tổng cost**, cost TB/trace, **latency TB của LLM call**, tổng tokens, similarity top-1 TB.
- **Latency theo loại trace** (avg / p50 / p95, kèm thanh so sánh): thấy ngay agent chậm hơn RAG bao nhiêu lần.
- **Bảng theo model:** số call, tokens, latency, cost.
- **Bảng theo tool:** số lần gọi, số lỗi, latency.
- **Top failed queries:** gom theo query, đếm số lần, lý do, thời điểm gần nhất.

Có thể lọc theo một lần chạy (`--run 20261107-163419`) hoặc từ một ngày (`--since 2026-11-01`).

Langfuse cũng có dashboard riêng: trang *Home* hiển thị latency, cost theo model và số trace theo thời gian; trang *Traces* lọc được theo tag `rag`/`agent`, `level = WARNING`, score `failed`, hoặc session. `dashboard.py` cho thấy cách tự tính những con số đó từ log thô.

---

## 8. Hạn chế và hướng phát triển

1. **Chưa đo chất lượng câu trả lời.** "Failed" hiện chỉ dựa trên heuristic (`no_answer`, score thấp, exception). Một câu trả lời *sai nhưng tự tin* sẽ không bị phát hiện. Assignment 3.8 bổ sung LLM-as-judge (faithfulness), answer relevance và context precision.
2. **Thời gian cho tool `search_web` phụ thuộc mạng**, nên latency của agent dao động mạnh (p50 13.7 s, p95 28.9 s). Cần chạy nhiều lần hơn để có số liệu ổn định.
3. **Bảng giá viết cứng trong code.** Khi Anthropic đổi giá phải sửa `pricing.py`. Langfuse có bảng giá model riêng, có thể dùng làm nguồn đối chiếu như đã làm ở §5.4.
4. **Log cục bộ cắt input/output ở 4000 ký tự** (`OBS_MAX_FIELD_CHARS`) để file JSONL không phình to. Bản đầy đủ nằm trên Langfuse.
5. **Chưa bật prompt caching cho agent** (§6.3). Đây là cách giảm cost dễ nhất cho các câu nhiều bước.

---

## 9. Cách chạy lại

```bat
cd "C:\phase3\assignment 37"
pip install -r requirements.txt
copy .env.example .env          :: rồi điền key (mục 4)

python -m tests.test_tracer_offline              :: test tracer + dashboard, không cần key
python run_rag.py                                :: 10 câu 3.2 + 3 câu OOS
python run_rag.py --embedder hashing --dry-run   :: thử toàn bộ luồng, không tải model, không tốn tiền
python run_agent.py                              :: 10 câu test 3.3 (hoặc --ids T02 T04)
python dashboard.py                              :: in tóm tắt + ghi dashboard.html
```

Thêm `--no-langfuse` để chỉ ghi JSONL, không gửi Langfuse.
