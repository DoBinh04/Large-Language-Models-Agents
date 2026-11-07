# Báo cáo Assignment 3.3 — ReAct Agent from Scratch

## 1. Mục tiêu

Xây dựng một ReAct agent bằng Python **không dùng agent framework** (không LangChain, LangGraph hay SDK tool runner). Agent tự điều phối ba công cụ:

- `search_web(query)`: tìm thông tin mới trên Internet.
- `calculate(expr)`: tính biểu thức số học an toàn.
- `get_current_time()`: lấy thời điểm hiện tại theo múi giờ IANA.

Agent được đánh giá qua 10 câu hỏi nhiều bước, buộc kết hợp tìm kiếm/thời gian với tính toán.

## 2. Kiến trúc và cách triển khai

```
Question
   │
   ▼
LLM sinh text ReAct: Thought → Action → Action Input
   │                              │
   │                              ▼
   │                     parser.py kiểm tra cú pháp
   │                              │
   ▼                              ▼
Final Answer                  tools.py thực thi tool
                                  │
                                  ▼
                         Observation trả lại LLM
                                  │
                                  └── lặp đến Final Answer / max_steps
```

| Thành phần | File | Vai trò |
|---|---|---|
| Vòng lặp agent | `react_agent/agent.py` | Gọi LLM, parse hành động, chạy tool, thêm Observation và giới hạn vòng lặp. |
| Prompt | `react_agent/prompt.py` | Khai báo tool và ép định dạng `Thought` / `Action` / `Action Input` / `Final Answer`. |
| Parser | `react_agent/parser.py` | Đọc phản hồi text của LLM; hỗ trợ JSON và input chuỗi trần. |
| Tool registry | `react_agent/tools.py` | Cài đặt 3 tool, kiểm tra tham số và chuyển lỗi thành Observation. |
| LLM adapter | `react_agent/llm.py` | Gọi Claude và lưu message theo kiểu append-only. |
| Test runner | `run_tests.py` | Chạy 10 câu, chấm đáp án và tạo trace/report. |

Điểm cốt lõi của cách làm “from scratch” là LLM **chỉ sinh văn bản**. Code trong repo tự parse `Action`, tự gọi hàm Python và tự nối `Observation` vào hội thoại. Không sử dụng native tool use của API.

### Cơ chế ReAct

Ví dụ một lượt xử lý:

```text
Thought: Cần biết hai đại lượng trước khi chia.
Action: search_web
Action Input: {"query": "Vietnam population 2023"}
Observation: Kết quả thật từ search_web
Thought: Đã có số liệu, cần tính chính xác.
Action: calculate
Action Input: {"expr": "..."}
Observation: Kết quả thật từ calculate
Thought: Đủ thông tin để trả lời.
Final Answer: ...
```

`Observation:` được đặt làm stop sequence khi gọi LLM. Nhờ vậy model không thể tự tạo dữ liệu quan sát giả; chỉ chương trình mới thêm Observation sau khi tool thật chạy xong. Agent cũng dùng `max_steps=10` để chặn lặp vô hạn. Lỗi parse, tool không tồn tại hoặc biểu thức không hợp lệ không làm chương trình dừng mà được trả về dưới dạng `Observation: ERROR: ...` để model tự sửa.

## 3. Cài đặt các tool

### `search_web(query)`

Tool thử theo thứ tự Tavily (nếu đặt `TAVILY_API_KEY`), DuckDuckGo qua `ddgs`, sau đó Wikipedia API. Kết quả được chuẩn hoá thành tiêu đề, mô tả ngắn và URL; mỗi Observation bị giới hạn 3.000 ký tự để tránh làm phình context.

### `calculate(expr)`

Không dùng `eval()`. Tool parse biểu thức bằng `ast`, chỉ cho phép số, `+ - * / // % **`, unary `+/-`, các hằng `pi`, `e` và một whitelist hàm `math` như `sqrt`, `log`, `ceil`, `round`. Lũy thừa quá lớn bị từ chối. Ví dụ `__import__('os')` và `open('x')` đều trả về lỗi thay vì được thực thi.

### `get_current_time()`

Sử dụng `zoneinfo` với múi giờ IANA, mặc định `Asia/Ho_Chi_Minh`. Observation là JSON gồm ngày, giờ, UTC offset, thứ và ngày thứ bao nhiêu trong năm. Nhờ đó agent có dữ liệu để giải các câu phụ thuộc thời gian chạy thực tế.

## 4. Bộ đánh giá 10 câu multi-step

| ID | Bài toán | Tool kỳ vọng |
|---|---|---|
| T01 | So sánh dân số Nhật Bản và Việt Nam | search ×2, calculate |
| T02 | Đếm ngày đến 01/01/2027 | current time, calculate |
| T03 | So sánh giờ Tokyo và New York | current time ×2, calculate |
| T04 | Tính lãi kép 5 năm | calculate ×2 |
| T05 | Vô địch World Cup 2022 và số năm đã qua | search, current time, calculate |
| T06 | Thời gian ánh sáng từ Mặt Trời đến Trái Đất | search, calculate |
| T07 | Xác định thứ sau 100 ngày | current time, calculate |
| T08 | Số tháp Eiffel vượt Everest | search ×2, calculate |
| T09 | GDP bình quân đầu người Việt Nam 2023 | search ×2, calculate |
| T10 | Phần trăm năm đã trôi qua | current time, calculate |

Mỗi lượt chạy live sinh `results/<timestamp>/summary.md` (đúng/sai, số bước, coverage tool, tokens và latency) và `traces.jsonl` (toàn bộ Thought → Action → Observation) để có thể kiểm tra lại quá trình suy luận.

## 5. Kết quả chạy đã xác minh

Đã chạy thành công bộ test offline tại workspace:

```text
python tests\test_offline.py

OK test_calculate
OK test_time
OK test_parser
OK test_agent_loop
OK test_checks
```

Kết quả: **5/5 nhóm test đạt**. Các kiểm tra này bao phủ evaluator an toàn, múi giờ, parser, cơ chế phục hồi khi sai format trong vòng lặp ReAct và bộ chấm đáp án.

Chưa có file `.env` với `ANTHROPIC_API_KEY` trong workspace hiện tại, vì vậy chưa thể chạy và báo cáo trung thực kết quả live 10/10. Sau khi cấu hình key, chạy lệnh bên dưới; file `summary.md` tạo ra là số liệu thực nghiệm cần nộp kèm báo cáo.

## 6. Hướng dẫn chạy

```bat
cd "C:\phase3\assignment 33"
pip install -r requirements.txt
copy .env.example .env
```

Điền `ANTHROPIC_API_KEY` vào `.env`, sau đó:

```bat
python tests\test_offline.py
python ask.py "Bây giờ ở Tokyo là mấy giờ?"
python run_tests.py
```

Có thể chạy một phần bộ đề hoặc đổi model:

```bat
python run_tests.py --ids T02 T07 --quiet
python run_tests.py --model claude-haiku-5-5 --max-steps 8
```

## 7. Nhận xét và giới hạn

- Stop sequence và Observation do chương trình tạo giúp giảm nguy cơ LLM bịa kết quả tool.
- `calculate` an toàn hơn `eval`, tuy nhiên chỉ hỗ trợ tập biểu thức có chủ đích; đây là đánh đổi để giảm rủi ro thực thi mã tùy ý.
- Chất lượng `search_web` phụ thuộc mạng và nguồn tìm kiếm; các số liệu như dân số/GDP có thể chênh lệch theo nguồn hoặc năm thống kê.
- Độ đúng của 10 câu live phụ thuộc model, prompt và thời điểm chạy. Vì thế trace và summary được lưu để tái lập, thay vì chỉ báo cáo đáp án cuối.

## 8. Kết luận

Project đã hiện thực đầy đủ một ReAct agent từ scratch với ba tool yêu cầu, cơ chế đa bước, xử lý lỗi và test offline đạt 5/5. Khi được cấp API key, `run_tests.py` sẽ tự động thực nghiệm toàn bộ 10 câu và xuất báo cáo định lượng để hoàn thiện phần đánh giá live.
