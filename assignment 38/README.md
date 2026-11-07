# Báo cáo Assignment 3.8: Golden test set và 3 automated evals cho RAG pipeline

## 1. Đề bài

Tạo golden test set gồm 20 câu hỏi cho RAG pipeline, sau đó implement 3 automated eval:

| Metric | Yêu cầu |
|---|---|
| **Faithfulness** | LLM-as-judge: câu trả lời có dựa trên context không? |
| **Answer Relevance** | Cosine similarity giữa embedding của answer và của question |
| **Context Precision** | % chunk được retrieve thực sự liên quan |

Pipeline được đánh giá là RAG của bài 3.2 (`assignment 32`). Tài liệu nguồn là sổ tay vận hành một đài thiên văn giả định, khoảng 3.700 từ và 27 heading. Pipeline gồm: chunking → embedding `BAAI/bge-small-en-v1.5` → vector store in-memory → top-k → prompt có trích dẫn `[n]` → Claude sinh câu trả lời.

## 2. Thiết lập thí nghiệm

| Thành phần | Giá trị |
|---|---|
| Generator | Claude Haiku 5.5 (`claude-haiku-5-5`), `max_tokens = 4096` |
| Judge | Claude Haiku 5.5, `effort = medium`, output ràng buộc bằng JSON Schema |
| Embedder | `BAAI/bge-small-en-v1.5` (384 chiều), chạy local |
| Cấu hình chính | chunking `recursive` (≈300 token), top-k = 5 |
| Cấu hình so sánh | `fixed_256` (overlap 32), `fixed_512` (overlap 64), `recursive` với k = 3 |
| Số lời gọi API | 20 lần sinh câu trả lời + 40 lần judge cho mỗi cấu hình |
| Chi phí | Judge dùng khoảng 112k token vào / 14k token ra mỗi lần chạy (Haiku 5.5: $0.10 / $0.50 cho 1M token), tổng chưa đến 0.05 USD mỗi lần chạy |

## 3. Golden test set (`data/golden_set.json`)

Mỗi câu gồm `id`, `type`, `question`, `answer` (đáp án chuẩn) và `evidence`, là các câu **trích nguyên văn** từ tài liệu. Script dùng `EvidenceLocator` để kiểm tra mọi evidence đều tìm thấy trong tài liệu, rồi đổi thành offset ký tự `(start, end)`. Context Precision dựa vào các offset này để gán nhãn ground truth.

Bộ câu hỏi được thiết kế để mỗi câu thử một điểm yếu cụ thể của RAG:

| Nhóm | Câu | Mục đích kiểm tra |
|---|---|---|
| Fact đơn + số gây nhiễu | G01, G05, G11 | Tài liệu có nhiều tốc độ gió / nhiệt độ / ngày tháng khác nhau |
| Multi-evidence | G02, G08, G12, G17 | Cần ≥ 2 câu evidence, đo precision và độ đầy đủ |
| Paraphrase | G03, G09, G16 | Câu hỏi ít trùng từ với tài liệu, thử khả năng hiểu nghĩa của embedding |
| Evidence dài (danh sách) | G04 | Danh sách có đánh số, dễ bị cắt ngang ở biên chunk |
| Multi-hop | G06 (2 section xa nhau), G13 (phải suy luận tháng 10 thuộc Semester B) | Giới hạn của retrieval chỉ dùng một query |
| "Vì sao" / điều kiện | G07, G10, G14, G15 | Phải giữ đúng quan hệ nhân quả / điều kiện |
| Tính toán | G18 (45% × 14 triệu) | Claim "6.3 million" không có nguyên văn, judge phải coi là *suy ra trực tiếp* |
| Thời gian | G19 | |
| **Không trả lời được** | G20 (giá vé tham quan) | Model phải từ chối, kiểm tra judge xử lý abstention |

G01–G10 lấy lại từ bài 3.2, G11–G20 viết mới.

## 4. Cài đặt 3 metric

### 4.1 Faithfulness: LLM-as-judge (`evals/faithfulness.py`)

Dùng **một** lời gọi judge, output là JSON theo schema cố định. Trong lời gọi đó judge làm hai bước (giống RAGAS):

1. Tách câu trả lời thành các **claim nguyên tử** (bỏ qua ký hiệu trích dẫn, câu đệm, câu nhắc lại đề).
2. Gán mỗi claim một nhãn `supported` / `unsupported` / `contradicted`, **chỉ so với các chunk đã retrieve**. Judge không được dùng kiến thức riêng hay đáp án chuẩn. Phép tính đơn giản và paraphrase được tính là supported.

```
faithfulness = #supported / #claims
```

Nếu câu trả lời là từ chối thì không có claim nào. Khi đó score = 1.0 (vì model không bịa) và câu được gắn cờ `abstained`. Mỗi claim được lưu kèm `chunk_ids` và lý do, để kiểm tra lại bằng tay.

### 4.2 Answer Relevance: cosine similarity (`evals/answer_relevance.py`)

```
relevance = cos(embed(question), embed(answer))
```

Dùng chính embedder bge-small của pipeline. Cả hai phía được embed như passage để phép so sánh đối xứng, và ký hiệu `[1][2]` được bỏ trước khi embed. Cosine giữa câu hỏi và câu trả lời hiếm khi gần 1, nên script tính thêm một **mốc tham chiếu** là `cos(question, đáp án chuẩn)`, kèm cột `normalized = relevance / mốc`.

### 4.3 Context Precision (`evals/context_precision.py`)

Mỗi chunk trong top-k được gán nhãn "liên quan" theo **hai cách độc lập**:

- **Ground truth (GT)**: chunk chứa ≥ 30% số ký tự của ít nhất một evidence (tính bằng offset). Cách này tất định và không tốn API.
- **LLM judge**: judge xem chunk có chứa thông tin cần để tạo ra đáp án chuẩn không. Cách này bắt được chunk hữu ích nằm ngoài evidence.

Với mỗi cách, script báo hai số:

```
precision@k = #chunk liên quan / k
AP@k        = Σ_i precision@i · rel_i / #chunk liên quan     (tương tự "context precision" của RAGAS, thưởng khi chunk đúng đứng đầu)
```

Script cũng báo **tỉ lệ đồng thuận nhãn** giữa GT và LLM. G20 không có chunk "đúng" nên bị loại khỏi trung bình precision.

## 5. Kết quả

### 5.1 Cấu hình chính: `recursive`, k = 5

| Metric | Giá trị | Ghi chú |
|---|---|---|
| **Faithfulness** | **0.983** | 1/59 claim unsupported (G02) |
| **Answer relevance** | **0.768** | mốc cos(question, đáp án chuẩn) = 0.608, normalized = 1.27 |
| **Context precision@5 (GT)** | **0.200** | AP@5 = 0.825 |
| Context precision@5 (LLM) | 0.200 | AP@5 = 0.825, đồng thuận với GT = 100% |
| Hit rate (≥ 1 chunk đúng trong top-5) | 18/19 | trượt G15 |
| Abstention | G20 từ chối đúng | G15 từ chối nhầm vì retrieve trượt |

Kết quả từng câu:

| ID | Faith | Relevance | P@5 (GT) | AP@5 | Ghi chú |
|---|---|---|---|---|---|
| G01 | 1.00 | 0.920 | 0.20 | 1.00 | |
| G02 | **0.67** | 0.806 | 0.20 | 1.00 | 1 claim suy diễn (xem 6.2) |
| G03 | 1.00 | 0.654 | 0.20 | 1.00 | paraphrase, relevance thấp nhất trong các câu trả lời đúng |
| G04 | 1.00 | 0.806 | 0.20 | 1.00 | |
| G05 | 1.00 | 0.787 | 0.20 | 1.00 | |
| G06 | 1.00 | 0.757 | 0.40 | 0.58 | multi-hop: lấy đủ 2 section nhưng chunk đúng ở hạng 2 và 3 |
| G07 | 1.00 | 0.803 | 0.20 | 1.00 | |
| G08 | 1.00 | 0.824 | 0.20 | 1.00 | |
| G09 | 1.00 | 0.777 | 0.20 | 1.00 | |
| G10 | 1.00 | 0.887 | 0.20 | 0.50 | |
| G11 | 1.00 | 0.801 | 0.20 | 1.00 | |
| G12 | 1.00 | 0.876 | 0.20 | 1.00 | |
| G13 | 1.00 | 0.727 | 0.20 | 1.00 | suy luận tháng 10 thuộc Semester B đúng |
| G14 | 1.00 | 0.830 | 0.20 | 0.25 | chunk đúng ở hạng 4 |
| G15 | 1.00 ✓abstain | 0.466 | **0.00** | 0.00 | **retrieve trượt** section "8.4 Fire" |
| G16 | 1.00 | 0.809 | 0.20 | 1.00 | |
| G17 | 1.00 | 0.768 | 0.20 | 0.33 | |
| G18 | 1.00 | 0.769 | 0.20 | 1.00 | judge chấp nhận phép tính 45% × 14 = 6.3 |
| G19 | 1.00 | 0.848 | 0.20 | 1.00 | |
| G20 | 1.00 ✓abstain | 0.439 | — | — | từ chối đúng |

### 5.2 So sánh cấu hình

| Cấu hình | Faithfulness | Relevance | P@k (GT) | AP (GT) | P@k (LLM) | Đồng thuận GT/LLM | Hit rate | Context TB (ký tự) |
|---|---|---|---|---|---|---|---|---|
| recursive, k=5 | 0.983 | 0.768 | 0.200 | 0.825 | 0.200 | 1.000 | 18/19 | 6.403 |
| recursive, k=3 | **1.000** | 0.770 | **0.316** | 0.811 | 0.333 | 0.982 | 17/19 | **3.899** |
| fixed_256, k=5 | 0.988 | 0.770 | 0.221 | **0.883** | 0.284 | 0.937 | 18/19 | 6.303 |
| fixed_512, k=5 | 0.983 | **0.814** | 0.232 | 0.717 | 0.263 | 0.947 | **19/19** | 12.668 |

Kết quả chi tiết của từng lần chạy (`summary.md`, `per_question.csv`, `details.jsonl`) nằm trong `results/`:
`20261107-174905` (recursive k=5), `20261107-175655` (recursive k=3), `20261107-175150` (fixed_256), `20261107-175422` (fixed_512).

## 6. Phân tích

### 6.1 Context precision thấp là do cách tính, chưa chắc retrieval kém

Precision@5 chỉ đạt 0.20, nhưng hit rate là 18/19 và AP@5 là 0.825. Nghĩa là chunk đúng gần như luôn có mặt và thường đứng đầu. Precision thấp vì **evidence của mỗi câu thường chỉ nằm trong 1 chunk**: khi chỉ có 1 chunk liên quan thì precision@5 tối đa chỉ là 1/5 = 0.20. Ngay cả câu multi-evidence (G02, G12, G17) cũng có các câu evidence nằm liền nhau trong cùng một chunk.

Vì vậy, **precision@k cần đọc cùng với hit rate và AP**:
- Giảm k từ 5 xuống 3 làm precision tăng từ 0.20 lên 0.32 và context ngắn đi gần một nửa. Đổi lại, hit rate giảm từ 18 xuống 17: G14 bị mất chunk chứa lý do "tháng 2 nhiều mây nhất".
- AP của `fixed_512` thấp nhất (0.717): chunk lớn chứa nhiều chủ đề nên điểm similarity bị "loãng", chunk đúng hay rơi xuống hạng dưới. Tuy vậy, chunk lớn lại là cấu hình **duy nhất** lấy được G15 (fire alarm), vì section 8.4 nằm chung chunk với nội dung khác có điểm cao hơn.
- `fixed_256` có AP cao nhất (0.883): chunk nhỏ, tập trung nên dễ xếp đúng lên đầu.

### 6.2 Faithfulness gần tuyệt đối, và những claim "unsupported" đều là suy diễn có rào đón

Qua 4 cấu hình chỉ có 3 claim bị đánh unsupported, và cả 3 đều là suy diễn mà generator **đã tự nói rõ là suy diễn**:

- **G02**: "DDT được gửi thẳng cho Director thay vì để TAC xếp hạng". Tài liệu chỉ nói yêu cầu DDT gửi cho Director, không nói TAC không xét. Chính câu trả lời cũng viết *"the context doesn't explicitly say… so that point is an inference"*.
- **G14 (fixed_256)**: "việc chọn tháng 2 *có thể* nhằm giảm thời gian mất quan sát". Câu trả lời ghi rõ *"the document doesn't confirm that reason"*.

Judge áp dụng đúng định nghĩa (claim không có trong context thì unsupported), không bị đánh lừa bởi câu rào đón. Ngược lại, ở **G18**, claim "6.3 million" không có nguyên văn trong tài liệu nhưng judge vẫn chấm supported, vì đây là phép tính trực tiếp 45% × 14. Kết quả này đúng như hướng dẫn trong prompt.

Không thấy claim `contradicted` nào, tức Haiku 5.5 không bịa thông tin trái với tài liệu. Hai câu từ chối (G15, G20) đều được nhận diện là `abstained`.

### 6.3 Faithfulness khác correctness, và đây là lý do cần nhiều metric

Hai trường hợp cho thấy faithfulness = 1.0 vẫn có thể là câu trả lời **không đạt**:

- **G15 (recursive)**: retrieve trượt section "Fire" nên model nói "I don't know". Faithfulness = 1.0 vì model không bịa, nhưng câu hỏi trả lời được mà lại không được trả lời. Lỗi này chỉ lộ ra ở **context precision = 0** và cờ "từ chối nhầm".
- **G14 (k=3)**: top-3 chỉ có chunk nói "recoat vào tháng 2", không có chunk giải thích lý do. Model trả lời trung thực rằng tài liệu không nêu lý do. Kết quả là faithfulness = 1.0 và relevance = 0.888 (cao), nhưng câu hỏi "vì sao" **không được trả lời**. Precision (GT) = 0 là tín hiệu duy nhất bắt được lỗi này.

### 6.4 Điểm mù của Answer Relevance (cosine)

- **Không phân biệt được trả lời đúng với từ chối có nhắc chủ đề.** Ở `fixed_512`, câu trả lời G20 là *"I don't know… public summit visits are offered on Saturday afternoons… but it does not state any ticket price"* nhưng vẫn có relevance **0.812**, cao hơn G03 vốn trả lời đúng (0.657). Trong khi đó, khi chỉ nói "I don't know based on the provided document" thì relevance chỉ còn 0.44.
- **`normalized` > 1 không có nghĩa là tốt hơn đáp án chuẩn.** Đáp án chuẩn viết ngắn gọn, còn câu trả lời của model thường lặp lại các từ trong câu hỏi, nên cosine với câu hỏi cao hơn. Metric này đo "nói đúng chủ đề", không đo "nói đúng".
- **Câu paraphrase (G03) bị điểm thấp** dù trả lời đúng, vì câu hỏi tả triệu chứng ("confused, struggles to breathe") còn câu trả lời dùng từ chuyên môn của tài liệu (oxygen, Midway Lodge).

Vì vậy relevance chỉ hữu ích để **so sánh tương đối** và phát hiện câu trả lời lạc đề hoàn toàn. Không thể dùng nó làm thước đo chất lượng riêng lẻ.

### 6.5 Độ tin cậy của LLM judge

- **Đồng thuận nhãn chunk GT/LLM** đạt 94–100%. Bất đồng thường gặp nhất là LLM đánh "relevant" cho chunk **chứa một phần thông tin** mà GT không tính vì phần evidence trong chunk chưa đủ 30%. Ví dụ ở G14 (k=3), chunk nói "recoat vào tháng 2" được LLM coi là relevant nhưng không chứa evidence lý do. Vì thế precision (LLM) cao hơn precision (GT) ở `fixed_256` và k=3. Chiều ngược lại cũng có: ở `fixed_512`, câu G06 có một chunk chứa ≥ 30% evidence nhưng LLM đánh là không liên quan.
- **Kiểm tra tay**: mình đọc lại verdict của judge cho toàn bộ claim của G02, G06, G13, G18 (13 claim), claim unsupported của G14 (fixed_256) và 2 câu từ chối G15/G20, đối chiếu với tài liệu gốc. Mình đồng ý với **14/14** verdict và cả 2 lần nhận diện từ chối. Mẫu kiểm tra còn nhỏ, nên cần đọc thêm để có tỉ lệ đáng tin hơn.
- **Hạn chế**: judge và generator là **cùng một model** (Haiku 5.5), nên có thể có self-preference bias. Hướng khắc phục là chấm lại cùng câu trả lời bằng model mạnh hơn mà không phải sinh lại:
  `python run_evals.py --reuse results\20261107-174905\details.jsonl --judge-model claude-opus-5-5`.

## 7. Kết luận

1. Golden set 20 câu đủ đa dạng để làm lộ các lỗi cụ thể: retrieve trượt (G15), multi-hop bị mất một nửa ở k nhỏ (G14), từ chối đúng (G20) và suy diễn có rào đón (G02).
2. Với Haiku 5.5, **generator rất trung thực**: faithfulness ≥ 0.98 ở mọi cấu hình, không có claim mâu thuẫn. **Điểm nghẽn của pipeline nằm ở retrieval**, không nằm ở generation.
3. Không metric nào đủ khi dùng một mình. Faithfulness không bắt được câu trả lời thiếu ý, relevance không phân biệt được từ chối với trả lời đúng, còn precision@k bị chặn trên bởi số chunk chứa evidence. Phải đọc cả ba metric cùng hit rate và AP.
4. Về cấu hình: `recursive` với k = 5 là cân bằng tốt nhất. `fixed_512` lấy đủ evidence nhất (19/19) nhưng context dài gấp đôi và xếp hạng kém hơn. Giảm k xuống 3 thì context gọn hơn nhưng mất câu multi-hop.
5. Hướng cải thiện: thêm **reranker** để tăng AP, dùng **hybrid search (BM25 + vector)** để bắt từ khoá như "fire alarm" (G15), và đổi judge sang model mạnh hơn để kiểm tra chéo.

## 8. Cách chạy lại

```bat
cd "C:\phase3\assignment 38"
pip install -r requirements.txt
:: tạo file .env có dòng ANTHROPIC_API_KEY=...

python -m tests.test_offline                  :: kiểm tra golden set + metric, không cần API key
python run_evals.py                           :: recursive, k=5
python run_evals.py --strategy fixed_256
python run_evals.py --strategy fixed_512
python run_evals.py --k 3
python run_evals.py --no-llm-precision        :: bỏ 20 lời gọi judge cho precision
python run_evals.py --reuse results\<run>\details.jsonl --judge-model claude-opus-5-5   :: đổi judge, giữ nguyên câu trả lời
```

Model mặc định là `claude-haiku-5-5` cho cả generator và judge. Có thể đổi bằng `--gen-model` / `--judge-model` hoặc bằng biến `ANTHROPIC_MODEL` / `EVAL_JUDGE_MODEL` trong `.env`.

Cấu trúc thư mục:

```
data/golden_set.json          20 câu hỏi + đáp án chuẩn + evidence nguyên văn
evals/judge.py                client LLM-as-judge (structured output)
evals/faithfulness.py         metric 1
evals/answer_relevance.py     metric 2
evals/context_precision.py    metric 3 (GT + LLM, precision@k + AP@k)
run_evals.py                  chạy pipeline 3.2 trên golden set, ghi results/<thời điểm>/
tests/test_offline.py         test không cần API
results/                      summary.md, per_question.csv, details.jsonl của từng lần chạy
```
