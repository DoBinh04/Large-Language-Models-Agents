# Assignment 3.9: Small LLM cho Multi-Agent Orchestration (TechViet Customer Support)

> Mọi số liệu dưới đây là kết quả chạy thật ngày 2026-11-07 trên máy thuê 4× RTX 5090 (vast.ai), file gốc trong
> `results/eval/*/summary.json`, `results/eval/*/details.jsonl` (lý do của Judge từng kịch bản), `results/routing/*.json`,
> trace từng span trong `results/traces/*.jsonl`, sổ chi tiêu `results/spend.json`.
> API key: Anthropic (không có gpt-oss), nên baseline của đề **gpt-oss-120b @ medium reasoning** được thay bằng
> **Claude Opus 5.5 @ effort `medium`** (`output_config.effort` là mức reasoning của Claude).

**Mục lục:** 1. Tóm tắt · 2. Hệ thống · 3. Thiết lập thí nghiệm · 4. Routing-only · 5. End-to-end · 6. Phân tích lỗi · 7. Hạn chế · 8. Chi phí · 9. Hướng dẫn cài đặt & chạy

## 1. Tóm tắt

| | Kết quả chính |
|---|---|
| Small LLM | Qwen2.5-1.5B-Instruct, LoRA bf16 (r=16) → SFT (4.2 phút) → DPO (10.1 phút) trên 1 GPU 5090 |
| Routing (cùng 97 quyết định) | base 46.4% (JSON hợp lệ 12%) → **SFT 99.0%** → DPO 95.9% (JSON 90.7%); API: Haiku@low 91.8%, Opus@medium 94.8% |
| End-to-end (32 kịch bản, LLM Judge) | Baseline Opus@medium: **81.2% PASS**, Tool F1 0.673, 12.0 s/lượt, **0.0360 $/kịch bản** |
| Cấu hình tốt nhất về chi phí | `api-haiku-low`: 87.5% PASS, F1 0.724, 6.1 s/lượt, **0.00086 $/kịch bản (−98%)**; router SFT tự host + agent rẻ (`opt-sft-mixed`): 81.2% PASS, F1 0.738, **0.00558 $ (−85%)**, routing 100% |
| DPO end-to-end | router DPO + agent Opus (`small-dpo-opus`): 78.1% PASS, **−34% chi phí** so với baseline; router DPO + agent rẻ: 71.9% PASS |
| Tổng chi API cho toàn bộ thí nghiệm | **7.77 $** / giới hạn 20 $ (1 563 lời gọi; có bộ chặn cứng `API_BUDGET_USD`) |

Kết luận ngắn: (1) SFT là bước tạo khác biệt lớn nhất cho Small LLM (46% → 99% routing, JSON 12% → 100%), vượt cả router API
trên tập routing; (2) với hệ thống này, **model lớn hơn / reasoning cao hơn không làm tăng tỉ lệ hoàn thành**: Opus high chỉ hơn
medium 1 kịch bản (84.4% vs 81.2%), Sonnet medium và Haiku low đạt 87.5% với chi phí thấp hơn 63% và 98%;
(3) chất lượng bị giới hạn chủ yếu ở **Policy agent** (đổi trả / bảo hành), không phải ở router;
(4) **DPO lần này làm router kém đi so với SFT** (routing 99.0% → 95.9%, JSON hợp lệ 100% → 90.7%, câu multi 100% → 65%), nguyên nhân phân tích ở mục 6: tập cặp lệch về phía phạt "gọi thêm agent" và DPO bị over-optimize. Bản dùng để triển khai nên là **SFT**.

## 2. Hệ thống đã xây (ánh xạ yêu cầu → code)

### Kiến trúc

```
Khách ──► Orchestrator ─────────────────────────────────────────────────────────────┐
           │ 1. regex slot filling (DHxxxx, SĐT, Pxxx) ─► ConversationState           │
           │ 2. Router.decide(state, msg) ─► {intent, agents, plan, missing_info, ...}│  span: turn / router
           │      ├─ APIRouter   : Claude + JSON Schema (structured output)           │
           │      ├─ SmallLLM    : Qwen2.5-1.5B (+LoRA SFT / +LoRA DPO)                │
           │      └─ RuleRouter  : từ khoá (baseline)                                  │
           │ 3. thiếu mã đơn ─► hỏi lại, lưu pending_intent (không gọi agent nào)       │
           │ 4. chạy plan: A2A message/send ─► Task (completed | input-required)        │  span: a2a.send
           ▼                                                                           │
   ┌───────────────┬───────────────┬────────────────┐                                  │
   │ Product Agent │  Order Agent  │  Policy Agent  │  (Claude tool-use loop)          │  span: agent / llm
   └──────┬────────┴──────┬────────┴───────┬────────┘                                  │
          └──── LocalTools | MCPTools (stdio → support/mcp_server.py) ────┘            │  span: tool
                 search_products, get_product_details, compare_products,               │
                 search_knowledge_base / search_policy ─► Hybrid RAG (BM25 + dense     │
                 → RRF → cross-encoder rerank), lookup_order, list_orders_by_phone,    │
                 cancel_order, check_return_eligibility, create_return_request,        │
                 check_warranty, create_warranty_ticket                                │
           5. 1 agent ─► dùng thẳng câu trả lời; ≥2 agent ─► Composer gộp ◄────────────┘
```

### Ánh xạ yêu cầu → code

| Yêu cầu | Cài đặt |
|---|---|
| Multi-Agent: Orchestrator + Product / Order / Policy | `support/orchestrator.py` (slot filling → Router → plan → A2A → composer), `support/agents.py` |
| Routing, Planning, State-aware | Router trả JSON `{intent, agents, plan, missing_info, entities, reply}`; `ConversationState` giữ entities, `pending_intent`, `last_agents`, lịch sử (`support/state.py`, `support/router.py`) |
| Fine-tune 1B–3B LoRA/QLoRA | `training/sft_train.py` (TRL SFTTrainer, loss chỉ trên completion; `--qlora` 4-bit cho GPU nhỏ) |
| DPO | `training/gen_dpo_pairs.py` (8 kiểu lỗi điều phối + on-policy), `training/dpo_train.py` (TRL DPOTrainer) |
| RAG Hybrid + Rerank | `support/rag.py`: BM25 + dense (multilingual-e5-small) → RRF → cross-encoder bge-reranker-v2-m3; query rewrite `support/query_rewriter.py` |
| Tool use | 12 tool có JSON Schema, phân quyền theo agent (`support/tools.py`) |
| MCP & A2A | `support/mcp_server.py` (FastMCP stdio / Streamable HTTP), `support/a2a.py` (AgentCard, JSON-RPC `message/send`, Task lifecycle) |
| LLMOps | span tree + token + cost + latency → JSONL (`support/tracing.py`), dashboard `eval/report.py`, JSON log có trace_id, prompt caching, refusal fallback, **sổ chi tiêu + chặn ngân sách** (`support/llm.py`) |
| Evaluation | `eval/routing_eval.py` (routing-only), `eval/run_eval.py` (end-to-end), `eval/judge.py` (LLM Judge), `eval/metrics.py` (Tool F1) |

## 3. Thiết lập thí nghiệm

**Dữ liệu train router.**
- Template (`training/gen_data.py`): 1 500 mẫu có nhãn xác định theo luật, đủ các nhóm state-aware (single, missing, pending, in_state, follow_product, switch, multi, chitchat, out_of_scope).
- Hugging Face [`Tson29/techviet-cs-orchestrator-data`](https://huggingface.co/datasets/Tson29/techviet-cs-orchestrator-data): dataset này dạy orchestrator **từng bước** (`call_agent / ask_user / finish`), khác schema router của hệ thống (quyết định **cả lượt**). `training/hf_import.py` mô phỏng lại `ConversationState` và gộp các bước của mỗi lượt thành một RoutingDecision: 412 lượt → giữ **402** (bỏ 9 lượt "finish không gọi agent" không có nhãn tương ứng, 1 lượt hỏi mã đơn khi đã có mã). Chia theo hội thoại 80/10/10 để không lộ lượt giữa các split; mã đơn DH10xx được đổi sang dải khác để không trùng tập eval.
- Tổng: train 1 815, val 340, test 343 mẫu.

**SFT.** Qwen2.5-1.5B-Instruct, LoRA r=16 / α=32 trên mọi projection, bf16 (5090 có 32 GB nên không cần QLoRA), lr 2e-4 cosine, 2 epoch, batch hiệu dụng 16, 228 step, **4.2 phút**.
train loss 0.626 → 0.019; eval loss 0.053 → **0.037**, token accuracy (eval) **0.991**.

**DPO.** 2 711 cặp train / 527 val, rejected sinh có chủ đích theo 8 kiểu lỗi điều phối: `wrong_agent` 1 251, `extra_agent` 451, `state_unaware` 342 (hỏi lại mã đơn đã có), `drop_agent` 257, `no_ask` 137 (đoán khi thiếu mã), `bad_format` 118, `ignore_pending` 81, `chat_routed` 63, cộng **11 cặp on-policy** (sample 120 prompt × 4 từ model SFT, chỉ 11 output sai: model SFT đã rất ít lỗi).
β = 0.1, lr 5e-6, 1 epoch, 170 step (**10.1 phút**, train loss 0.080). Log: `rewards/accuracies` 0.71 → 1.0, `rewards/margins` 0.09 → 6.6; trên tập val cuối: eval loss **0.0036**, reward chosen −0.28 / rejected −7.24. Chosen giữ gần 0 còn rejected bị đẩy xuống mạnh: DPO phạt đúng các quyết định sai thay vì làm lệch quyết định đúng.
Ghi chú kỹ thuật: TRL 1.17 không nhận `ref_model` là PeftModel; script tự chuyển sang chế độ PEFT của TRL. `rewards/chosen` ≈ 0 ở step đầu cho thấy reference trùng với policy ban đầu (tức model SFT), đúng thiết kế "ref = SFT đóng băng".

**Eval end-to-end.** `data/eval_set.json`: 32 kịch bản / 37 lượt (product 6, order 4, policy 12 gồm các ca biên ngày thứ 16 đổi ý, tai nghe đã mở hộp, máy đã kích hoạt, rơi vỡ, đơn đã huỷ…; multi 3, state-aware nhiều lượt 5, chitchat / out-of-scope 2). Ngày hệ thống cố định 2026-10-29.
- **LLM Judge**: Claude Sonnet 5.5 @ low, structured output: chấm từng tiêu chí → lý do → PASS/FAIL; FAIL khi sai số / ngày / mã, bịa, thực hiện hành động ghi không được yêu cầu, không hỏi lại khi cần. Judge khác model với baseline để giảm thiên vị; chi phí Judge tính riêng.
- **Tool-call F1**: call khớp khi cùng tên + mọi tham số kỳ vọng bằng nhau; ghép 1-1; báo micro (P/R) và macro theo kịch bản.
- **Cost**: token thực tế × bảng giá (input, output gồm thinking, cache write ×1.25, cache read); Small LLM: thời gian generate × 0.35 $/giờ GPU.
- **Latency**: wall-clock mỗi lượt, gồm RAG (dense + reranker trên GPU) và tool.

## 4. Routing-only: API LLM vs Small LLM vs SFT vs DPO

Cùng 97 quyết định cho mọi router: 37 lượt của eval set (STATE mô phỏng các lượt trước đúng) + 60 mẫu đầu tập test.

| Router | Routing acc | Intent acc | Eval set (37) | Test (60) | JSON hợp lệ | Latency mean / p95 (ms) | $ / 1 000 quyết định |
|---|---|---|---|---|---|---|---|
| Rule (từ khoá) | 56.7% | 53.6% | 56.8% | 56.7% | 100% | 0 / 0 | 0 |
| Small LLM base (Qwen2.5-1.5B) | 46.4% | 43.3% | 51.4% | 43.3% | **12.4%** | 655 / 994 | 0.064 |
| **SFT** | **99.0%** | **99.0%** | 97.3% | 100% | 100% | 2 210 / 2 742 | 0.215 |
| DPO | 95.9% | 95.9% | 94.6% | 96.7% | 90.7% | 2 510 / 4 394 | 0.244 |
| API Haiku 5.5 @ low | 91.8% | 86.6% | 94.6% | 90.0% | 100% | 1 410 / 2 557 | 0.129 |
| API Opus 5.5 @ medium (baseline) | 94.8% | 92.8% | **100%** | 91.7% | 100% | 2 716 / 4 247 | **4.742** |

Theo nhóm (157 mẫu, base → SFT → DPO): multi 0% → 100% → **65%**, pending (khách chỉ gửi mã đơn để tiếp tục yêu cầu dở) 33% → 100% → 100%, missing (phải hỏi lại mã) 20% → 100% → 100%, follow_product 14% → 100% → 86%, in_state 83% → 100% → 100%, state-aware nhiều lượt 70% → 90% → 90%.

Nhận xét:
- Base model gần như không theo được schema (87.6% output không phải JSON, router rơi về luật) nên còn tệ hơn rule router.
- SFT vượt router API trên **tập test** (100% vs 90–92%) vì học đúng quy ước nhãn của hệ thống. Lỗi API trên test chủ yếu ở mẫu "đơn X đổi địa chỉ nhận được không?" (nhãn order + policy, Opus chỉ route order), một quy ước nhãn có thể tranh luận. Trên **eval set viết tay**, Opus đạt 100% còn SFT 97.3% (1 lỗi: S13.2 "Mã đơn DH1009, đổi giúp mình cái mới nhé" khi đang pending đổi trả, SFT route sang order thay vì policy), cho thấy SFT vẫn có khoảng cách tổng quát hoá ngoài phân phối template.
- Chi phí router: SFT tự host rẻ hơn Opus@medium ~22 lần / quyết định. Latency SFT 2.2 s cao hơn mong đợi do sinh JSON dài (~150 token) bằng `generate` không batch, chạy chung GPU với các eval khác; dùng vLLM / rút gọn schema sẽ giảm mạnh.

## 5. End-to-end: task success, Tool F1, latency, cost

Agents của mọi cấu hình `api-*` dùng cùng model / effort với router. Các cấu hình `opt-*` dùng chung bộ agent rẻ (Product, Order = Haiku 5.5 @ low; Policy = Sonnet 5.5 @ low; composer Haiku), chỉ khác router, để cô lập ảnh hưởng của router.

| Config | Router | Agents | Task success (Judge) | Tool F1 micro (P / R) | Tool F1 macro | Routing acc | Latency mean / p95 (s) | Router (s) | $ / kịch bản | vs baseline | $ / PASS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **api-opus-medium (baseline)** | Opus @ medium | Opus @ medium | 81.2% | 0.673 (0.52 / 0.95) | 0.707 | 100% | 12.0 / 26.8 | 2.72 | 0.03603 | 0% | 0.04434 |
| api-opus-low | Opus @ low | Opus @ low | 75.0% | 0.701 (0.60 / 0.85) | 0.672 | 94.6% | 9.1 / 20.1 | 2.36 | 0.02696 | −25% | 0.03595 |
| api-opus-high | Opus @ high | Opus @ high | 84.4% | 0.655 (0.49 / 0.97) | 0.719 | 100% | 11.7 / 27.6 | 2.79 | 0.03710 | +3% | 0.04397 |
| api-sonnet-medium | Sonnet @ medium | Sonnet @ medium | **87.5%** | 0.720 (0.60 / 0.90) | 0.729 | 100% | 6.9 / 13.6 | 1.80 | 0.01330 | −63% | 0.01520 |
| api-haiku-low | Haiku @ low | Haiku @ low | **87.5%** | 0.724 (0.58 / 0.95) | 0.755 | 91.9% | 6.1 / 11.6 | 1.33 | **0.00086** | **−98%** | **0.00098** |
| opt-api-mixed | Haiku @ low | mixed | 84.4% | **0.760** (0.63 / 0.95) | **0.788** | 94.6% | 6.2 / 12.7 | 1.39 | 0.00578 | −84% | 0.00685 |
| opt-small-mixed | Small base | mixed | 34.4% | 0.480 (0.51 / 0.45) | 0.373 | 46.0% | 3.5 / 9.6 | 0.66 | 0.00188 | −95% | 0.00548 |
| opt-sft-mixed | **SFT** | mixed | 81.2% | 0.738 (0.60 / 0.95) | 0.767 | **100%** | 6.4 / 11.9 | 2.22 | 0.00558 | −85% | 0.00687 |
| opt-dpo-mixed | DPO | mixed | 71.9% | 0.737 (0.64 / 0.88) | 0.747 | 97.3% | 6.1 / 11.1 | 2.41 | 0.00507 | −86% | 0.00706 |
| small-dpo-opus | DPO | Opus @ medium | 78.1% | 0.700 (0.58 / 0.88) | 0.702 | 97.3% | 9.3 / 17.8 | 2.40 | 0.02372 | −34% | 0.03036 |

Tỉ lệ PASS theo nhóm kịch bản (baseline → Sonnet medium → SFT mixed): policy 58% → 75% → 67%; order 75% → 100% → 75%; product 100% → 83% → 100%; multi 100% → 100% → 67%; state-aware 100% → 100% → 100%; chitchat / out-of-scope 100% ở mọi cấu hình.

### 5.1 Đổi model / reasoning level (yêu cầu so sánh của đề)

| Thay đổi so với baseline Opus @ medium | Δ task success | Δ Tool F1 micro | Δ latency mean | Δ cost / kịch bản |
|---|---|---|---|---|
| Opus @ low | −6.2 điểm | +0.028 | −24% | −25% |
| Opus @ high | +3.1 điểm | −0.018 | −3% | +3% |
| Sonnet 5.5 @ medium | +6.2 điểm | +0.047 | −43% | −63% |
| Haiku 5.5 @ low | +6.2 điểm | +0.051 | −49% | −98% |
| Haiku router + mixed agents | +3.1 điểm | +0.087 | −49% | −84% |
| SFT router + mixed agents | 0 | +0.065 | −47% | −85% |
| DPO router + mixed agents | −9.3 điểm | +0.064 | −49% | −86% |
| DPO router, agents giữ Opus @ medium | −3.1 điểm | +0.027 | −23% | −34% |

- Hạ effort trên Opus (low) tiết kiệm 25% nhưng mất 2 kịch bản (một phần do router bỏ sót agent: routing 94.6%); tăng lên high hầu như không thêm gì vì chi phí bị chi phối bởi số vòng tool chứ không phải độ dài suy luận.
- Model nhỏ hơn **không** kém hơn ở bài này: tác vụ là tra cứu + áp luật rõ ràng từ tool, nên Haiku / Sonnet đủ năng lực, lại ít "gọi thêm tool cho chắc" (precision cao hơn) và nhanh hơn ~2 lần.
- Chênh lệch ±1–2 kịch bản (3–6 điểm) trên 32 kịch bản **chưa có ý nghĩa thống kê**; kết luận vững là: chi phí giảm 60–98% mà không giảm chất lượng đo được.

### 5.2 Tối ưu chi phí và thời gian phản hồi

Đường Pareto (success vs cost / kịch bản): `api-haiku-low` (87.5%, 0.00086 $) chiếm ưu thế mọi cấu hình còn lại trên tập này; `api-sonnet-medium` là lựa chọn an toàn hơn cho Policy (75% vs 58% của baseline).
Các kỹ thuật đã cài và hiệu quả đo được:
1. **Router tự host (SFT/DPO)**: bỏ một lời gọi API mỗi lượt: số lời gọi LLM / lượt 3.38 → 2.30 (`opt-api-mixed` → `opt-sft-mixed`), routing 94.6% → 100%.
2. **Hỏi lại ngay khi thiếu mã đơn / chitchat trả lời tại router**: không gọi agent.
3. **Bỏ composer khi chỉ 1 agent**.
4. **Prompt caching** system + tools: 170–180 nghìn token đọc từ cache mỗi cấu hình API (Opus cache read 0.20 $ thay vì 4 $ / 1M).
5. **Chọn model theo agent** (`opt-*`): Policy cần suy luận điều kiện dùng Sonnet, tra cứu dùng Haiku.
6. **Chặn ngân sách** toàn hệ thống (`API_BUDGET_USD`, sổ chi tiêu chung nhiều process).

## 6. Phân tích lỗi

Nguồn: `results/eval/<config>/details.jsonl` (lý do Judge + tool calls).
1. **Policy agent là nút thắt** (58–75% PASS ở mọi cấu hình): các ca biên đổi trả / bảo hành cần tính đúng % hoàn tiền, số ngày, tình trạng hộp; lỗi thường là nêu sai % / mức phí hoặc không tạo phiếu khi khách đã yêu cầu. Router đúng 100% ở baseline nên lỗi nằm ở agent / RAG chứ không ở điều phối.
2. **Tool precision thấp (0.49–0.63)** ở mọi cấu hình: agent gọi thêm `search_policy` / `get_product_details` để "kiểm tra lại". Không làm FAIL nhưng tốn latency và tiền; Opus high gọi thừa nhiều nhất (P = 0.49).
3. **Small base**: 87.6% output không phải JSON → rơi về rule router → bỏ sót agent ở câu multi, không hiểu pending → 34.4% PASS. Đây là lý do cần SFT.
4. **SFT**: lỗi duy nhất trên eval set là S13.2 (pending đổi trả + "đổi giúp mình cái mới", route nhầm order); end-to-end vẫn đạt routing 100% vì lượt thật có lịch sử đầy đủ hơn.
5. **API router**: Haiku hay gọi thừa `order` cho câu bảo hành có mã đơn (S18, S19), Opus bỏ `policy` ở câu "đổi địa chỉ nhận".
6. **DPO kém hơn SFT**: lỗi tập trung ở câu cần 2 agent (S22.1, T18, T34 chỉ route 1 agent hoặc không agent nào) và 9.3% output không còn là JSON hợp lệ (router rơi về luật, ví dụ S24.2 "Máy đó có NFC không?" → không agent). Nguyên nhân hợp lý từ dữ liệu và log:
   - Tập cặp **lệch**: 451 cặp `extra_agent` (rejected = gọi 2 agent) và 1 251 `wrong_agent`, trong khi chỉ 257 cặp `drop_agent` → DPO học "bớt agent" mạnh hơn "đủ agent", làm hỏng câu multi.
   - **Over-optimization**: rewards/margins tăng tới 6.6, eval loss 0.0036, `logps/chosen` cũng giảm (−2.35 → −2.92): xác suất của câu trả lời đúng bị kéo xuống theo, hiện tượng đã biết của DPO, gây hỏng định dạng.
   - Chỉ 11 cặp on-policy: model SFT gần như không sai trên phân phối train, nên DPO chủ yếu tối ưu trên lỗi giả lập chứ không phải lỗi thật.
   - Hướng sửa (chưa chạy do giới hạn thời gian): cân bằng số cặp theo loại lỗi (bỏ phần lớn `extra_agent` cho câu multi), tăng β (0.3–0.5) hoặc thêm SFT loss trên chosen (`rpo_alpha` trong TRL), giảm số step / lr, lấy on-policy trên eval set viết tay hoặc paraphrase để có lỗi thật.

## 7. Hạn chế

- Eval set nhỏ (32 kịch bản / 37 lượt): chênh lệch vài điểm % không đủ ý nghĩa thống kê; nên tăng lên ≥ 100 kịch bản và chạy lặp nhiều seed.
- Tập test routing sinh cùng phân phối với tập train (template + HF), nên 99–100% của SFT trên test là cận trên; số liệu trên eval set viết tay (97.3%) đáng tin hơn.
- Judge là một LLM (Sonnet 5.5 @ low): có thể sai ở ca biên; lý do từng kịch bản đã lưu để kiểm tra tay.
- Latency Small LLM đo khi nhiều eval chạy song song trên cùng máy; giá GPU 0.35 $/giờ là giả định.
- Do giới hạn thời gian (toàn bộ train + eval trong ~35 phút), Small LLM dùng bản 1.5B (không thử 3B), on-policy DPO chỉ lấy 120 prompt và không lặp lại DPO sau khi thấy kết quả kém.

## 8. Chi phí thực tế của lần chạy

Theo `results/spend.json` (sổ chi tiêu ghi sau mỗi lời gọi, chặn cứng ở 19.5 $):

| Model | Chi phí | Dùng cho |
|---|---|---|
| Claude Opus 5.5 | 4.41 $ | router + agents các cấu hình Opus, routing-only Opus |
| Claude Sonnet 5.5 | 3.27 $ | LLM Judge (≈0.19–0.26 $ / cấu hình), agents Sonnet, Policy agent trong `opt-*` |
| Claude Haiku 5.5 | 0.08 $ | router + agents Haiku |
| **Tổng** | **7.77 $** (1 563 lời gọi) | 10 cấu hình end-to-end × 32 kịch bản + 2 router API routing-only × 97 quyết định |

GPU: 4× RTX 5090 trong ~35 phút (SFT 4.2 phút, DPO 10.1 phút, phần còn lại là eval chạy song song).

## 9. Hướng dẫn cài đặt & chạy

### Tái lập toàn bộ thí nghiệm (máy GPU)

```bash
bash training/remote_run.sh setup      # GPU + thư viện + test offline
bash training/remote_run.sh train      # data (template + HF) -> SFT -> DPO pairs -> DPO
bash training/remote_run.sh route-small && bash training/remote_run.sh route-api
bash training/remote_run.sh e2e api-opus-medium api-opus-low api-opus-high api-sonnet-medium api-haiku-low \
     opt-api-mixed opt-small-mixed opt-sft-mixed opt-dpo-mixed small-dpo-opus
bash training/remote_run.sh report     # results/comparison.md + results/dashboard.html
```
Không có GPU riêng: `training/train_router_colab.ipynb` (notebook tự chứa, chỉ upload 1 file lên Colab).

### Cài đặt

```bat
cd "C:\phase3\assignment 39"
copy .env.example .env          &:: điền ANTHROPIC_API_KEY
pip install anthropic mcp numpy                       &:: tối thiểu để chạy agent
pip install sentence-transformers                     &:: RAG dense + reranker (không có thì tự lùi về BM25)
```

Máy yếu RAM: đặt `RAG_LIGHT=1` trong `.env` (chỉ BM25) và train / chạy Small LLM trên Colab.

### Chạy

```bat
python -m tests.test_offline                     &:: không tốn tiền, không cần GPU
python run_chat.py                               &:: chat: router API (Opus 5.5 @ medium)
python run_chat.py --router rule --rag-light     &:: rẻ nhất để thử luồng
python run_chat.py --tools mcp --a2a http        &:: tool qua MCP server, 3 agent là A2A HTTP service (port 9101–9103)
python -m support.mcp_server --http 8765         &:: chạy riêng MCP server (Streamable HTTP) để gắn vào client khác
```

Hội thoại mẫu kiểm tra state-aware:
```
Bạn: Màn hình mình mới mua về bị điểm chết, muốn đổi cái khác
Trợ lý: Bạn vui lòng cho mình mã đơn hàng ...              (missing_info → pending_intent=return_request)
Bạn: DH1009
Trợ lý: ... lỗi NSX trong 30 ngày, được đổi mới ...       (router tiếp tục return_request, không hiểu là tra đơn)
```

Claude Desktop: sửa đường dẫn trong `claude_desktop_config.example.json` rồi chép vào `%APPDATA%\Claude\claude_desktop_config.json`.

### Train Small LLM (Colab, GPU T4/L4)

**Cách nhanh:** chỉ upload 1 file `training/train_router_colab.ipynb` lên Colab, chọn T4 GPU, *Run all*.
Notebook nhúng sẵn 15 file cần cho training (config, state/router/tracing/llm, gen_data, hf_import, gen_dpo_pairs,
sft_train, dpo_train, routing_eval, metrics, test_training, products.json, eval_set.json); dữ liệu lấy từ HF.
Sửa code ở máy xong thì build lại notebook: `python training/build_colab_notebook.py`.

Chạy tay (máy có GPU):
```bash
pip install -r requirements.txt
python -m training.gen_data --n 1500                        # + --paraphrase 400 nếu muốn đa dạng câu (tốn ~vài cent Haiku)
python -m training.hf_import --mode append                  # + hội thoại từ HF Tson29/techviet-cs-orchestrator-data (convert schema)
python -m eval.routing_eval --routers small --test-limit 300  # đo base trước khi train
python -m training.sft_train --qlora --epochs 2               # ~20–30 phút trên T4 với 1.5B
python -m training.gen_dpo_pairs --sft-adapter training/checkpoints/sft --onpolicy-n 800
python -m training.dpo_train --beta 0.1 --epochs 1
python -m eval.routing_eval --routers small sft dpo --test-limit 300
```
Đổi base: `--base meta-llama/Llama-3.2-1B-Instruct` hoặc `Qwen/Qwen2.5-3B-Instruct` (3B nên bật `--qlora`), và đặt `SMALL_BASE_MODEL` tương ứng khi inference.
Adapter LoRA (`training/checkpoints/{sft,dpo}`, ~74 MB mỗi bản) không đưa vào repo; train lại theo các lệnh trên (SFT ~4 phút, DPO ~10 phút trên RTX 5090) nếu muốn chạy `run_chat.py --router sft`.

**Vì sao cần DPO sau SFT?** SFT học "bắt chước nhãn", nhưng lỗi nặng của router là các lỗi *quyết định*: hỏi lại mã đơn khi state đã có
(`state_unaware`), route thẳng khi thiếu mã (`no_ask`, dẫn tới agent đoán bừa), quên `pending_intent`, bỏ sót agent trong câu multi, gọi thừa agent (tốn tiền).
DPO đặt cặp đúng/sai cạnh nhau, phạt trực tiếp các lỗi đó. Negatives on-policy (lỗi model SFT thật sự mắc) hiệu quả nhất.

### Chạy đánh giá

#### Chạy

```bat
python -m eval.routing_eval --routers rule api                          &:: routing-only, rẻ
python -m eval.routing_eval --routers api --api-model claude-haiku-5-5 --api-effort low
python -m eval.run_eval --configs api-opus-medium                       &:: baseline end-to-end
python -m eval.run_eval --configs api-opus-low api-opus-high api-sonnet-medium api-haiku-medium opt-api-mixed
python -m eval.run_eval --configs small-sft-opus small-dpo-opus opt-dpo-mixed   &:: máy có GPU + adapter
python -m eval.report                                                   &:: results/comparison.md + results/dashboard.html
```
Thử nhỏ trước: `--ids S07 S13 S16 --no-judge`. Một lần chạy baseline đầy đủ ≈ 38 lượt × (router + 1–2 agent × 2–4 lượt tool) + 32 lần judge.

#### Các cấu hình (`eval/run_eval.py: CONFIGS`)

| Config | Router | Agents | Mục đích |
|---|---|---|---|
| `api-opus-medium` | Opus 5.5 @ medium | Opus 5.5 @ medium | **Baseline** (≈ gpt-oss-120b medium) |
| `api-opus-low` / `-high` | Opus 5.5 @ low / high | như router | ảnh hưởng của reasoning level |
| `api-sonnet-medium` / `-low`, `api-haiku-medium` / `-low` | Sonnet / Haiku 5.5 | như router | đổi model |
| `rule-opus`, `small-base-opus`, `small-sft-opus`, `small-dpo-opus` | rule / Small / SFT / DPO | Opus 5.5 @ medium | cô lập ảnh hưởng của router |
| `opt-dpo-mixed` | DPO (tự host) | Haiku @ low; Policy = Sonnet @ low; composer Haiku | tối ưu chi phí |
| `opt-small-mixed` / `opt-sft-mixed` | Small base / SFT | như `opt-dpo-mixed` | so router base / SFT / DPO với cùng bộ agent |
| `opt-api-mixed` | Haiku @ low | như trên | tối ưu chi phí, không cần GPU |

### Serving: FastAPI + Streamlit + Docker

Local (2 terminal):
```bat
pip install fastapi uvicorn streamlit requests
python -m api.main                 &:: http://localhost:8000/docs
streamlit run frontend/app.py      &:: http://localhost:8501  (API_URL mặc định http://localhost:8000)
```

| Endpoint | |
|---|---|
| `GET /health` | cấu hình đang chạy, số session, RAG đã tải chưa (không tải model) |
| `GET /ready` | tải RAG index nếu chưa có |
| `GET /agents` | A2A AgentCard |
| `POST /chat` `{message, session_id?}` | 1 lượt: reply, intent, agents, plan, tool_calls, agent_results, trace_id, metrics (latency, tokens, cost) |
| `GET /sessions/{id}`, `POST /sessions/{id}/reset`, `DELETE /sessions/{id}` | xem / xoá ConversationState |
| `POST /rag/search` `{query, top_k, rewrite?}` | debug RAG |
| `POST /admin/reset-data` (header `X-Admin-Token`) | khôi phục dữ liệu đơn hàng; chỉ bật khi đặt `ADMIN_TOKEN` |

- Mỗi `session_id` có `ConversationState` riêng; router / agents / tool backend / tracer dùng chung (shallow copy Orchestrator).
  Request cùng session chạy tuần tự (lock), khác session chạy song song. Hết hạn sau `SESSION_TTL_S`, tối đa `MAX_SESSIONS` (LRU).
- Session nằm trong RAM: restart là mất. Dữ liệu đơn hàng (huỷ đơn, phiếu) là chung cho mọi session như bản CLI.
- Router Small LLM được bọc lock (generate không an toàn khi gọi song song).
- Lỗi: refusal → 422, lỗi Anthropic API → 502, còn lại 500; body luôn có `request_id` (cũng trả ở header `X-Request-ID`).

Docker:
```bat
copy .env.example .env                                &:: điền ANTHROPIC_API_KEY
docker compose up --build                             &:: api :8000 + ui :8501, BM25 only (image nhỏ)
set "WITH_RAG_MODELS=1"
docker compose up --build                             &:: + torch CPU, dense + reranker (model tải vào volume hf-cache)
```
`Dockerfile` có 2 target (`api`, `ui`), chạy user không phải root, có `HEALTHCHECK`; `ui` chỉ khởi động khi `api` healthy.
Trace ghi ra `./results/traces/api-*.jsonl` trên máy host (bind mount) → `python -m eval.report` dùng được luôn.
Router SFT/DPO cần GPU nên không đóng gói vào image; trong Docker dùng `API_ROUTER=api` hoặc `rule`.

### Query rewriting (RAG)

`search_knowledge_base` / `search_policy` giờ chạy: **rewrite → multi-query hybrid search → RRF → rerank**.

| `QUERY_REWRITE` | Làm gì | Chi phí |
|---|---|---|
| `off` | giữ nguyên câu | 0 |
| `rules` (mặc định) | chuẩn hoá viết tắt / teencode (`bh`→bảo hành, `tws`→true wireless, `bóc seal`→đã mở hộp, `đc k`→được không), tiền (`5tr`→5 triệu), tách câu nhiều ý theo `? ; !` | 0, deterministic → eval tái lập được |
| `llm` | rules + Claude (`REWRITE_MODEL`, mặc định Haiku 5.5 @ low, structured output) viết 1–3 truy vấn đúng thuật ngữ tài liệu (danh sách heading được đưa vào prompt) + từ khoá BM25; cache LRU; lỗi → lùi về rules | +1 lời gọi Haiku / lần search (lần lặp lại đọc cache) |

- Mỗi truy vấn viết lại chạy cả BM25 lẫn dense, mọi danh sách xếp hạng vào chung RRF; cross-encoder chấm theo câu đầy đủ đã chuẩn hoá.
- Mã đơn (`dh 1005`), mã SP, `4K`, tên `Nova Tab S` được giữ nguyên (có test).
- Trace có span `rag.rewrite` (queries, keywords, source = rules / llm / cache / fallback) và `rag.search` (chunk lọt top).
  Tool trả thêm `searched_as` khi truy vấn bị đổi, để agent biết đã tìm bằng gì.
- Lưu ý: agent Claude vốn đã tự viết lại `query` khi gọi tool, nên `llm` chủ yếu có lợi khi agent dùng model nhỏ / effort thấp
  (`opt-*`). So sánh bằng: `QUERY_REWRITE=off|rules|llm python -m eval.run_eval --configs opt-api-mixed`.
- Debug trực quan: tab **RAG search** trong Streamlit, hoặc `POST /rag/search {"query": "...", "rewrite": "llm"}`.
- CLI: `python run_chat.py --rewrite llm`.

### Structured logging

`support/logging_setup.py`, chỉ dùng stdlib. `LOG_FORMAT=json` (mặc định trong Docker) → mỗi dòng 1 JSON:
```json
{"ts": "2026-11-02T08:15:03.120+00:00", "level": "INFO", "logger": "api", "msg": "turn done",
 "trace_id": "9f2c41d0a7b3e611", "request_id": "a1b2c3d4e5f6", "session_id": "...", "intent": "return_request",
 "agents": ["policy"], "tools": ["check_return_eligibility"], "latency_ms": 5321.4, "llm_calls": 3, "cost_usd": 0.0187}
```
- `trace_id` / `span_id` tự lấy từ span đang mở → lọc log theo `trace_id` rồi mở đúng trace trong JSONL / dashboard.
- Các sự kiện được log: `http request` (method, path, status, latency), `routed` (intent, agents, missing_info, pending),
  `turn done` (tokens, cost), `agent task failed`, `span failed`, `rag index ready`, `dense search disabled`,
  `query rewrite LLM lỗi`, `session created / evicted`. `LOG_LEVEL=DEBUG` log thêm mọi span khi đóng.
- Log luôn ra **stderr** (MCP stdio server dùng stdout cho giao thức). `run_chat.py` và MCP server mặc định mức WARNING;
  `LOG_FILE=...` ghi thêm file JSONL.

### LLMOps / Observability

- Trace JSONL: `results/traces/<run>.jsonl`. Mỗi span có `trace_id`, `parent_id`, `kind` (turn / router / a2a / agent / llm / tool / compose), latency,
  attrs (model, effort, input/output/cache tokens, cost, args, result, decision, A2A task state).
- `python -m eval.report` → `results/dashboard.html`: success, F1, cost, latency theo cấu hình + waterfall các trace mới nhất.
- Refusal: `support/llm.py` bật server-side fallback (`fallbacks="default"`, beta `server-side-fallback-2026-07-01`) cho Opus / Sonnet và vẫn kiểm tra `stop_reason == "refusal"`. Tắt bằng `USE_FALLBACKS=0`.
- Lỗi ở một agent → A2A Task `failed`, Orchestrator vẫn trả lời (không sập cả lượt); span đánh `status=error`.
- Ở chế độ `--a2a http`, span của agent nằm trong thread của HTTP server nên thành trace riêng (không lồng dưới `turn`); khi eval để so chi phí, dùng A2A in-process (mặc định).

## Kết quả thô

`results/eval/<config>/{summary.json,details.jsonl}` (lý do Judge từng kịch bản), `results/routing/*.json`, `results/traces/*.jsonl` (span tree), `results/logs/*.log` (log train / eval), `results/comparison.md`, `results/dashboard.html`, `results/spend.json`.
