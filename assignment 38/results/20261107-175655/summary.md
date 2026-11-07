# RAG evaluation: golden set

- Strategy `recursive`, top-k = 3, generator `claude-haiku-5-5`, judge `claude-haiku-5-5`
- Judge usage: 40 calls, 79,152 in / 10,074 out tokens

| Metric | Giá trị | Ghi chú |
|---|---|---|
| **Faithfulness** (LLM judge) | 1.000 | 0 claim không có trong context, trên tổng 55 claim |
| **Answer relevance** (cosine) | 0.770 | mốc cos(question, đáp án chuẩn) = 0.608; normalized = 1.277 |
| **Context precision@3** (ground truth) | 0.316 | AP = 0.811 (thưởng xếp chunk đúng lên đầu) |
| Context precision@3 (LLM judge) | 0.333 | AP = 0.838; đồng thuận nhãn với ground truth = 0.982 |
| Abstention | 1/1 câu không-trả-lời-được đã từ chối đúng | 1/19 câu trả-lời-được bị từ chối nhầm |

## Theo loại câu hỏi

| Loại | n | Faithfulness | Relevance | Precision (GT) | Precision (LLM) |
|---|---|---|---|---|---|
| single-fact + distractor | 1 | 1.000 | 0.920 | 0.333 | 0.333 |
| multi-evidence (same section) | 1 | 1.000 | 0.826 | 0.333 | 0.333 |
| paraphrase (low lexical overlap) | 1 | 1.000 | 0.631 | 0.333 | 0.333 |
| long evidence (numbered list) | 1 | 1.000 | 0.805 | 0.333 | 0.333 |
| single-fact + many numeric distractors | 1 | 1.000 | 0.787 | 0.333 | 0.333 |
| multi-hop (two distant sections) | 1 | 1.000 | 0.827 | 0.667 | 0.667 |
| single-fact (conditional rule) | 1 | 1.000 | 0.773 | 0.333 | 0.333 |
| multi-evidence (adjacent sentences) | 2 | 1.000 | 0.824 | 0.333 | 0.333 |
| single-fact (paraphrase) | 1 | 1.000 | 0.777 | 0.333 | 0.333 |
| single-fact (reasoning: why) | 2 | 1.000 | 0.896 | 0.167 | 0.333 |
| single-fact (date) | 1 | 1.000 | 0.776 | 0.333 | 0.333 |
| multi-evidence (rule + exception) | 1 | 1.000 | 0.771 | 0.333 | 0.333 |
| multi-hop (reasoning within a section) | 1 | 1.000 | 0.731 | 0.333 | 0.333 |
| procedure (conditional) | 1 | 1.000 | 0.466 | 0.000 | 0.000 |
| paraphrase (mechanism) | 1 | 1.000 | 0.800 | 0.333 | 0.333 |
| numeric reasoning | 1 | 1.000 | 0.778 | 0.333 | 0.333 |
| single-fact (timeline) | 1 | 1.000 | 0.848 | 0.333 | 0.333 |
| unanswerable (should abstain) | 1 | 1.000 | 0.439 | — | — |

## Từng câu

| ID | Faith | Claims (unsup.) | Abstain | Relevance | Rel/ref | P_gt | AP_gt | P_llm | Agree |
|---|---|---|---|---|---|---|---|---|---|
| G01 | 1.00 | 4 (0) |  | 0.920 | 1.42 | 0.33 | 1.00 | 0.33 | 1.00 |
| G02 | 1.00 | 3 (0) |  | 0.826 | 1.26 | 0.33 | 1.00 | 0.33 | 1.00 |
| G03 | 1.00 | 4 (0) |  | 0.631 | 1.11 | 0.33 | 1.00 | 0.33 | 1.00 |
| G04 | 1.00 | 5 (0) |  | 0.805 | 1.15 | 0.33 | 1.00 | 0.33 | 1.00 |
| G05 | 1.00 | 2 (0) |  | 0.787 | 1.20 | 0.33 | 1.00 | 0.33 | 1.00 |
| G06 | 1.00 | 3 (0) |  | 0.827 | 1.36 | 0.67 | 0.58 | 0.67 | 1.00 |
| G07 | 1.00 | 2 (0) |  | 0.773 | 1.23 | 0.33 | 1.00 | 0.33 | 1.00 |
| G08 | 1.00 | 3 (0) |  | 0.903 | 1.32 | 0.33 | 1.00 | 0.33 | 1.00 |
| G09 | 1.00 | 2 (0) |  | 0.777 | 1.32 | 0.33 | 1.00 | 0.33 | 1.00 |
| G10 | 1.00 | 4 (0) |  | 0.903 | 1.78 | 0.33 | 0.50 | 0.33 | 1.00 |
| G11 | 1.00 | 2 (0) |  | 0.776 | 1.09 | 0.33 | 1.00 | 0.33 | 1.00 |
| G12 | 1.00 | 3 (0) |  | 0.771 | 1.22 | 0.33 | 1.00 | 0.33 | 1.00 |
| G13 | 1.00 | 3 (0) |  | 0.731 | 1.10 | 0.33 | 1.00 | 0.33 | 1.00 |
| G14 | 1.00 | 3 (0) |  | 0.888 | 1.50 | 0.00 | 0.00 | 0.33 | 0.67 |
| G15 | 1.00 | 0 (0) | ✓ | 0.466 | 0.72 | 0.00 | 0.00 | 0.00 | 1.00 |
| G16 | 1.00 | 4 (0) |  | 0.800 | 1.61 | 0.33 | 1.00 | 0.33 | 1.00 |
| G17 | 1.00 | 3 (0) |  | 0.745 | 1.45 | 0.33 | 0.33 | 0.33 | 1.00 |
| G18 | 1.00 | 3 (0) |  | 0.778 | 1.37 | 0.33 | 1.00 | 0.33 | 1.00 |
| G19 | 1.00 | 2 (0) |  | 0.848 | 1.44 | 0.33 | 1.00 | 0.33 | 1.00 |
| G20 | 1.00 | 0 (0) | ✓ | 0.439 | 0.89 | — | — | — | — |

## Cần xem lại (mở details.jsonl)

- G14: precision (GT) = 0, không chunk nào trong top-3 chứa evidence
- G15: precision (GT) = 0, không chunk nào trong top-3 chứa evidence
- G15: câu trả-lời-được nhưng model từ chối
