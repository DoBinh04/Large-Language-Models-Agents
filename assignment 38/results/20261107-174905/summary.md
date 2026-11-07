# RAG evaluation: golden set

- Strategy `recursive`, top-k = 5, generator `claude-haiku-5-5`, judge `claude-haiku-5-5`
- Judge usage: 40 calls, 112,704 in / 13,667 out tokens

| Metric | Giá trị | Ghi chú |
|---|---|---|
| **Faithfulness** (LLM judge) | 0.983 | 1 claim không có trong context, trên tổng 59 claim |
| **Answer relevance** (cosine) | 0.768 | mốc cos(question, đáp án chuẩn) = 0.608; normalized = 1.274 |
| **Context precision@5** (ground truth) | 0.200 | AP = 0.825 (thưởng xếp chunk đúng lên đầu) |
| Context precision@5 (LLM judge) | 0.200 | AP = 0.825; đồng thuận nhãn với ground truth = 1.000 |
| Abstention | 1/1 câu không-trả-lời-được đã từ chối đúng | 1/19 câu trả-lời-được bị từ chối nhầm |

## Theo loại câu hỏi

| Loại | n | Faithfulness | Relevance | Precision (GT) | Precision (LLM) |
|---|---|---|---|---|---|
| single-fact + distractor | 1 | 1.000 | 0.920 | 0.200 | 0.200 |
| multi-evidence (same section) | 1 | 0.667 | 0.806 | 0.200 | 0.200 |
| paraphrase (low lexical overlap) | 1 | 1.000 | 0.654 | 0.200 | 0.200 |
| long evidence (numbered list) | 1 | 1.000 | 0.806 | 0.200 | 0.200 |
| single-fact + many numeric distractors | 1 | 1.000 | 0.787 | 0.200 | 0.200 |
| multi-hop (two distant sections) | 1 | 1.000 | 0.757 | 0.400 | 0.400 |
| single-fact (conditional rule) | 1 | 1.000 | 0.803 | 0.200 | 0.200 |
| multi-evidence (adjacent sentences) | 2 | 1.000 | 0.796 | 0.200 | 0.200 |
| single-fact (paraphrase) | 1 | 1.000 | 0.777 | 0.200 | 0.200 |
| single-fact (reasoning: why) | 2 | 1.000 | 0.858 | 0.200 | 0.200 |
| single-fact (date) | 1 | 1.000 | 0.801 | 0.200 | 0.200 |
| multi-evidence (rule + exception) | 1 | 1.000 | 0.876 | 0.200 | 0.200 |
| multi-hop (reasoning within a section) | 1 | 1.000 | 0.727 | 0.200 | 0.200 |
| procedure (conditional) | 1 | 1.000 | 0.466 | 0.000 | 0.000 |
| paraphrase (mechanism) | 1 | 1.000 | 0.809 | 0.200 | 0.200 |
| numeric reasoning | 1 | 1.000 | 0.769 | 0.200 | 0.200 |
| single-fact (timeline) | 1 | 1.000 | 0.848 | 0.200 | 0.200 |
| unanswerable (should abstain) | 1 | 1.000 | 0.439 | — | — |

## Từng câu

| ID | Faith | Claims (unsup.) | Abstain | Relevance | Rel/ref | P_gt | AP_gt | P_llm | Agree |
|---|---|---|---|---|---|---|---|---|---|
| G01 | 1.00 | 4 (0) |  | 0.920 | 1.42 | 0.20 | 1.00 | 0.20 | 1.00 |
| G02 | 0.67 | 3 (1) |  | 0.806 | 1.22 | 0.20 | 1.00 | 0.20 | 1.00 |
| G03 | 1.00 | 4 (0) |  | 0.654 | 1.15 | 0.20 | 1.00 | 0.20 | 1.00 |
| G04 | 1.00 | 6 (0) |  | 0.806 | 1.16 | 0.20 | 1.00 | 0.20 | 1.00 |
| G05 | 1.00 | 1 (0) |  | 0.787 | 1.20 | 0.20 | 1.00 | 0.20 | 1.00 |
| G06 | 1.00 | 3 (0) |  | 0.757 | 1.24 | 0.40 | 0.58 | 0.40 | 1.00 |
| G07 | 1.00 | 2 (0) |  | 0.803 | 1.28 | 0.20 | 1.00 | 0.20 | 1.00 |
| G08 | 1.00 | 3 (0) |  | 0.824 | 1.21 | 0.20 | 1.00 | 0.20 | 1.00 |
| G09 | 1.00 | 2 (0) |  | 0.777 | 1.32 | 0.20 | 1.00 | 0.20 | 1.00 |
| G10 | 1.00 | 4 (0) |  | 0.887 | 1.75 | 0.20 | 0.50 | 0.20 | 1.00 |
| G11 | 1.00 | 2 (0) |  | 0.801 | 1.13 | 0.20 | 1.00 | 0.20 | 1.00 |
| G12 | 1.00 | 3 (0) |  | 0.876 | 1.38 | 0.20 | 1.00 | 0.20 | 1.00 |
| G13 | 1.00 | 4 (0) |  | 0.727 | 1.09 | 0.20 | 1.00 | 0.20 | 1.00 |
| G14 | 1.00 | 5 (0) |  | 0.830 | 1.40 | 0.20 | 0.25 | 0.20 | 1.00 |
| G15 | 1.00 | 0 (0) | ✓ | 0.466 | 0.72 | 0.00 | 0.00 | 0.00 | 1.00 |
| G16 | 1.00 | 5 (0) |  | 0.809 | 1.63 | 0.20 | 1.00 | 0.20 | 1.00 |
| G17 | 1.00 | 3 (0) |  | 0.768 | 1.49 | 0.20 | 0.33 | 0.20 | 1.00 |
| G18 | 1.00 | 3 (0) |  | 0.769 | 1.35 | 0.20 | 1.00 | 0.20 | 1.00 |
| G19 | 1.00 | 2 (0) |  | 0.848 | 1.44 | 0.20 | 1.00 | 0.20 | 1.00 |
| G20 | 1.00 | 0 (0) | ✓ | 0.439 | 0.89 | — | — | — | — |

## Cần xem lại (mở details.jsonl)

- G02: faithfulness 0.67
- G15: precision (GT) = 0, không chunk nào trong top-5 chứa evidence
- G15: câu trả-lời-được nhưng model từ chối
