# So sánh cấu hình

Baseline: `api-opus-medium` (Claude Opus 5.5, effort medium — thay cho gpt-oss-120b medium).

## End-to-end (data/eval_set.json)

| Config | Router | Agents | Task success (Judge) | Tool F1 micro (P / R) | Tool F1 macro | Routing acc | Latency mean / p95 (s) | Router (ms) | Cost / kịch bản ($) | Cost / PASS ($) | LLM calls / lượt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| api-haiku-low | api (claude-haiku-5-5@low) | claude-haiku-5-5@low | 87.5% | 0.724 (0.58 / 0.95) | 0.755 | 91.9% | 6.1 (-49%) / 11.6 | 1330 | 0.0009 (-98%) | 0.00098 | 3.54 |
| api-sonnet-medium | api (claude-sonnet-5-5@medium) | claude-sonnet-5-5@medium | 87.5% | 0.720 (0.60 / 0.90) | 0.729 | 100.0% | 6.9 (-43%) / 13.6 | 1796 | 0.0133 (-63%) | 0.0152 | 3.24 |
| api-opus-high | api (claude-opus-5-5@high) | claude-opus-5-5@high | 84.4% | 0.655 (0.49 / 0.97) | 0.719 | 100.0% | 11.7 (-2%) / 27.6 | 2791 | 0.0371 (+3%) | 0.04397 | 3.49 |
| opt-api-mixed | api (claude-haiku-5-5@low) | claude-haiku-5-5@low + {'policy': ['claude-sonnet-5-5', 'low']} | 84.4% | 0.760 (0.63 / 0.95) | 0.788 | 94.6% | 6.2 (-49%) / 12.7 | 1394 | 0.0058 (-84%) | 0.00685 | 3.38 |
| api-opus-medium | api (claude-opus-5-5@medium) | claude-opus-5-5@medium | 81.2% | 0.673 (0.52 / 0.95) | 0.707 | 100.0% | 12.0 (+0%) / 26.8 | 2724 | 0.0360 (+0%) | 0.04434 | 3.49 |
| opt-sft-mixed | sft | claude-haiku-5-5@low + {'policy': ['claude-sonnet-5-5', 'low']} | 81.2% | 0.738 (0.60 / 0.95) | 0.767 | 100.0% | 6.4 (-47%) / 11.9 | 2219 | 0.0056 (-85%) | 0.00687 | 2.3 |
| small-dpo-opus | dpo | claude-opus-5-5@medium | 78.1% | 0.700 (0.58 / 0.88) | 0.702 | 97.3% | 9.3 (-23%) / 17.8 | 2403 | 0.0237 (-34%) | 0.03036 | 2.27 |
| api-opus-low | api (claude-opus-5-5@low) | claude-opus-5-5@low | 75.0% | 0.701 (0.60 / 0.85) | 0.672 | 94.6% | 9.1 (-25%) / 20.1 | 2359 | 0.0270 (-25%) | 0.03595 | 3.14 |
| opt-dpo-mixed | dpo | claude-haiku-5-5@low + {'policy': ['claude-sonnet-5-5', 'low']} | 71.9% | 0.737 (0.64 / 0.88) | 0.747 | 97.3% | 6.1 (-49%) / 11.1 | 2412 | 0.0051 (-86%) | 0.00706 | 2.11 |
| opt-small-mixed | small | claude-haiku-5-5@low + {'policy': ['claude-sonnet-5-5', 'low']} | 34.4% | 0.480 (0.51 / 0.45) | 0.373 | 46.0% | 3.5 (-71%) / 9.6 | 659 | 0.0019 (-95%) | 0.00548 | 1.51 |

### Task success theo nhóm kịch bản

| Config | chitchat | multi | order | out_of_scope | policy | product | state |
|---|---|---|---|---|---|---|---|
| api-haiku-low | 100.0% | 100.0% | 75.0% | 100.0% | 75.0% | 100.0% | 100.0% |
| api-opus-high | 100.0% | 100.0% | 75.0% | 100.0% | 75.0% | 83.3% | 100.0% |
| api-opus-low | 100.0% | 100.0% | 75.0% | 100.0% | 50.0% | 100.0% | 80.0% |
| api-opus-medium | 100.0% | 100.0% | 75.0% | 100.0% | 58.3% | 100.0% | 100.0% |
| api-sonnet-medium | 100.0% | 100.0% | 100.0% | 100.0% | 75.0% | 83.3% | 100.0% |
| opt-api-mixed | 100.0% | 100.0% | 100.0% | 100.0% | 66.7% | 83.3% | 100.0% |
| opt-dpo-mixed | 100.0% | 66.7% | 25.0% | 100.0% | 66.7% | 100.0% | 80.0% |
| opt-sft-mixed | 100.0% | 66.7% | 75.0% | 100.0% | 66.7% | 100.0% | 100.0% |
| opt-small-mixed | 100.0% | 0.0% | 50.0% | 100.0% | 16.7% | 83.3% | 0.0% |
| small-dpo-opus | 100.0% | 66.7% | 50.0% | 100.0% | 75.0% | 100.0% | 80.0% |

## Routing-only (API LLM vs Small LLM vs SFT vs DPO)

| Router | Routing acc | Intent acc | Acc eval set | Acc test | JSON valid | Latency mean / p95 (ms) | $ / 1k quyết định |
|---|---|---|---|---|---|---|---|
| api:claude-haiku-5-5@low | 91.8% | 86.6% | 94.6% | 90.0% | 100.0% | 1410 / 2560 | 0.1289 |
| api:claude-opus-5-5@medium | 94.8% | 92.8% | 100.0% | 91.7% | 100.0% | 2716 / 4275 | 4.7424 |
| rule | 57.3% | 54.8% | 56.8% | 57.5% | 100.0% | 0 / 0 | 0.0000 |
| small-dpo | 93.6% | 93.6% | 94.6% | 93.3% | 87.3% | 2569 / 4843 | 0.2496 |
| small-sft | 99.4% | 99.4% | 97.3% | 100.0% | 100.0% | 2217 / 2757 | 0.2153 |
| small-small | 45.2% | 42.7% | 51.3% | 43.3% | 14.0% | 641 / 1045 | 0.0622 |

### Routing acc theo nhóm (state-aware: pending / in_state / follow_product / switch)

| Router | chitchat | follow_product | in_state | missing | multi | order | out_of_scope | pending | policy | product | single | state | switch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| api:claude-haiku-5-5@low | 100.0% | 100.0% | 100.0% | 100.0% | 71.4% | 100.0% | 100.0% | 100.0% | 83.3% | 100.0% | 88.9% | 100.0% | 100.0% |
| api:claude-opus-5-5@medium | 100.0% | 100.0% | 100.0% | 100.0% | 64.3% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| rule | 100.0% | 14.3% | 83.3% | 20.0% | 0.0% | 75.0% | 100.0% | 100.0% | 16.7% | 100.0% | 56.8% | 80.0% | 83.3% |
| small-dpo | 100.0% | 85.7% | 100.0% | 100.0% | 65.2% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 90.0% | 100.0% |
| small-sft | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 90.0% | 100.0% |
| small-small | 100.0% | 14.3% | 83.3% | 20.0% | 0.0% | 75.0% | 100.0% | 33.3% | 8.3% | 100.0% | 48.6% | 70.0% | 83.3% |
