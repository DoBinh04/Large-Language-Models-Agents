#!/usr/bin/env bash
# Pipeline trên máy GPU thuê (4x RTX 5090). Mỗi stage chạy riêng để kiểm soát chi phí API giữa các bước.
#
#   bash training/remote_run.sh setup            # kiểm tra GPU, cài thư viện, test offline
#   bash training/remote_run.sh train            # data + SFT + DPO (GPU, không cần key)
#   bash training/remote_run.sh route-small      # routing-only: rule / base / SFT / DPO (GPU, không cần key)
#   bash training/remote_run.sh key-check        # 1 lời gọi Haiku rất nhỏ để kiểm tra key
#   bash training/remote_run.sh route-api        # routing-only router Claude (tốn tiền)
#   bash training/remote_run.sh e2e CFG [CFG..]  # end-to-end + LLM Judge (tốn tiền)
#   bash training/remote_run.sh spend            # xem sổ chi tiêu
#   bash training/remote_run.sh report           # results/comparison.md + dashboard.html
#
# Key chỉ nằm trong .env (chmod 600, ghi qua ssh stdin) và bị xoá khi xong việc.
# Mọi lời gọi Claude dừng hẳn khi sổ chi tiêu (results/spend.json) chạm API_BUDGET_USD.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f /venv/main/bin/activate ] && source /venv/main/bin/activate

export BASE="${BASE:-Qwen/Qwen2.5-1.5B-Instruct}"
export SMALL_BASE_MODEL="$BASE"
export API_BUDGET_USD="${API_BUDGET_USD:-19.5}"
export JUDGE_MODEL="${JUDGE_MODEL:-claude-sonnet-5-5}"
export JUDGE_EFFORT="${JUDGE_EFFORT:-low}"
export TOKENIZERS_PARALLELISM=false
GPU="${GPU:-0}"
LOG=results/logs
mkdir -p "$LOG"

need_key() {
  [ -s .env ] && grep -q '^ANTHROPIC_API_KEY' .env || { echo "Thiếu .env có ANTHROPIC_API_KEY" >&2; exit 2; }
}
spend() { python -c "from support.llm import spent; import json; print(json.dumps(spent()))"; }

case "${1:-}" in
setup)
  python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "| GPUs:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(" ", i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
print("bf16 matmul ok:", float((x @ x).float().abs().mean()))
EOF
  pip install -q "transformers>=4.45" "peft>=0.12" "accelerate>=0.33" "trl>=0.19" "datasets>=2.20" \
      "anthropic>=1.0" "mcp>=1.10" numpy "sentence-transformers>=3.0" huggingface_hub
  python -c "import trl, peft, transformers; print('trl', trl.__version__, '| peft', peft.__version__, '| transformers', transformers.__version__)"
  python -m tests.test_offline
  ;;

train)
  python -m training.gen_data --n "${N_TRAIN:-1500}" 2>&1 | tee "$LOG/gen_data.log"
  python -m training.hf_import --mode append 2>&1 | tee "$LOG/hf_import.log"
  # baseline Small LLM chưa train chạy song song trên GPU 1 trong lúc SFT trên GPU $GPU
  CUDA_VISIBLE_DEVICES=1 python -m eval.routing_eval --routers rule small --test-limit ${SMALL_TEST_LIMIT:-120} > "$LOG/route_base.log" 2>&1 &
  BASE_PID=$!
  CUDA_VISIBLE_DEVICES=$GPU python -m training.sft_train --base "$BASE" --epochs "${SFT_EPOCHS:-2}" 2>&1 | tee "$LOG/sft.log"
  CUDA_VISIBLE_DEVICES=$GPU python -m training.gen_dpo_pairs --base "$BASE" --sft-adapter training/checkpoints/sft \
      --onpolicy-n "${ONPOLICY_N:-120}" 2>&1 | tee "$LOG/dpo_pairs.log"
  CUDA_VISIBLE_DEVICES=$GPU python -m training.dpo_train --base "$BASE" --beta "${DPO_BETA:-0.1}" \
      --epochs "${DPO_EPOCHS:-1}" 2>&1 | tee "$LOG/dpo.log"
  wait "$BASE_PID" && tail -n 5 "$LOG/route_base.log"
  ;;

route-small)
  CUDA_VISIBLE_DEVICES=1 python -m eval.routing_eval --routers sft --test-limit ${SMALL_TEST_LIMIT:-120} > "$LOG/route_sft.log" 2>&1 &
  P1=$!
  CUDA_VISIBLE_DEVICES=2 python -m eval.routing_eval --routers dpo --test-limit ${SMALL_TEST_LIMIT:-120} > "$LOG/route_dpo.log" 2>&1 &
  P2=$!
  wait "$P1" "$P2"
  tail -n 3 "$LOG/route_sft.log" "$LOG/route_dpo.log"
  ;;

key-check)
  need_key
  python - <<'EOF'
from support.llm import Claude, spent
print(Claude(model="claude-haiku-5-5", effort="low", max_tokens=200).text("Trả lời đúng 1 từ.", "Xin chào"))
print(spent())
EOF
  ;;

route-api)
  need_key
  python -m eval.routing_eval --routers api --api-model claude-haiku-5-5 --api-effort low \
      --test-limit "${API_TEST_LIMIT:-60}" 2>&1 | tee "$LOG/route_api_haiku.log"
  spend
  python -m eval.routing_eval --routers api --api-model claude-opus-5-5 --api-effort medium \
      --test-limit "${API_TEST_LIMIT:-60}" 2>&1 | tee "$LOG/route_api_opus.log"
  spend
  ;;

e2e)
  need_key
  shift
  for c in "$@"; do
    CUDA_VISIBLE_DEVICES=$GPU python -m eval.run_eval --configs "$c" 2>&1 | tee "$LOG/e2e_$c.log"
    spend
  done
  ;;

spend) spend ;;
report) python -m eval.report ;;
*) sed -n '2,13p' "$0"; exit 1 ;;
esac
