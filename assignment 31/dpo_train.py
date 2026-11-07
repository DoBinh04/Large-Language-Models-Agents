"""
Phần 2: Train DPO thật với TRL trên 1 model nhỏ + dataset preference.

Cài đặt:
    pip install -U trl transformers datasets accelerate matplotlib

Chạy:
    python dpo_train.py
"""
import json
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET = "trl-lib/ultrafeedback_binarized"   # chosen/rejected là hội thoại 2 lượt
N_TRAIN = 2000                                # nhỏ để chạy nhanh
BETA = 0.1
SEED = 42

# T4 (Colab free) có compute capability 7.5 -> KHÔNG có bf16 thật.
# torch.cuda.is_bf16_supported() có thể trả True trên T4 (giả lập) -> phải check capability.
HAS_CUDA = torch.cuda.is_available()
USE_BF16 = HAS_CUDA and torch.cuda.get_device_capability()[0] >= 8   # A100 / L4
USE_FP16 = HAS_CUDA and not USE_BF16                                 # T4

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# Load fp32 (master weights) + để Trainer dùng mixed precision (autocast).
# Nếu load thẳng bf16 rồi train với lr=5e-6, nhiều update nhỏ hơn độ chính xác
# của bf16 (~3 chữ số) sẽ bị làm tròn mất -> loss gần như không giảm.
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
model.config.use_cache = False  # cần thiết khi gradient checkpointing đang bật

ds = load_dataset(DATASET, split="train").shuffle(seed=SEED).select(range(N_TRAIN))
print(ds[0])  # nhìn thử 1 mẫu: prompt / chosen / rejected

args = DPOConfig(
    output_dir="dpo-out",
    beta=BETA,
    learning_rate=5e-6,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    max_length=512,
    logging_steps=5,           # log dày để vẽ đồ thị
    logging_first_step=True,   # để thấy loss ~0.693 ngay step 1
    gradient_checkpointing=True,
    seed=SEED,
    bf16=USE_BF16,
    fp16=USE_FP16,
    save_strategy="no",
    report_to="none",
)

trainer = DPOTrainer(model=model, args=args, train_dataset=ds, processing_class=tok)
trainer.train()

# ---- Lưu log & vẽ đồ thị ----
logs = [l for l in trainer.state.log_history if "loss" in l]   # bỏ dòng tổng kết "train_loss"
with open("dpo_logs.json", "w") as f:
    json.dump(trainer.state.log_history, f, indent=2)


def series(key):
    return [l["step"] for l in logs if key in l], [l[key] for l in logs if key in l]


fig, ax = plt.subplots(2, 3, figsize=(16, 8))
for a, key, title in [
    (ax[0, 0], "loss", "DPO loss (bắt đầu ~0.693 = log 2)"),
    (ax[0, 1], "rewards/margins", "reward margin = r_chosen - r_rejected"),
    (ax[0, 2], "rewards/accuracies", "accuracy: P(r_chosen > r_rejected)"),
    (ax[1, 0], ("rewards/chosen", "rewards/rejected"), "implicit reward chosen vs rejected"),
    (ax[1, 1], ("logps/chosen", "logps/rejected"), "log-prob tuyệt đối của policy"),
]:
    if isinstance(key, str):
        x, y = series(key)
        a.plot(x, y)
    else:
        for k in key:
            x, y = series(k)
            a.plot(x, y, label=k.split("/")[1])
        a.legend()
    a.set_title(title)
    a.set_xlabel("step")
    a.grid(alpha=0.3)
ax[1, 2].axis("off")
plt.tight_layout()
plt.savefig("dpo_curves.png", dpi=120)
print("Đã lưu dpo_curves.png và dpo_logs.json")
