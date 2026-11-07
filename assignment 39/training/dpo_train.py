"""DPO: tối ưu quyết định điều phối của Router sau SFT.

Policy    = model SFT (base + adapter SFT), tiếp tục train chính adapter đó.
Reference = bản sao model SFT, đóng băng (DPO kéo xác suất chosen lên / rejected xuống *so với* SFT).
Với 1.5B ở fp16, 2 bản model ~6GB VRAM -> Colab T4 đủ. Model 3B: thêm --qlora.

    python -m training.gen_dpo_pairs [--sft-adapter training/checkpoints/sft]
    python -m training.dpo_train --beta 0.1 --epochs 1

Output: training/checkpoints/dpo (adapter LoRA, load lên base giống adapter SFT).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from training.sft_train import eval_every, load_model, total_steps  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=config.SMALL_BASE_MODEL)
    ap.add_argument("--sft", default=str(config.SFT_ADAPTER))
    ap.add_argument("--out", default=str(config.DPO_ADAPTER))
    ap.add_argument("--qlora", action="store_true")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=1536)
    a = ap.parse_args()

    from datasets import load_dataset
    from peft import PeftModel
    from trl import DPOConfig, DPOTrainer

    data = load_dataset("json", data_files={"train": str(config.TRAIN_DATA_DIR / "dpo_train.jsonl"),
                                            "val": str(config.TRAIN_DATA_DIR / "dpo_val.jsonl")})
    data = data.select_columns(["prompt", "chosen", "rejected"])

    base, tok, bf16 = load_model(a.base, a.qlora)
    model = PeftModel.from_pretrained(base, a.sft, is_trainable=True)
    ref_base, _, _ = load_model(a.base, a.qlora)
    ref_model = PeftModel.from_pretrained(ref_base, a.sft, is_trainable=False).eval()
    every = eval_every(len(data["train"]), a.batch, a.grad_accum, a.epochs)
    print(f"train={len(data['train'])} val={len(data['val'])} | eval/save mỗi {every} step")

    args = DPOConfig(
        output_dir=a.out, beta=a.beta, loss_type="sigmoid", num_train_epochs=a.epochs, learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.1 * total_steps(len(data["train"]), a.batch, a.grad_accum, a.epochs))), per_device_train_batch_size=a.batch,
        per_device_eval_batch_size=a.batch, gradient_accumulation_steps=a.grad_accum, gradient_checkpointing=True,
        max_length=a.max_length, bf16=bf16, fp16=not bf16, logging_steps=min(10, every), eval_strategy="steps",
        eval_steps=every, save_strategy="steps", save_steps=every, save_total_limit=2, report_to="none", seed=39,
    )
    try:
        trainer = DPOTrainer(model=model, ref_model=ref_model, args=args, train_dataset=data["train"],
                             eval_dataset=data["val"], processing_class=tok)
    except ValueError as e:
        if "ref_model" not in str(e):
            raise
        # TRL >= 1.x không nhận ref_model là PeftModel: reference = chính model với adapter bị tắt (= base model)
        print("TRL mới: bỏ ref_model -> reference = base model (adapter disabled)")
        import torch

        del ref_model, ref_base
        torch.cuda.empty_cache()
        trainer = DPOTrainer(model=model, args=args, train_dataset=data["train"],
                             eval_dataset=data["val"], processing_class=tok)
    trainer.train()
    trainer.save_model(a.out)
    tok.save_pretrained(a.out)
    # Theo dõi trong log: rewards/chosen tăng, rewards/rejected giảm, rewards/accuracies -> ~1.0
    print(f"Đã lưu adapter DPO -> {a.out}")


if __name__ == "__main__":
    main()
