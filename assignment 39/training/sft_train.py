"""SFT Router: fine-tune Small LLM (mặc định Qwen2.5-1.5B-Instruct) bằng LoRA / QLoRA với TRL.

Chạy trên GPU (Colab T4 16GB là đủ cho 1.5B QLoRA; 3B dùng --qlora):
    python -m training.gen_data --n 3000
    python -m training.sft_train --qlora --epochs 2
    python -m training.sft_train --base meta-llama/Llama-3.2-1B-Instruct --out training/checkpoints/sft-llama

Dataset ở dạng conversational prompt-completion (prompt = [system, user], completion = [assistant]),
nên SFTTrainer chỉ tính loss trên phần completion (JSON quyết định), không học lại prompt.
Output: adapter LoRA + tokenizer ở training/checkpoints/sft (vài chục MB).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


def load_model(base: str, qlora: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    kw = {"dtype": dtype, "device_map": "auto"}
    if qlora:
        from transformers import BitsAndBytesConfig

        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(base, **kw)
    if qlora:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    return model, tok, bf16


def lora_config(r: int):
    from peft import LoraConfig

    return LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])


def total_steps(n_train: int, batch: int, grad_accum: int, epochs: float) -> int:
    import math

    return max(1, int(math.ceil(n_train / (batch * grad_accum)) * epochs))


def eval_every(n_train: int, batch: int, grad_accum: int, epochs: float, cap: int = 100) -> int:
    """Số step giữa 2 lần eval/save: ~4 lần mỗi lần train, tối đa `cap` (dữ liệu nhỏ vẫn có eval + best ckpt)."""
    return max(5, min(cap, total_steps(n_train, batch, grad_accum, epochs) // 4 or 5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=config.SMALL_BASE_MODEL)
    ap.add_argument("--out", default=str(config.SFT_ADAPTER))
    ap.add_argument("--qlora", action="store_true", help="load 4-bit (QLoRA); không bật = LoRA trên bf16/fp16")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=1536)
    a = ap.parse_args()

    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    data = load_dataset("json", data_files={"train": str(config.TRAIN_DATA_DIR / "sft_train.jsonl"),
                                            "val": str(config.TRAIN_DATA_DIR / "sft_val.jsonl")})
    data = data.select_columns(["prompt", "completion"])
    model, tok, bf16 = load_model(a.base, a.qlora)
    every = eval_every(len(data["train"]), a.batch, a.grad_accum, a.epochs)
    print(f"train={len(data['train'])} val={len(data['val'])} | eval/save mỗi {every} step")

    args = SFTConfig(
        output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr, lr_scheduler_type="cosine",
        # warmup_steps (int) thay warmup_ratio: transformers 5 đã bỏ warmup_ratio
        warmup_steps=max(1, int(0.05 * total_steps(len(data["train"]), a.batch, a.grad_accum, a.epochs))), per_device_train_batch_size=a.batch, per_device_eval_batch_size=a.batch,
        gradient_accumulation_steps=a.grad_accum, gradient_checkpointing=True,
        max_length=a.max_length, completion_only_loss=True,
        bf16=bf16, fp16=not bf16, logging_steps=min(10, every), eval_strategy="steps", eval_steps=every,
        save_strategy="steps", save_steps=every, save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", report_to="none", seed=39,
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=data["train"], eval_dataset=data["val"],
                         processing_class=tok, peft_config=lora_config(a.rank))
    trainer.model.print_trainable_parameters()
    trainer.train()
    trainer.save_model(a.out)
    tok.save_pretrained(a.out)
    print(f"Đã lưu adapter SFT -> {a.out}")


if __name__ == "__main__":
    main()
