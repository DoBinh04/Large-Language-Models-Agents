"""Sinh notebook Colab TỰ CHỨA: nhúng sẵn đúng các file code + data cần cho training (bằng %%writefile),
nên trên Colab chỉ cần upload 1 file .ipynb, không cần zip / Drive.

    python training/build_colab_notebook.py            # -> training/train_router_colab.ipynb

Sửa code ở máy (router.py, gen_data.py, ...) thì chạy lại lệnh trên để notebook cập nhật theo.
Script kiểm tra import: nếu một file trong danh sách import module nội bộ chưa được nhúng thì báo lỗi.
Chỉ dùng thư viện chuẩn, chạy tức thì.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "training" / "train_router_colab.ipynb"
WORK = "/content/a39"

# Đúng những gì training + routing eval cần (không kéo RAG / agents / MCP / API server)
FILES = [
    "config.py",
    "support/state.py", "support/tracing.py", "support/router.py", "support/llm.py",
    "eval/metrics.py", "eval/routing_eval.py",
    "training/gen_data.py", "training/hf_import.py", "training/gen_dpo_pairs.py",
    "training/sft_train.py", "training/dpo_train.py",
    "tests/test_training.py",
    "data/products.json", "data/eval_set.json",
]
PACKAGES = ["support", "eval", "training", "tests"]
LOCAL_TOP = {"config", *PACKAGES}


def check_imports() -> None:
    have = {f[:-3].replace("/", ".") for f in FILES if f.endswith(".py")}
    missing = []
    for f in FILES:
        if not f.endswith(".py"):
            continue
        tree = ast.parse((ROOT / f).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                mods = [node.module]
                if node.module in PACKAGES:                  # from support import a2a
                    mods = [f"{node.module}.{a.name}" for a in node.names]
            for m in mods:
                if m.split(".")[0] in LOCAL_TOP and m not in have and m not in PACKAGES:
                    missing.append(f"{f}: import {m}")
    if missing:
        sys.exit("Thiếu file trong FILES:\n  " + "\n  ".join(missing))


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str, collapsed: bool = False) -> dict:
    meta = {"cellView": "form", "jupyter": {"source_hidden": True}} if collapsed else {}
    return {"cell_type": "code", "execution_count": None, "metadata": meta, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


INTRO = """
# Assignment 3.9: Train Router Small LLM (SFT → DPO) trên Colab

**Chỉ cần upload đúng file notebook này** lên Colab (*File → Upload notebook*). Source code cần thiết đã được nhúng sẵn
(phần *Source code*, mỗi cell `%%writefile` ghi ra 1 file), dữ liệu lấy thẳng từ Hugging Face
[`Tson29/techviet-cs-orchestrator-data`](https://huggingface.co/datasets/Tson29/techviet-cs-orchestrator-data).

1. *Runtime → Change runtime type → **T4 GPU***.
2. Sửa cell **1. Cấu hình** nếu muốn, rồi *Runtime → Run all*. T4 + 1.5B: khoảng 1 giờ; 3B (QLoRA): khoảng 2 giờ.
3. Cuối notebook tải về `a39_router_artifacts.zip` (adapter `sft/`, `dpo/`, dữ liệu, bảng routing eval).
   Giải nén vào thư mục `assignment 39/` trên máy để chạy `python run_chat.py --router dpo`.

`ANTHROPIC_API_KEY` chỉ cần khi `PARAPHRASE > 0` hoặc `EVAL_API = True` (🔑 *Secrets* bên trái, bật *Notebook access*).

> Notebook được sinh bởi `training/build_colab_notebook.py`. Sửa code ở máy rồi build lại, đừng sửa trong các cell `%%writefile`.
"""

CONFIG = """
# 1. Cấu hình
BASE_MODEL   = "Qwen/Qwen2.5-1.5B-Instruct"   # hoặc "Qwen/Qwen2.5-3B-Instruct" (giữ QLORA=True), "meta-llama/Llama-3.2-1B-Instruct"
DATA_SOURCE  = "hf+template"  # "hf": chỉ dataset HF | "template": chỉ gen_data | "hf+template": trộn (khuyên dùng)
HF_REPO      = "Tson29/techviet-cs-orchestrator-data"
N_TRAIN      = 1500      # số mẫu SFT sinh từ template (khi DATA_SOURCE có "template")
PARAPHRASE   = 0         # >0: Claude Haiku viết lại N câu cho đa dạng (cần ANTHROPIC_API_KEY, tốn vài cent)
QLORA        = True      # 4-bit, an toàn cho T4 16GB
SFT_EPOCHS   = 2         # DATA_SOURCE="hf" (chỉ vài trăm mẫu): nên để 3
DPO_BETA     = 0.1
DPO_EPOCHS   = 1
ONPOLICY_N   = 400       # số prompt lấy negatives on-policy từ model SFT (0 = chỉ negatives sinh bằng luật)
TEST_LIMIT   = 300       # số mẫu test cho routing eval
EVAL_API     = False     # True: so thêm router Claude API (cần key, tốn tiền)
SAVE_TO_DRIVE = False    # True: chép file kết quả vào Google Drive (MyDrive/a39_router_artifacts.zip), phòng Colab ngắt

import os
os.environ["SMALL_BASE_MODEL"] = BASE_MODEL
os.environ["TOKENIZERS_PARALLELISM"] = "false"
!nvidia-smi --query-gpu=name,memory.total --format=csv
"""

SETUP = f"""
# 2. Tạo thư mục làm việc
import os, pathlib
WORK = pathlib.Path("{WORK}")
for pkg in {PACKAGES!r} + ["data"]:
    (WORK / pkg).mkdir(parents=True, exist_ok=True)
for pkg in {PACKAGES!r}:
    (WORK / pkg / "__init__.py").touch()
os.chdir(WORK)
print("Làm việc tại", os.getcwd())
"""

AFTER = [
    md("## Cài đặt và kiểm tra"),
    code("""
# 3. Cài thư viện (torch có sẵn trên Colab)
!pip install -q "transformers>=4.45" "peft>=0.12" "accelerate>=0.33" "trl>=0.19" "datasets>=2.20" "bitsandbytes>=0.43" "anthropic>=1.0"
import torch, transformers, trl, peft
print("torch", torch.__version__, "| transformers", transformers.__version__, "| trl", trl.__version__, "| peft", peft.__version__)
assert torch.cuda.is_available(), "Chưa bật GPU: Runtime -> Change runtime type -> T4 GPU"
print("GPU:", torch.cuda.get_device_name(0), "| bf16:", torch.cuda.is_bf16_supported())
"""),
    code("""
# 4. (Tuỳ chọn) API key từ Colab Secrets, chỉ cần cho PARAPHRASE > 0 hoặc EVAL_API
key = None
try:
    from google.colab import userdata
    key = userdata.get("ANTHROPIC_API_KEY")
except Exception as e:
    print("Không đọc được secret ANTHROPIC_API_KEY:", type(e).__name__)
if key:
    os.environ["ANTHROPIC_API_KEY"] = key
    print("Đã nạp ANTHROPIC_API_KEY")
elif PARAPHRASE or EVAL_API:
    raise SystemExit("PARAPHRASE / EVAL_API cần ANTHROPIC_API_KEY trong Secrets")
else:
    print("Không dùng API, ổn: các bước train không gọi Claude")
"""),
    code("""
# 5. Test nhẹ (không GPU, không API): code + nhãn dữ liệu nhất quán trước khi tốn giờ GPU
!python -m tests.test_training
"""),
    md("## Dữ liệu"),
    code("""
# 6. Dữ liệu SFT: training/data/sft_{train,val,test}.jsonl
#    - template: training.gen_data (luật, đủ nhóm state-aware)
#    - hf      : training.hf_import convert hội thoại HF (orchestrator từng bước) -> RoutingDecision cả lượt
if DATA_SOURCE == "template":
    !python -m training.gen_data --n {N_TRAIN} --paraphrase {PARAPHRASE}
elif DATA_SOURCE == "hf":
    !python -m training.hf_import --repo {HF_REPO} --mode replace
elif DATA_SOURCE == "hf+template":
    !python -m training.gen_data --n {N_TRAIN} --paraphrase {PARAPHRASE}
    !python -m training.hf_import --repo {HF_REPO} --mode append
else:
    raise ValueError(DATA_SOURCE)
!wc -l training/data/*.jsonl
"""),
    code("""
# 6b. Xem thử 1 mẫu convert từ HF (prompt router nhìn thấy + nhãn)
import json
p = pathlib.Path("training/data/hf_sft_train.jsonl")
if p.exists():
    r = json.loads(p.open(encoding="utf-8").readline())
    print("kind:", r["kind"], "| conv:", r.get("conv_id"))
    print(r["prompt"][1]["content"])
    print("->", json.dumps(r["target"], ensure_ascii=False, indent=1))
"""),
    md("## Train"),
    code("""
# 7. Baseline: Small LLM chưa fine-tune (+ router luật), để có mốc so sánh
!python -m eval.routing_eval --routers rule small --test-limit {TEST_LIMIT}
"""),
    code("""
# 8. SFT (LoRA / QLoRA) -> training/checkpoints/sft   (theo dõi eval_loss giảm)
qlora = "--qlora" if QLORA else ""
!python -m training.sft_train --base {BASE_MODEL} {qlora} --epochs {SFT_EPOCHS}
!ls -la training/checkpoints/sft
"""),
    code("""
# 9. Cặp DPO: negatives theo 8 kiểu lỗi điều phối + on-policy (lỗi model SFT thật sự mắc)
#    (dpo/dpo.jsonl của HF là cặp mức "bước", khác schema -> không dùng; script sinh lại các lỗi tương đương)
onp = f"--sft-adapter training/checkpoints/sft --onpolicy-n {ONPOLICY_N}" if ONPOLICY_N else ""
!python -m training.gen_dpo_pairs --base {BASE_MODEL} {onp}
!wc -l training/data/dpo_*.jsonl
"""),
    code("""
# 10. DPO trên nền SFT (ref = SFT đóng băng) -> training/checkpoints/dpo
#     Theo dõi: rewards/accuracies -> ~1.0, rewards/chosen > rewards/rejected. OOM: thêm --batch 2 --grad-accum 8
import gc; gc.collect(); torch.cuda.empty_cache()
!python -m training.dpo_train --base {BASE_MODEL} {qlora} --beta {DPO_BETA} --epochs {DPO_EPOCHS}
!ls -la training/checkpoints/dpo
"""),
    md("## Đánh giá và tải kết quả"),
    code("""
# 11. So sánh router: rule vs base vs SFT vs DPO (+ API nếu bật)
routers = "rule small sft dpo" + (" api" if EVAL_API else "")
!python -m eval.routing_eval --routers {routers} --test-limit {TEST_LIMIT}
"""),
    code("""
# 12. Bảng accuracy theo nhóm (state-aware: pending / in_state / follow_product / multi ...)
import glob, pandas as pd
rows = {}
for p in sorted(glob.glob("results/routing/*.json")):
    r = json.load(open(p, encoding="utf-8"))
    s = r["summary"]
    row = {k: s.get(k) for k in ("routing_acc", "intent_acc", "routing_acc_eval_set", "routing_acc_test",
                                 "json_valid", "latency_ms_mean", "cost_per_1k_usd")}
    row.update({f"acc:{k}": v.get("acc") for k, v in s.get("by_kind", {}).items()})
    rows[r["router"]] = row
pd.DataFrame(rows).T
"""),
    code("""
# 13. Đóng gói adapter (bỏ checkpoint trung gian), dữ liệu, kết quả eval rồi tải về
import zipfile, shutil
out = pathlib.Path("/content/a39_router_artifacts.zip")
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for pat in ("training/checkpoints/sft/*", "training/checkpoints/dpo/*", "training/data/*", "results/routing/*"):
        for f in glob.glob(pat):
            if os.path.isfile(f):                         # bỏ thư mục checkpoint-XXX (optimizer state, rất nặng)
                z.write(f)
print(out, f"{out.stat().st_size / 1e6:.1f} MB")
if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    shutil.copy(out, "/content/drive/MyDrive/a39_router_artifacts.zip")
    print("Đã chép vào MyDrive/a39_router_artifacts.zip")
from google.colab import files
files.download(str(out))
"""),
]


def main() -> None:
    check_imports()
    cells = [md(INTRO), code(CONFIG), md("## Source code (nhúng sẵn, tự sinh)"), code(SETUP)]
    for f in FILES:
        text = (ROOT / f).read_text(encoding="utf-8")
        if not text.endswith("\n"):
            text += "\n"
        cells.append({"cell_type": "code", "execution_count": None,
                      "metadata": {"cellView": "form", "jupyter": {"source_hidden": True}}, "outputs": [],
                      "source": f"%%writefile {WORK}/{f}\n{text}".splitlines(keepends=True)})
    cells += AFTER
    nb = {"cells": cells, "metadata": {
        "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Đã ghi {OUT.relative_to(ROOT)}: {len(cells)} cell, nhúng {len(FILES)} file "
          f"({sum((ROOT / f).stat().st_size for f in FILES) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
