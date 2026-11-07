"""
Phần 1: Hiểu DPO loss TỪ ĐẦU (chạy được trên CPU, ~1-2 phút).

DPO loss cho 1 cặp (prompt x, chosen y_w, rejected y_l):

    L = -log sigmoid( beta * [ (logpi(y_w) - logref(y_w)) - (logpi(y_l) - logref(y_l)) ] )

Trong đó logpi(y) = tổng log-prob các token của câu trả lời y (không tính prompt).
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
BETA = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

tok = AutoTokenizer.from_pretrained(MODEL)
# fp32 cho cả 2: policy==ref phải cho margin đúng bằng 0 ở thí nghiệm 1
policy = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEVICE)
ref = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEVICE)
policy.eval()  # tắt dropout để loss tất định (vẫn train được, eval() không chặn gradient)
ref.eval()
for p in ref.parameters():
    p.requires_grad_(False)

prompt = "Question: What is the capital of France?\nAnswer:"
chosen = " The capital of France is Paris."
rejected = " I don't know, maybe Berlin."


def seq_logprob(model, prompt, response):
    """Tổng log-prob của các token response, có điều kiện trên prompt."""
    # Tokenize RIÊNG rồi nối id: nếu tokenize (prompt + response) thì BPE có thể
    # gộp token ở ranh giới -> n_prompt bị lệch, tính nhầm token prompt/response.
    p_ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    r_ids = tok(response, return_tensors="pt", add_special_tokens=False).input_ids
    full_ids = torch.cat([p_ids, r_ids], dim=1).to(DEVICE)
    logits = model(full_ids).logits[:, :-1]          # logits[i] dự đoán token i+1
    targets = full_ids[:, 1:]
    logps = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    n_prompt = p_ids.shape[1]
    return logps[:, n_prompt - 1:].sum()             # chỉ tính phần response


def dpo_loss(policy, ref, prompt, chosen, rejected, beta=BETA):
    pi_c = seq_logprob(policy, prompt, chosen)
    pi_r = seq_logprob(policy, prompt, rejected)
    with torch.no_grad():
        ref_c = seq_logprob(ref, prompt, chosen)
        ref_r = seq_logprob(ref, prompt, rejected)
    reward_c = beta * (pi_c - ref_c)                 # "implicit reward" của chosen
    reward_r = beta * (pi_r - ref_r)                 # "implicit reward" của rejected
    margin = reward_c - reward_r
    loss = -F.logsigmoid(margin)
    return loss, reward_c.item(), reward_r.item(), margin.item(), pi_c.item(), pi_r.item()


# ---- Thí nghiệm 1: policy == reference -> loss PHẢI bằng log(2) = 0.6931 ----
loss, rc, rr, m, lpc, lpr = dpo_loss(policy, ref, prompt, chosen, rejected)
print("=== Thí nghiệm 1: policy == ref ===")
print(f"logp(chosen)={lpc:.2f}  logp(rejected)={lpr:.2f}   <- khác nhau, nhưng ref cũng y hệt")
print(f"reward_chosen={rc:.4f}  reward_rejected={rr:.4f}  margin={m:.4f}")
print(f"loss={loss.item():.4f}   (log 2 = {torch.log(torch.tensor(2.0)).item():.4f})")
print("-> reward là log-ratio so với ref, nên ban đầu = 0 dù model đã 'thích' chosen hơn.")

# ---- Thí nghiệm 2: train overfit 1 cặp này, xem loss/margin thay đổi ----
print("\n=== Thí nghiệm 2: overfit 1 cặp (30 bước) ===")
print("grad_w = sigmoid(-margin): trọng số gradient, càng nhỏ thì update càng yếu")
opt = torch.optim.AdamW(policy.parameters(), lr=1e-5)
for step in range(30):
    loss, rc, rr, m, lpc, lpr = dpo_loss(policy, ref, prompt, chosen, rejected)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 3 == 0 or step == 29:
        gw = torch.sigmoid(torch.tensor(-m)).item()
        print(f"step {step:2d} | loss={loss.item():.4f} | r_c={rc:+.3f} | r_r={rr:+.3f} "
              f"| margin={m:+.3f} | grad_w={gw:.3f} | logp_c={lpc:7.2f} | logp_r={lpr:7.2f}")
print("Quan sát: r_rejected giảm mạnh hơn r_chosen tăng (thậm chí logp_c có thể giảm) —")
print("DPO chỉ tối ưu HIỆU, không bắt buộc chosen tăng xác suất tuyệt đối.")

# ---- Thí nghiệm 3: bảng loss theo margin (không cần model) ----
print("\n=== Thí nghiệm 3: loss = -log sigmoid(margin), dL/dmargin = -sigmoid(-margin) ===")
for margin in [-2, -1, 0, 0.5, 1, 2, 4, 8]:
    t = torch.tensor(float(margin))
    l = -F.logsigmoid(t).item()
    g = torch.sigmoid(-t).item()
    print(f"margin={margin:+5.1f} -> loss={l:.4f}   |grad|={g:.4f}")
print("-> loss giảm nhanh lúc đầu rồi 'bão hoà' khi margin lớn vì gradient -> 0.")
