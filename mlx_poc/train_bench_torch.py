"""
PyTorch/MPS side: same-shape t5-small seq2seq training step, for a like-for-like
step-time comparison against MLX. Uses the pinned .venv-p5 (torch + tf 4.26).

Run: ../.venv-p5/bin/python train_bench_torch.py
"""
import time
import torch
from transformers import T5ForConditionalGeneration

B, L, T, VOCAB = 8, 128, 8, 32128
STEPS_MEASURE, WARMUP = 20, 3


def main():
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    torch.manual_seed(0)
    model = T5ForConditionalGeneration.from_pretrained("t5-small").to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    input_ids = torch.randint(0, VOCAB, (B, L), device=dev)
    attn = torch.ones(B, L, dtype=torch.long, device=dev)
    labels = torch.randint(0, VOCAB, (B, T), device=dev)

    def step():
        opt.zero_grad()
        out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        out.loss.backward()
        opt.step()
        return out.loss

    for _ in range(WARMUP):
        step()
    torch.mps.synchronize() if dev.type == "mps" else None

    t0 = time.perf_counter()
    for _ in range(STEPS_MEASURE):
        loss = step()
    if dev.type == "mps":
        torch.mps.synchronize()
    dt = (time.perf_counter() - t0) / STEPS_MEASURE
    print(f"[time] PyTorch ({dev.type}): {dt*1000:.1f} ms/step  "
          f"(B={B}, L={L}, T={T}, {STEPS_MEASURE} steps)")


if __name__ == "__main__":
    main()
