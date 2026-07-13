"""
MLX side: prove a T5P5 training step works (loss decreases on a fixed batch)
and measure step time. Shapes mirror P5 on ML100K: long encoder input, short
decoder target.

Run: ../.venv-mlx/bin/python train_bench_mlx.py
"""
import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from t5_mlx import T5P5

B, L, T, VOCAB = 8, 128, 8, 32128
STEPS_MEASURE, WARMUP, OVERFIT_STEPS = 20, 3, 30


def make_batch(seed=0):
    key = mx.random.key(seed)
    k1, k2, k3, k4 = mx.random.split(key, 4)
    input_ids = mx.random.randint(0, VOCAB, (B, L), key=k1)
    whole_word_ids = mx.random.randint(0, 60, (B, L), key=k2)
    dec_in = mx.random.randint(0, VOCAB, (B, T), key=k3)
    labels = mx.random.randint(0, VOCAB, (B, T), key=k4)
    label_mask = mx.ones((B, T))
    return input_ids, whole_word_ids, dec_in, labels, label_mask


def loss_fn(model, input_ids, wwids, dec_in, labels, mask):
    logits = model(input_ids, dec_in, wwids)          # (B, T, V)
    ce = nn.losses.cross_entropy(logits, labels, reduction="none")  # (B, T)
    return (ce * mask).sum() / mask.sum()


def main():
    model, _ = T5P5.from_pretrained("t5-small", dtype=mx.float32)
    opt = optim.AdamW(learning_rate=1e-3)
    lg = nn.value_and_grad(model, loss_fn)
    batch = make_batch()

    # --- correctness: loss must decrease when overfitting one fixed batch ---
    losses = []
    for _ in range(OVERFIT_STEPS):
        loss, grads = lg(model, *batch)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
        losses.append(float(loss))
    print(f"[loss] start={losses[0]:.4f}  end={losses[-1]:.4f}  "
          f"decreased={losses[-1] < losses[0]}")

    # --- timing helper ---
    def bench(compiled: bool):
        model, _ = T5P5.from_pretrained("t5-small", dtype=mx.float32)
        opt = optim.AdamW(learning_rate=1e-3)
        lg = nn.value_and_grad(model, loss_fn)

        def raw_step(*b):
            loss, grads = lg(model, *b)
            opt.update(model, grads)
            return loss

        if compiled:
            state = [model.state, opt.state]
            step = mx.compile(raw_step, inputs=state, outputs=state)
        else:
            step = raw_step

        for _ in range(WARMUP):
            loss = step(*batch)
            mx.eval(model.parameters(), opt.state)
        t0 = time.perf_counter()
        for _ in range(STEPS_MEASURE):
            loss = step(*batch)
            mx.eval(model.parameters(), opt.state)
        return (time.perf_counter() - t0) / STEPS_MEASURE

    dt_raw = bench(compiled=False)
    dt_cmp = bench(compiled=True)
    print(f"[time] MLX eager   : {dt_raw*1000:.1f} ms/step")
    print(f"[time] MLX compiled: {dt_cmp*1000:.1f} ms/step  "
          f"(B={B}, L={L}, T={T}, {STEPS_MEASURE} steps)")


if __name__ == "__main__":
    main()
