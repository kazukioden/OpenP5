# Test Design — `mlx-t5-poc`

Two questions this PoC must answer, each with a concrete test.

## 1. Is the MLX port faithful? (correctness)
**Test:** `parity_check.py`. Load `t5-small` once via HF, copy the same weights
into the MLX model, run both on a fixed `(input_ids, decoder_input_ids)` with
the whole-word term off, compare decoder logits.

**Pass criteria:** identical argmax at every position AND relative mean abs diff
`< 1e-2`. (Small numeric diff is expected: different kernels / reduction order.)

## 2. Does MLX training work, and is it faster on this Mac? (function + speed)
**Test A (function):** `train_bench_mlx.py` overfits one fixed random batch for
30 steps; the loss must strictly decrease (loop + autograd + optimizer wired
correctly).

**Test B (speed):** measure ms/step (after warmup, `mx.eval`-synced) for MLX
eager and MLX `mx.compile`d, and compare to the same-shape PyTorch/MPS step from
`train_bench_torch.py`. Shapes mirror P5 on ML100K: B=8, encoder L=128,
decoder T=8, vocab 32128, fp32.

**Pass criteria:** loss decreases (A). For (B) there is no hard threshold — the
number *is* the finding that informs the go/no-go.

## Results (2026-07-13, M3, fp32)

### Parity
```
max_abs_diff = 3.43e-05   mean_rel = 2.65e-07   cosine = 1.0000000   argmax match = True
=> PARITY PASS
```
Only unfilled parameter: `whole_word_embeddings.weight` (P5 addition, expected).

### Training + speed
```
[loss] start=19.02  end=0.0001  decreased=True     (overfit sanity: PASS)

step time (B=8, L=128, T=8):
  PyTorch / MPS     170.8 ms/step   (baseline)
  MLX  eager        177.9 ms/step   ~1.0x  (no faster than MPS)
  MLX  compiled     111.0 ms/step   1.54x faster than MPS
```

## Findings / recommendation
- **Faithful:** yes — the MLX forward matches HF T5 to ~3e-5.
- **Speed:** a *naive* MLX port is **not** faster than PyTorch/MPS. The gain
  comes entirely from `mx.compile` (op fusion), which lands **~1.5x** over MPS.
  This concretely confirms the earlier hypothesis: on the same M3, the lever is
  *compilation/fusion*, not the framework label.
- **Go/no-go on a full MLX port:** **borderline / likely no** for the stated
  goal (reproduce the blog). ~1.5x on training does not justify reimplementing
  constrained beam-search generation in MLX (the hard, unported part), and a
  cloud GPU beats 1.5x by a wide margin. A full MLX port makes sense only if
  "must run fast, fully on this Mac" is a hard requirement, or as a learning
  exercise. The PoC has produced the number needed to decide; it stops here.

## Not covered
Generation/beam search in MLX; padding-mask paths; multi-batch throughput;
convergence/metric reproduction; fp16/bf16 (fp32 only, for a clean comparison).
