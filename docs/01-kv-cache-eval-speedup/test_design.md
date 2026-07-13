# Test Design — `kv-cache-eval-speedup`

## Invariant under test
KV caching is a **performance-only** optimization. For the same weights and
inputs, constrained beam-search generation must produce **identical** output.
So the primary test is an **exact-match** of generations before vs. after the
fix — not a metrics delta (aggregate Hit/NDCG on the tiny, undertrained model
are ~0 and would mask output drift).

## Harness
`tests/bench_generate.py` — standalone, imports `P5_T5`, `TestDataset`,
`Collator`, and the `generation_trie`.

- Loads `t5-small` + `checkpoint/tiny.pt` (a real, if lightly trained, model).
- Builds a fixed slice of `ML100K_tiny` test inputs (default n=16).
- Runs `model.generate(..., prefix_allowed_tokens_fn=trie, num_beams=10,
  num_return_sequences=10, return_dict_in_generate=True)`.
- Dumps, per input: top-10 decoded item strings + rounded sequence scores, and
  the total `generate` wall-time, to JSON.
- `--compare BEFORE AFTER` diffs two dumps and prints pass/fail + speedup.

Modes:
- **Correctness** on `--device cpu` (fully deterministic; expect exact match).
- **Performance** on `--device mps` (representative of real eval).

## Procedure
```
# 1. BEFORE (current code)
python tests/bench_generate.py --device cpu --task sequential --out /tmp/seq_before_cpu.json
python tests/bench_generate.py --device mps --task sequential --out /tmp/seq_before_mps.json

# 2. apply the P5_T5 fix

# 3. AFTER
python tests/bench_generate.py --device cpu --task sequential --out /tmp/seq_after_cpu.json
python tests/bench_generate.py --device mps --task sequential --out /tmp/seq_after_mps.json

# 4. verdict
python tests/bench_generate.py --compare /tmp/seq_before_cpu.json /tmp/seq_after_cpu.json
python tests/bench_generate.py --compare /tmp/seq_before_mps.json /tmp/seq_after_mps.json
```

## Pass criteria
- **Blocking:** CPU predictions identical across all samples (`PASS`).
- MPS predictions identical too (allow score rounding noise only if
  predictions themselves match; investigate if scores drift).
- Performance: AFTER `elapsed` < BEFORE on MPS (expected multiple-x).

## Coverage notes
- Exercises the constrained-Trie + beam-search + whole-word-embedding path end
  to end — the exact code eval uses.
- Both `sequential` and `straightforward` tasks can be run (swap `--task`).
- Not covered here: full-dataset metric reproduction (separate, expensive run).

## Results (2026-07-13, n=16, beams=10, checkpoint/tiny.pt)

| device | task | BEFORE (s) | AFTER (s) | speedup | preds identical |
|---|---|---|---|---|---|
| cpu | sequential | 7.45 | 2.45 | 3.04x | ✅ PASS |
| mps | sequential | 7.48 | 5.18 | 1.44x | ✅ PASS |

**Verdict:** correctness invariant holds (predictions byte-identical before/after
on both devices). Speedup is real but modest, and smaller on MPS than CPU.

**Why modest:** P5 generates very short targets (`<dataset> item_<id>` ≈ 6
tokens), so the O(L²)→O(L) decoder win is bounded by L≈6. On MPS the per-step
op-dispatch overhead of incremental (1-token) decoding further offsets the
compute savings. The dominant eval cost is elsewhere: beam width (10) ×
#users × the single encoder pass over long history prompts — none of which the
KV cache touches. Training speed is unaffected by this change.

**Implication for staging:** ① is a correct, free win but does not make MPS
evaluation dramatically faster, and does nothing for training. If overall Mac
speed is still the blocker, that motivates stage ② (MLX) — **which is pursued on
a separate branch, not here** — though note even MLX cannot beat a real GPU; for
pure reproduction speed, a cloud GPU remains best.
