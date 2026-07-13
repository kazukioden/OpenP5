# Overall Design — `kv-cache-eval-speedup`

## Background
OpenP5 (T5 backbone) now runs on the M3 Mac (MPS) after branch `mac-mps-support`.
But it is slow. Two cost centers:

1. **Training** — MPS-bound matmul throughput. Not addressable without a
   different compute backend (see the MLX exploration, stage ②).
2. **Evaluation (generation)** — constrained beam search over the item Trie.
   This is *artificially* slow because of a bug that disables the KV cache.

This branch tackles **(2)**, the cheap, high-leverage win: fix the KV cache so
evaluation runs at its intended speed, with **zero change to results**.

## Root cause (summary)
`P5_T5.prepare_inputs_for_generation` declares its cache argument as `past`,
but transformers 4.26 passes the decoder cache under the keyword
`past_key_values`. The cache therefore never reaches the model during
generation, so every decode step recomputes the full decoder sequence
(O(L²) instead of O(L)). See `detail_design.md`.

## Approach (staged)
- **Stage ① (this branch):** repair `prepare_inputs_for_generation` so the KV
  cache flows through. Verify identical outputs + measure the speedup.
- **Stage ② (only if ① is insufficient):** PoC-port just the T5 forward +
  training loop to Apple's MLX (reusing the HF tokenizer and, initially, HF
  generation for validation). Tracked on a separate branch if pursued.

## Scope
**In:** `src/src_t5/model/P5_T5.py` generation-input plumbing; an A/B
benchmark harness (`tests/bench_generate.py`); this documentation.

**Out:** training-loop speed; the LLaMA path; collaborative indexing; any
change to model weights, prompts, metrics, or data.

## Success criteria
- **Correctness (blocking):** constrained beam-search predictions (top-k item
  ids + scores) are **identical** before/after, on a fixed checkpoint + inputs.
- **Performance:** evaluation generation is measurably faster on MPS.

## Non-goals / risks
- KV caching must not alter outputs; if predictions diverge, the fix is wrong
  and is reverted. This is why the test is an exact-match invariant, not a
  metric-delta check.
- t5-small numbers on ML100K are small; aggregate metrics are too coarse to
  detect output drift, so we compare raw generations instead.
