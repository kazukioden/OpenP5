# Overall Design — `mlx-t5-poc`

> **This branch = Stage ② (MLX proof-of-concept) ONLY.** It is independent of
> Stage ① (`kv-cache-eval-speedup`). It does **not** modify the PyTorch OpenP5
> code; it adds a self-contained MLX experiment under `mlx_poc/`.
>
> **Status: PoC complete.** Port is faithful (parity ~3e-5). MLX training works;
> speed = MLX-compiled **~1.5x** over PyTorch/MPS, while naive MLX ≈ MPS.
> Recommendation: a full MLX port is **not** justified just to reproduce the
> blog (generation is unported and a cloud GPU beats 1.5x). See `test_design.md`
> Findings.

## Question this PoC answers
Stage ① showed the KV-cache fix helps eval only modestly on MPS and does nothing
for **training**, which is the real bottleneck (PyTorch's MPS path underuses the
M3 GPU). MLX is Apple's own array framework (unified memory, lazy eval + fusion)
and can, in principle, use the M3 far better.

**PoC goal:** determine whether re-implementing the T5 (P5) *model + training
step* in MLX is (a) faithful and (b) actually faster on this Mac — enough signal
to decide if a full port is worth it. Deliberately small and disposable.

## What we build
1. A t5-small encoder–decoder in MLX (`mlx_poc/t5_mlx.py`), plus P5's
   whole-word embedding added to the encoder input.
2. A weight loader that maps HuggingFace `t5-small` weights into the MLX model.
3. A training step in MLX: seq2seq cross-entropy with label masking + AdamW.
4. Validation + benchmark harnesses (see `test_design.md`).

## What we deliberately DON'T build (PoC boundaries)
- **Generation / constrained beam search in MLX** — the expensive, hard part.
  For any eval we keep using the HF/PyTorch path. This PoC only needs to prove
  *model + training*.
- Full training to convergence; matching paper/blog metrics.
- LLaMA backbone; collaborative/random indexing; the OpenP5 data/prompt plumbing
  (we reuse the existing HF tokenizer + a tiny hand-built batch).

## Environment
Isolated `.venv-mlx` (Python 3.10): `mlx`, `torch==2.2.2`,
`transformers==4.38.2` (HF T5 reference for parity), `safetensors`, numpy,
sentencepiece. **Kept separate from `.venv-p5`** because `mlx-lm` drags in
`transformers>=5`, which would break the pinned 4.26 PyTorch OpenP5 path.

## Success criteria
- **Faithful:** MLX forward logits/loss match HF T5 within fp tolerance on a
  fixed input (with the whole-word term disabled for the base-parity check).
- **Signal on speed:** a like-for-like training-step time, MLX vs PyTorch/MPS,
  on identical shapes. A clear win motivates a fuller port; a wash/loss argues
  for cloud GPU instead.

## Decision output
A short findings note (appended to `test_design.md` Results) with the parity
numbers and the MLX-vs-MPS step time, plus a go/no-go recommendation on a
full MLX port.
