# Detailed Design — `mlx-t5-poc`

Self-contained MLX experiment under `mlx_poc/`. Does not touch OpenP5's PyTorch
code. Runs in the isolated `.venv-mlx`.

## Files
- `t5_mlx.py` — t5-small (P5 variant) in MLX. Adapted from ml-explore
  `mlx-examples/t5/t5.py` (Apache-2.0), trimmed to forward + training.
- `parity_check.py` — MLX vs HF T5 logits parity (`.venv-mlx`).
- `train_bench_mlx.py` — MLX training-step correctness + timing (`.venv-mlx`).
- `train_bench_torch.py` — same-shape PyTorch/MPS step timing (`.venv-p5`).

## Model port (`t5_mlx.py`)
T5-specific details that must be right:
- **No attention scaling** — T5 uses raw `q @ k` (no `1/sqrt(d)`).
- **RMSNorm** (`T5LayerNorm`), pre-norm residual blocks.
- **Relative position bias** — bidirectional buckets for the encoder, causal
  for the decoder; bias added into the attention scores (shared across layers,
  stored on layer 0 in HF, hoisted to the stack in this layout).
- **Tied output** — `logits = (h * d_model**-0.5) @ wte.weight.T` when
  `tie_word_embeddings` (t5-small does).

### Fix vs. the reference
The upstream `DenseActivation` sets `self.gated = hasattr(config,
"feed_forward_proj")`, which is `True` for original t5-small
(`feed_forward_proj="relu"`) and would wrongly expect gated `wi_0/wi_1`. Changed
to `self.gated = ff.startswith("gated-")`, so t5-small uses a single `wi` and
matches the HF `DenseReluDense.{wi,wo}` weights.

### P5 addition
`whole_word_embeddings = nn.Embedding(512, d_model)`; `encode()` adds
`whole_word_embeddings(whole_word_ids)` to the token embeddings, mirroring
`JointEncoder` in `src/src_t5/model/P5_T5.py`. It has no pretrained weight and
is left unfilled (the base-parity test passes `whole_word_ids=None`).

## Weight loading
HF `t5-small` `state_dict` → numpy → `mx.array`, then `sanitize()` renames HF
keys to this module's layout (`block→layers`, `q/k/v/o→*_proj`, `shared→wte`,
`layer.N.layer_norm→lnK`, `final_layer_norm→ln`, relative-bias key hoist). Tied
duplicates (`encoder/decoder.embed_tokens`, `lm_head`) are dropped by keeping
only keys present in `tree_flatten(model.parameters())`.

## Training step
`loss_fn` = masked token cross-entropy over the decoder logits;
`nn.value_and_grad` + `optim.AdamW`. Two variants benchmarked:
- **eager**: call the step, then `mx.eval(params, opt.state)`.
- **compiled**: wrap the step in `mx.compile(step, inputs=state, outputs=state)`
  where `state = [model.state, opt.state]` — this is where MLX's op fusion kicks
  in (the analogue of XLA compilation).

## Boundaries
No generation/beam search in MLX (the expensive part); eval keeps using the
PyTorch path. Random fixed batches are used (goal: prove the loop + measure
speed, not reproduce metrics).
