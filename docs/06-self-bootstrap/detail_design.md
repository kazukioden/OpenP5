# Detailed Design — `self-bootstrap`

Reuse OpenP5's T5 pipeline; add three pieces + a round driver.

## Piece 1 — weighted, pseudo-augmentable training (foundation)
Pseudo-labels are extra `(prompt, target)` training rows at low weight; real
rows have weight 1. Eval untouched.

- CLI: `--pseudo_file` (path, optional), `--pseudo_weight` λ (default 0.3).
- `MultiTaskDataset` (train mode): after building real `data['input']/['output']`,
  if `pseudo_file` is set, append pseudo rows and maintain a parallel
  `data['weight']` (1.0 real, λ pseudo). Pseudo file format per line:
  `user_id <TAB> target_item_id` — the prompt is rebuilt from that user's real
  train history (same templates), only the *target* is the model's pseudo-label.
- `Collator`: return a `weight` tensor alongside the batch.
- `SingleRunner.train`: the per-example masked loss is already computed
  (before `.mean()`); multiply by the batch weights, then normalize by the
  weight sum. loss = Σ w_i·ℓ_i / Σ w_i.

Smoke: a dummy pseudo file trains with weighting, no crash; `--pseudo_file`
absent == original behavior.

## Piece 2 — Grow + Filter (pseudo-label generation)
`gen_pseudo.py`: load a trained checkpoint (frozen), for each **train** user
run the constrained beam search over the item Trie to get top-k next-item
candidates + sequence scores.

- **Filter (reward proxy):** keep a candidate if `score ≥ τ` (confidence) AND it
  passes a consistency check (candidate item ∈ neighbors of the user's history
  under item co-occurrence) AND is not already the user's train target. Cap N
  pseudo per user; optionally a global diversity cap to fight collapse.
- Write kept `(user_id, item_id)` to `pseudo_round{r}.txt`.

## Piece 3 — Round driver
`rest_selftrain.sh`:
```
train M0 (no pseudo)                         # round 0
for r in 1..R:
  gen_pseudo.py --load M_{r-1} -> pseudo_r   # Grow+Filter (frozen target model)
  train M_r --pseudo_file pseudo_r           # Improve (real + pseudo@λ)
eval M_R
```
Frozen `M_{r-1}` generates targets while `M_r` learns = the RL target-network
trick (stability).

## Metrics beyond Hit@k
Log **coverage / diversity** (e.g., #distinct items recommended across test
users, entropy of the recommended-item distribution) per round — the canary for
self-reinforcement collapse.

## Config
Same as 探求①: LastFM, sample 3,3 / batch128 / valid_select=1 / t5-small.
Rounds R∈{1,2}, λ=0.3, top-k=5, τ tuned on a quick check.
