# Detailed Design — `indexing-comparison`

## The collaborative-indexing fix
`src/src_t5/data/MultiTaskDataset.py`, non-distributed branch, called:
```python
indexing.collaborative_indexing(..., self.collaborative_last_token,
                                 self.collaborative_sparse,      # <- undefined + wrong arity
                                 self.collaborative_float32)
```
but the function is
`collaborative_indexing(data_path, dataset, user_sequence_dict, token_size,
cluster_num, last_token, float32)` and the *distributed* branch (and
`TestDataset`) already call it correctly. `self.collaborative_sparse` is never
set in `__init__`, so this raised `AttributeError` the moment you ran
collaborative indexing on a single device. Fix: drop the stray argument to match
the signature.

Verified (tiny smoke, MPS): collaborative now indexes and trains; items become
hierarchical codes, e.g. `item_<CI0><CI7><CI3><CI9><CI3>` (spectral-cluster
tree), and training starts normally.

## Run harness
`remote_indexing_sweep.sh` (runs detached on the GPU box). Same config for all
three methods, only `--item_indexing` changes; each method writes to
`outputs/<method>/` so nothing collides (also lets you run one method per
instance in parallel via `INDEXINGS=<method>`).

Config (paper-matched, defaults): `sample_num 3,3`, batch 128 (= the paper's
2-GPU effective batch), lr 1e-3, `--valid_select 1` (report the best-by-
validation model — principled, avoids the overfit-final-epoch trap we hit in
branch `reproduce-ml100k-blog`), collaborative `token_size 100 / cluster 10`
(the values in `generate_dataset.sh`). Epochs 12 (peak is ~ep10; valid_select
picks the best anyway).

## The three ID schemes (what actually differs)
- **random**: `item_<random int>` — no structure; model must memorize each.
- **sequential**: `item_<appearance-order int>` — co-occurring items get nearby
  integers → shared digit-token prefixes.
- **collaborative**: `item_<CIa><CIb>…` — spectral clustering of the
  co-occurrence graph into a hierarchy; similar items share a prefix.

## Retrieval
Results land on the NFS; pull with `lambda_poll.sh` / the one-shot
`lambda_run_once.sh "true"` retrieval, then read `outputs/<method>/metrics.txt`.
