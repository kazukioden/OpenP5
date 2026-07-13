# Test Design — `reproduce-ml100k-blog`

## What we are validating
That the fixed OpenP5 T5 pipeline, trained on ML100K with sequential indexing,
produces **Hit@5/10** and **NDCG@5/10** on the `sequential` task that are in the
ballpark of the OpenP5 paper (the blog's "roughly close to paper" claim).

## Reference target (paper, OpenP5-S seen, ML-100K, sequential)
| Metric | Paper |
|---|---|
| Hit@5 | 0.0678 |
| Hit@10 | 0.1208 |
| NDCG@5 | 0.0427 |
| NDCG@10 | 0.0595 |

## Acceptance
- **Sanity (must):** metrics are non-trivial and correctly ordered
  (Hit@10 > Hit@5 ≥ 0; NDCG@10 > NDCG@5 ≥ 0; well above random ~1/1349).
- **Reduced run (expected):** below paper (less training compute) but clearly
  learning — metrics rise with epochs, trending toward the target.
- **Faithful run (stretch):** within a reasonable margin of the paper numbers.

## Procedure
`./run_repro_ml100k.sh <epochs> <sample_num> <test_epoch>` on MPS. Read the
`hit@k` / `ndcg@k` lines for the `sequential` task from the log (the multi-task
run also prints `straightforward`; the headline metric is `sequential`).

## Configs
| Name | sample_num | epochs | ~time on M3 | note |
|---|---|---|---|---|
| Faithful (paper) | 3,3 | 20 | ~40 h | matches official command |
| M3-reduced | 1,1 | 10 | ~7 h | fits overnight; expect undershoot |
| Quick look | 1,1 | 5 | ~3.5 h | trajectory only |

## Results (to be filled)
Config actually run: **_tbd_** (sample_num, epochs, wall-clock).

| Metric | Paper | This run (sequential, seen) |
|---|---|---|
| Hit@5 | 0.0678 | _tbd_ |
| Hit@10 | 0.1208 | _tbd_ |
| NDCG@5 | 0.0427 | _tbd_ |
| NDCG@10 | 0.0595 | _tbd_ |

Per-epoch trajectory (if `test_epoch` > 0): _tbd_

## Notes / threats to validity
- Reduced training under-trains vs. the paper; a gap is expected and honest.
- Eval uses the KV-cache fix (correctness verified identical on branch
  `kv-cache-eval-speedup`), so it does not perturb metrics.
- Data built from public ml-100k with 5-core + timestamp order; matches the
  README statistics, but any preprocessing drift vs. the authors' exact file
  could shift numbers slightly.
