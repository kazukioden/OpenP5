# Test Design — `indexing-comparison`

## Hypothesis
On ML100K sequential recommendation, item-ID structure dominates: **random ≪
sequential ≈ collaborative**. The paper (Tables 1-2) shows random ~3x worse on
Hit@10. We test whether our pipeline reproduces that ordering under an otherwise
identical config.

## Method
Run `remote_indexing_sweep.sh` (random / sequential / collaborative), identical
config, `--valid_select 1` so each method reports its best-by-validation model
(one clean number per method — fair for a bar chart). Read
`outputs/<method>/metrics.txt` for the `sequential`-task Hit@5/10 + NDCG@5/10.

## Pass criteria
- **Ordering (main claim):** sequential and collaborative both clearly beat
  random (expect random Hit@10 roughly 1/2–1/3 of the others).
- **Sanity:** collaborative runs at all (the fix) and produces `<CI…>` IDs.
- Absolute values in the paper's ballpark (sequential/collaborative Hit@10
  ~0.10–0.12; random ~0.03–0.05).

## Paper reference (OpenP5, seen, ML-100K, sequential task)
| method | Hit@5 | Hit@10 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|
| random (R) | 0.0551 | 0.0922 | 0.0352 | 0.0470 |
| sequential (S) | 0.0678 | 0.1208 | 0.0427 | 0.0595 |
| collaborative (C) | 0.0530 | 0.1006 | 0.0331 | 0.0481 |

(Note: on ML-100K the paper's random is closer to the others than on sparser
datasets; the ~3x gap is most dramatic on large sparse sets. Our plot will show
ML100K honestly and can cite the cross-dataset trend.)

## Results (to fill)
| method | Hit@5 | Hit@10 | NDCG@5 | NDCG@10 | notes |
|---|---|---|---|---|---|
| random | _tbd_ | _tbd_ | _tbd_ | _tbd_ | |
| sequential | _tbd_ | _tbd_ | _tbd_ | _tbd_ | |
| collaborative | _tbd_ | _tbd_ | _tbd_ | _tbd_ | |

## Blog deliverables
- One grouped bar chart (4 metrics × 3 methods).
- The three-ID-scheme visual (same item rendered as random / sequential /
  collaborative code) + a real prompt→generation example.
- The "representation vs computation" framing (indexing as tokenization;
  semantic-ID / RQ-VAE frontier unifying it with the model).
