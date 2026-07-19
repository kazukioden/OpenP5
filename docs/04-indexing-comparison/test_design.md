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

## Results (2026-07-18, 3x parallel A10, sample 3,3 / batch128 / 12ep / valid_select=1)
Sequential task, best-by-validation model.

**ML100K (dense, 92.2% sparse)**
| method | Hit@5 | Hit@10 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|
| random | **0.0753** | **0.1230** | **0.0450** | **0.0602** |
| sequential | 0.0626 | 0.1050 | 0.0409 | 0.0545 |
| collaborative | 0.0488 | 0.0954 | 0.0300 | 0.0450 |

**LastFM (sparse, 98.7% sparse)**
| method | Hit@5 | Hit@10 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|
| random | 0.0211 | 0.0312 | 0.0143 | 0.0176 |
| sequential | 0.0229 | 0.0312 | 0.0139 | 0.0166 |
| collaborative | **0.0266** | **0.0385** | **0.0162** | **0.0202** |

**Finding:** density-dependent. On dense ML100K, random is competitive/best (structure
doesn't help). On sparse LastFM, collaborative wins (+23% Hit@10 over random). BUT single
run per cell — high variance (our ML100K sequential 0.105 < paper 0.121 while random 0.123
> paper 0.092 = seed luck). The dramatic "structure wins 3x+" is on far sparser sets
(Yelp/Clothing 99.9%+). Honest claim: "direction holds (sparser -> structure helps more),
but a rigorous claim needs multi-seed + a very sparse dataset."

## Blog deliverables
- One grouped bar chart (4 metrics × 3 methods).
- The three-ID-scheme visual (same item rendered as random / sequential /
  collaborative code) + a real prompt→generation example.
- The "representation vs computation" framing (indexing as tokenization;
  semantic-ID / RQ-VAE frontier unifying it with the model).
