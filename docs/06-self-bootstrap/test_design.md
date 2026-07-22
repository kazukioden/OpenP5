# Test Design — `self-bootstrap`

## Hypothesis
Self-generated pseudo-supervision (bootstrap/ReST) **densifies** the sparse
signal. So on sparse LastFM, `random + self-bootstrap` should move **toward**
`collaborative` — i.e., you can densify with DATA (self-training) instead of
REPRESENTATION (structured IDs).

## Experiment
`remote_selfboot_sweep.sh`, DATASET=LastFM, `{random, collaborative} × {off, on}`,
R=1 round. Config = 探求① (sample 3,3 / batch128 / 12ep / valid_select=1).
- **off** = base M0 (also the 探求① sanity check: random Hit@10 ≈ 0.031,
  collaborative ≈ 0.039).
- **on**  = M1 trained on real + pseudo (pseudo from frozen M0, top-1 NEW item
  per user, weight 0.3).

## Read
| method | off (M0) Hit@10 | on (M1) Hit@10 | Δ |
|---|---|---|---|
| random | _tbd_ (~0.031 sanity) | _tbd_ | _tbd_ |
| collaborative | _tbd_ (~0.039 sanity) | _tbd_ | _tbd_ |

Headline: does `random_on` approach `collaborative_off`? (compute-densification
substituting for representation-densification).

## Collapse canary (must log)
Self-training can homogenize (recommend popular items to all). Alongside Hit@k,
compute a **coverage/diversity** proxy from the test predictions: #distinct items
recommended across test users, and/or entropy of the recommended-item histogram.
If diversity drops sharply from off→on while Hit rises, that's collapse (a
Pyrrhic gain), and the KL-to-M0 mitigation (see overall_design) is the next lever.

## Pass / outcomes (either is a clean story)
- **random_on ↑ toward collaborative** → densification is the mechanism; data-side
  self-bootstrap recovers what structured IDs give.
- **random_on flat / collapses** → self-generated signal ≠ real structure here;
  representation still wins. Report honestly (single run; note variance).

## Sanity checks
- `off` reproduces 探求①'s LastFM numbers.
- gen_pseudo writes only NEW items (no leave-one-out leakage), pseudo rows are
  actually drawn (batch/epoch grows), loss decreases.
