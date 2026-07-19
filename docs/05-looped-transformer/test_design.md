# Test Design — `looped-transformer`

## Hypothesis (representation vs computation)
Does **more computation (decoder looping)** substitute for **item-ID structure
(representation)**? Concretely: as `loop_k` grows, does the gap between **random**
(bad IDs) and **collaborative** (structured IDs) shrink?

- **shrinks** → computation can compensate for bad representation.
- **stays** → representation is king; looping doesn't rescue bad IDs.

## Grid
`{random, collaborative} × {loop_k = 1, 2, 4}` = 6 runs, on **LastFM** (sparse —
where 探求① showed indexing matters: collaborative > random). Same training
config as 探求①: `sample_num 3,3` / batch 128 / 12 epochs / `valid_select=1`,
T5-small.

Optional second pass on **ML100K** (dense — where random won) to see if looping
changes that surprising result.

## Sanity checks
- **`loop_k=1` reproduces 探求①'s LastFM numbers** (random ≈ 0.031, collaborative
  ≈ 0.039 Hit@10). If not, something in the loop wiring is off.
- Loss decreases; generation eval completes (cache auto-disabled at loop_k>1).

## What we read
Hit@10 (sequential task, best-by-val) as a function of `loop_k`, per indexing.
The headline is the **random↔collaborative gap vs loop_k**:

| loop_k | random Hit@10 | collab Hit@10 | gap |
|---|---|---|---|
| 1 | _tbd_ | _tbd_ | _tbd_ |
| 2 | _tbd_ | _tbd_ | _tbd_ |
| 4 | _tbd_ | _tbd_ | _tbd_ |

## Caveats (same honesty as 探求①)
- Single run per cell → variance; the *trend across loop_k* matters more than any
  one number. If ambiguous, a couple of seeds on the key cells.
- Looping adds compute but also changes optimization dynamics (deeper effective
  net at fixed LR/epochs); a fair reading compares the *gap*, not absolute lift.
- `loop_k>1` eval is uncached (slower) but numerically correct.

## Blog payoff
Either outcome is a clean 探求② story: "計算で表現を代替できるか" answered with a
single curve (gap vs loop_k), directly continuing 探求①.
