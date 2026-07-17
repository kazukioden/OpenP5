# Overall Design — `indexing-comparison`

Branched from `reproduce-ml100k-blog` (inherits the MPS/KV fixes + the Lambda
detached-run infra + the working ML100K pipeline).

## Goal / blog angle
The reproduction confirmed the pipeline works. This branch produces the
**central insight for a tech blog on generative recommendation**:

> In LLM-based recommendation the item is emitted as a *token sequence*, so
> **how you assign item IDs (the "tokenization" of items) matters more than
> model tweaks.** Random IDs ≈ memorization; structured IDs (sequential /
> collaborative) let the model share structure and generalize.

The OpenP5 paper (Tables 1-2) shows this starkly on ML100K: random indexing is
~3x worse than sequential/collaborative on Hit@10. We reproduce that head-to-head
on one clean plot, plus a qualitative demo of the three ID schemes.

## What we run
Same model/data/training, only `--item_indexing` varies:
- **random** — arbitrary integer IDs (`item_734`). No structure.
- **sequential** — IDs by order of appearance in user histories (`item_1035`);
  co-occurring items get nearby IDs / shared digit-token prefixes.
- **collaborative** — spectral-clustering hierarchical codes
  (`item_<CI0><CI7><CI3>…`); similar items share a prefix (a semantic-ID-like
  tree).

## Deliverables
1. A fair 3-way bar chart (Hit@5/10, NDCG@5/10) on ML100K sequential-rec.
2. A qualitative panel: the same item under the 3 schemes + a real
   prompt→generation example from a trained model.
3. The framing: **representation (indexing) vs computation (architecture)** —
   structure moved into the token space, and how the frontier (learned
   "semantic IDs" / RQ-VAE) unifies indexing with the model.

## Prerequisite fix (in this branch)
The non-distributed `collaborative` path was broken: `MultiTaskDataset` passed
an undefined `self.collaborative_sparse` to `collaborative_indexing()` (wrong
arity). Fixed to match the function signature; verified it now indexes + trains
(items become `<CI…>` hierarchical codes). See `detail_design.md`.

## Scope
In: the 3 indexing runs on ML100K (+ the fix, qualitative demo, plot).
Out: other datasets/backbones; changing the model; upstreaming the fix (can be
cherry-picked later).
