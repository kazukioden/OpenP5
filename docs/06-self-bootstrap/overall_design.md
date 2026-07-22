# Overall Design — `self-bootstrap`

探求②の本命。Branched from `indexing-comparison`.

## Hypothesis (the through-line: densify the sparse signal)
探求① found: structured item IDs help **sparse** data by injecting a similarity
prior = **effectively densifying supervision**. On dense data (enough signal)
it's redundant.

The user's framing: *a method that can be seen as **virtually providing a dense
signal** should help sparse data.* The RL-flavored answer is **bootstrapping**
(TD-style: learn from your own predictions) — its LLM analog is **self-training /
ReST** (generate → reward-filter → fine-tune). The model **manufactures extra
supervision from its own predictions**, filling in where observations are absent.

> **Q: Does self-generated pseudo-supervision (self-bootstrap) let random-ID
> models catch up to collaborative on sparse data? I.e., can you densify with
> DATA (self-training) instead of REPRESENTATION (structured IDs)?**

## Method: ReST-style self-bootstrap for generative recommendation
Rounds, each = Grow → Filter → Improve (Gulcehre et al. 2023, ReST):

1. **Round 0 — base:** train `M0` normally (supervised).
2. **Grow:** with the frozen `M_{r-1}` (the "target network", for stability),
   generate top-k next-item candidates per **training** user (constrained beam
   search over the Trie), keeping each candidate's score/confidence.
3. **Filter (reward):** keep a pseudo-label only if it passes a reward proxy —
   confidence ≥ threshold **and/or** a consistency check (the item co-occurs
   with / is similar to the user's real history). Prevents garbage.
4. **Improve:** add the kept `(user history → pseudo-item)` pairs as extra
   training examples at **low weight** λ, and fine-tune → `M_r`.
5. Repeat for a few rounds; evaluate `M_final` on the held-out test.

## The central risk: self-reinforcement collapse
Self-training amplifies the model's own biases (recommend popular items to
everyone → homogenize). Mitigations, all in the design:
- **Reward/filter** (confidence + history consistency + a diversity cap).
- **Frozen target model** for generation (decouple target from the learner — the
  RL target-network trick).
- **Low pseudo weight** λ so real labels dominate; cap #pseudo per user.
- Watch a diversity/coverage metric across rounds, not just Hit@k.

## Eval-protocol care
Pseudo-labels augment **training only**. The leave-one-out test/validation items
(last / second-to-last of each user sequence) are never touched, never used to
generate pseudo-labels. No test leakage.

## Experiment (see test_design)
Sparse **LastFM**: `{random, collaborative} × {self-bootstrap off / on}`.
Does `random + self-bootstrap` close the gap to `collaborative`? Track Hit@10
**and** a diversity/coverage metric per round (to catch collapse).

## Scope
In: the Grow/Filter/Improve loop (pseudo-label generation + weighted-augmented
retraining + a frozen generator), on OpenP5 T5. Out: learned reward models,
MCTS/tree-search variants, online RL.
