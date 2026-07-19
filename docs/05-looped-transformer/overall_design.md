# Overall Design — `looped-transformer`

探求②。Branched from `indexing-comparison` (inherits the indexing sweep harness,
datasets, collaborative fix, and the Lambda detached-run infra).

## The question (the through-line: representation vs computation)
探求① showed **item-ID structure (representation) matters** — collaborative wins
on sparse data. Now the flip side:

> **モデルに"考える回数"を増やしたら（looped transformer）、雑な random ID でも
> 構造化 ID に追いつけるのか？ ＝ 表現(ID) と 計算(反復) は代替可能か？**

- **縮む（loopでrandom↔collabの差が消える）** → 計算は表現を代替できる
- **縮まない** → 表現が王様（IDの筋が悪いと考えても無駄）

どちらに転んでもブログのオチが立つ。

## What "looped" means here
**Weight-tied depth recurrence on the decoder** (universal/looped transformer):
run the *same* decoder stack K times, feeding each pass's output hidden states
back as the next pass's input embeddings. **No extra parameters — only more
effective depth (compute).** That's the clean way to isolate "computation" from
"representation" and from "parameter count".

```
pass 0: h = decoder(decoder_input_ids, memory)          # normal
pass i: h = decoder(inputs_embeds=h,   memory)          # refine, weight-tied
logits = lm_head(h * d_model**-0.5)
```

- Loop the **decoder** (the generation side), not the encoder — the natural place
  for "thinking longer" about the output. (Encoder loop = a later variant.)
- `loop_k` is a config/CLI knob; `loop_k=1` == the original OpenP5 model exactly.

## Known tradeoff: KV cache
The KV cache (branch `kv-cache-eval-speedup`) assumes one decoder pass per step.
With `loop_k>1`, per-step re-looping makes incremental caching inconsistent. For
the first experiment we **disable the cache during generation when `loop_k>1`**
(recompute each step). Eval is slower but correct; datasets here are small so it
is tolerable. `loop_k=1` keeps the fast cached path.

## Scope
In: a `loop_k` knob on `P5_T5` decoder; CLI wiring; a smoke test; the grid runs.
Out: encoder looping; adaptive halting (ACT); learned per-step weights; the
boosting thread (separate branch if pursued).

## Experiment grid (see test_design)
`{random, collaborative} × {loop_k = 1, 2, 4}` on the dataset where indexing
matters most (LastFM, sparse) — does looping let random catch up to collaborative?
Optionally ML100K too. Same training config as 探求①.

## Success criteria
Not "beat SOTA" — it's a **mechanism probe**. Success = a clean, interpretable
answer to "does added compute substitute for ID structure?", with the loop_k=1
column reproducing 探求①'s numbers (sanity).
