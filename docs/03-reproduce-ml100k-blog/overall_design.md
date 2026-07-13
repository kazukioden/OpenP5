# Overall Design — `reproduce-ml100k-blog`

> This is the **original goal** of the whole effort: reproduce the OpenP5
> verification from the Wantedly blog. All prior branches (MPS support, KV fix,
> MLX PoC) were enablers/side-quests to make this runnable on the M3.

## Target
Blog: https://www.wantedly.com/companies/wantedly/post_articles/870149

The post runs OpenP5 (v2.0) on **MovieLens-100K**, **sequential recommendation**,
**T5** backbone, and reports **Hit@5/10** and **NDCG@5/10**, concluding the
output is "roughly close to the paper." It links three Colab notebooks (dataset
build / train / infer). It gives an example prompt→output pair
(`What is the top recommendation for ML100K user_4 ?` → `ML100K item_1081`) but
**no numeric table**. So our reference for "close to paper" is the OpenP5 paper.

## Reference (OpenP5 paper, `OpenP5_more_results.pdf`, Tables 1–2)
Our exact config = single dataset + **sequential indexing** + `seen` test prompt
= **OpenP5-S (seen)**, ML-100K, sequential task:

| Metric | Paper (OpenP5-S, seen) |
|---|---|
| Hit@5 | 0.0678 |
| Hit@10 | 0.1208 |
| NDCG@5 | 0.0427 |
| NDCG@10 | 0.0595 |

(For reference, random indexing OpenP5-R is much lower, collaborative OpenP5-C
is in between; we reproduce the sequential one the blog used.)

## Config
Official ML100K command (`command/command_t5/ML100K_sequential.sh`), adapted to
single MPS device:
`--datasets ML100K --item_indexing sequential --tasks sequential,straightforward
--backbone t5-small --epochs 20 --batch_size 64 --sample_prompt 1 --sample_num 3,3
--max_his 20 --lr 1e-3 --test_prompt seen:0`.

## The M3 reality (why this needs a scale decision)
Training is MPS-bound (~1.3 it/s). The faithful config is ~9044 batches/epoch
≈ 2 h/epoch → **20 epochs ≈ 40 h**. The KV fix (branch `kv-cache-eval-speedup`)
speeds *evaluation* ~1.4x but does nothing for training. So there is a genuine
time/fidelity tradeoff, recorded in `test_design.md`:
- **Faithful** (sample 3,3, 20 ep): approaches the paper numbers, ~40 h on M3.
- **M3-reduced** (sample 1,1, fewer ep): finishes in hours but will *under*shoot
  the paper (less training compute), used to show the pipeline produces sensible,
  trending-correct numbers.
- **Cloud GPU**: fastest + most faithful; not this machine.

## Success criteria
- **Primary:** the pipeline yields non-trivial Hit@k / NDCG@k on ML100K
  sequential, in the right ballpark and ordering (Hit@10 > Hit@5, etc.).
- **Stretch:** with enough training, land near the paper's OpenP5-S numbers.
- Honest reporting of the exact config + epochs actually run vs. the paper.

## Scope
In: run ML100K sequential (+straightforward, as the official command does) on
the fixed pipeline; record metrics vs. paper. Out: other datasets; other
indexings; LLaMA; changing the model/metrics.
