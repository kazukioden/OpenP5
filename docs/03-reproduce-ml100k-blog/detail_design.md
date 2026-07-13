# Detailed Design — `reproduce-ml100k-blog`

Built on `mac-mps-support` (MPS support + KV-cache fix). No code changes needed;
this branch is about *running* and *recording*.

## Data
`data/ML100K/user_sequence.txt`, already built by
`raw_data/build_user_sequence.py` from public GroupLens ml-100k (5-core,
timestamp order): 943 users / 1349 items / 99,287 interactions (matches README).
The T5 path derives sequential item indexing from this file on first run
(`item_sequential_indexing_original.txt`, `user_indexing.txt`, etc.).

## Command (single MPS device)
`run_repro_ml100k.sh` wraps `src/src_t5/main.py`:
```
--datasets ML100K --data_path ./data --prompt_file ./prompt.txt
--distributed 0 --gpu 0 --backbone t5-small
--tasks sequential,straightforward --item_indexing sequential
--epochs <E> --batch_size 64 --eval_batch_size 32
--sample_prompt 1 --sample_num <S> --max_his 20
--lr 1e-3 --train 1 --test_prompt seen:0 --test_before_train 0
--test_epoch <TE> --valid_select 0 --random_initialize 1
```
Env: `PYTORCH_ENABLE_MPS_FALLBACK=1`, `TOKENIZERS_PARALLELISM=false`.

- `<S>` = `3,3` faithful / `1,1` reduced (per-epoch prompt samples per task).
- `<TE>` = eval every N epochs (0 = only final). We use a small N to capture the
  metric trajectory.
- Evaluation = constrained beam search over the 1349-item Trie, num_beams=10,
  metrics Hit@5/10 + NDCG@5/10 (default `--metrics`), test prompt `seen:0`.

## What "sequential recommendation, seen" means here
- Task `sequential`: given a user's item history, predict the next item.
- `seen:0` test prompt = prompt template #0 from the "seen" set (templates the
  model trained on), matching the paper's "(seen)" column.
- The official command also trains `straightforward` jointly (multi-task); we
  keep it for faithfulness, but the reported metric of interest is the
  `sequential` task (what the blog/paper headline).

## Outputs
- stdout/log under `log/` (metric lines `hit@5: ...`, etc. per evaluated epoch).
- final checkpoint under `model/ML100K/…pt`.
- results transcribed into `test_design.md`.

## Note on the KV-cache fix
This branch includes the fix, so evaluation generation uses the decoder cache
(correct + ~1.4x faster on MPS). It does not change metrics — verified
byte-identical on branch `kv-cache-eval-speedup`.
