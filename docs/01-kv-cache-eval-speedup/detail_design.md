# Detailed Design — `kv-cache-eval-speedup`

## The bug

`src/src_t5/model/P5_T5.py`:

```python
def prepare_inputs_for_generation(
    self,
    input_ids,
    past=None,                 # <-- wrong keyword
    attention_mask=None,
    use_cache=None,
    encoder_outputs=None,
    **kwargs,
):
    if past is not None:
        input_ids = input_ids[:, -1:]
    output = {
        "decoder_input_ids": input_ids,
        "past_key_values": past,   # <-- always None
        "encoder_outputs": encoder_outputs,
        "attention_mask": attention_mask,
        "use_cache": use_cache,
    }
    return output
```

## Why it disables the cache (transformers 4.26)

Verified in the pinned install
(`.venv-p5/.../transformers/generation/utils.py`):

- After each decode step, `_update_model_kwargs_for_generation` stores the
  decoder cache as **`model_kwargs["past_key_values"]`** (line ~707), read from
  `outputs.past_key_values`.
- Generation then calls `self.prepare_inputs_for_generation(input_ids, **model_kwargs)`.
  So the cache arrives as the keyword **`past_key_values`**.
- Our override has no `past_key_values` parameter, so the cache lands in
  `**kwargs` and is dropped. `past` stays `None`, meaning:
  - `input_ids` is never truncated to the last token, and
  - the returned dict sets `"past_key_values": None`.
- Result: every step re-runs the whole decoder from scratch. Output is still
  correct (full recompute), but generation is O(L²). Beam search also wastes
  work reordering a cache that is never consumed.

## The fix

Rename the parameter to `past_key_values` and use it:

```python
def prepare_inputs_for_generation(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    use_cache=None,
    encoder_outputs=None,
    **kwargs,
):
    if past_key_values is not None:
        input_ids = input_ids[:, -1:]
    output = {
        "decoder_input_ids": input_ids,
        "past_key_values": past_key_values,
        "encoder_outputs": encoder_outputs,
        "attention_mask": attention_mask,
        "use_cache": use_cache,
    }
    return output
```

## Why this is safe

- `P5_T5.forward` already accepts `past_key_values` and forwards it to
  `self.decoder(..., past_key_values=past_key_values, ...)`; the T5 decoder
  stack implements incremental decoding. We are only reconnecting an input that
  was being dropped.
- `_reorder_cache` is inherited from `T5ForConditionalGeneration` and already
  runs in beam search; it now operates on a cache that is actually consumed.
- Cached vs. uncached autoregressive decoding is mathematically equivalent, so
  greedy/beam outputs are unchanged (validated by the exact-match test).
- No change to the encoder, whole-word embeddings, weights, or the constrained
  `prefix_allowed_tokens_fn` Trie logic.

## Affected paths
- `SingleRunner.test_dataset_task` / `_filtered` / `_filtered_batch` — all call
  `self.model.generate(...)`; they benefit automatically, no change needed.
- Training (`forward` with `labels`) does not use `prepare_inputs_for_generation`
  and is unaffected.
