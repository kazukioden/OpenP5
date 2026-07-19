# Detailed Design — `looped-transformer`

## Change
One knob, `loop_k`, threaded config → model. `loop_k=1` is byte-identical to the
original OpenP5 T5.

- `utils.parse_global_args`: `--loop_k` (int, default 1).
- `main.py`: `config.loop_k = args.loop_k` right after `T5Config.from_pretrained`.
- `P5_T5.forward`: replace the single decoder call with a weight-tied loop.

## The loop (in `P5_T5.forward`)
```python
loop_k = int(getattr(self.config, "loop_k", 1) or 1)
dec_use_cache = bool(use_cache) and (loop_k == 1)   # cache only valid at loop_k=1
sequence_output = None
for _i in range(loop_k):
    decoder_outputs = self.decoder(
        input_ids=(decoder_input_ids if _i == 0 else None),
        inputs_embeds=(decoder_inputs_embeds if _i == 0 else sequence_output),
        past_key_values=past_key_values,
        encoder_hidden_states=hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        use_cache=dec_use_cache, ...
    )
    sequence_output = decoder_outputs[0]
```
- **Weight-tied depth recurrence**: same `self.decoder` applied K times → more
  effective depth, **zero extra parameters**. Autograd backprops through all K
  passes.
- **Pass 0** embeds `decoder_input_ids` (the shifted labels / generated prefix).
  **Passes 1..K-1** feed the previous pass's hidden states as `inputs_embeds`
  (skip the token embedding; keep the causal mask + T5 relative position bias,
  which the decoder recomputes each pass).
- Only `forward()` is looped (used by both training and `generate()`). The dead
  `predict()` path is left untouched.

## KV cache handling
The cache assumes one decoder pass per step; per-step re-looping breaks that. So
`use_cache` is forced off when `loop_k>1` → generation recomputes the full
sequence each step (correct, just slower). `loop_k=1` keeps the fast cached path
(the `kv-cache-eval-speedup` fix). Fine here: datasets are small.

## Smoke test (passed)
`ML100K_tiny`, 1 epoch, `--loop_k 2` on MPS: training runs, and the **generation
eval goes through the looped path with cache disabled**, printing Hit/NDCG with
no error (metrics ~0 because it's a 40-user 1-epoch toy — same as the loop_k=1
toy). Confirms the train + looped-generation path is wired end to end.

## Notes / limits
- Memory grows ~K× (K× activations retained for backprop). K∈{1,2,4} on t5-small
  is fine on an A10.
- This is plain depth recurrence — no adaptive halting (ACT), no per-step scaling.
  Intentionally minimal, to isolate "more compute" as the only variable.
