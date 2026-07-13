"""
Parity check: MLX T5P5 forward vs HuggingFace T5 (PyTorch), same weights + input.

Loads t5-small once via transformers, copies its weights into the MLX model,
and compares decoder logits on a fixed (input_ids, decoder_input_ids).
whole_word_ids is left off so this validates the *base T5* port.
"""
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer

from t5_mlx import T5P5


def load_mlx_from_hf(hf_model, config):
    model = T5P5(config)
    # HF state_dict (torch) -> numpy -> mx, then rename to our module layout
    sd = {k: v.detach().cpu().numpy() for k, v in hf_model.state_dict().items()}
    weights = T5P5.sanitize({k: mx.array(v) for k, v in sd.items()})
    # keep only keys the model actually has (drops tied embed_tokens / lm_head dupes)
    model_keys = {k for k, _ in tree_flatten(model.parameters())}
    kept = {k: v for k, v in weights.items() if k in model_keys}
    dropped = sorted(set(weights) - set(kept))
    model.load_weights(list(kept.items()), strict=False)
    mx.eval(model.parameters())
    return model, model_keys, kept, dropped


def main():
    name = "t5-small"
    tok = AutoTokenizer.from_pretrained(name, legacy=False)
    hf = T5ForConditionalGeneration.from_pretrained(name).eval()
    config = hf.config

    enc = tok("Considering ML100K user_1 has interacted with items 1 , 2 , 3 . "
              "What is the next recommendation ?", return_tensors="pt")
    dec_ids = torch.tensor([[config.decoder_start_token_id, 3, 8, 1150, 5]])

    with torch.no_grad():
        hf_logits = hf(input_ids=enc.input_ids,
                       decoder_input_ids=dec_ids).logits.numpy()

    model, mkeys, kept, dropped = load_mlx_from_hf(hf, config)
    print(f"model params: {len(mkeys)} | loaded: {len(kept)} | "
          f"dropped(tied/dup): {len(dropped)}")
    missing = sorted(mkeys - set(kept))
    # whole_word_embeddings is expected-missing (no pretrained weight)
    print(f"unfilled params: {missing}")

    mlx_logits = np.array(
        model(mx.array(enc.input_ids.numpy()), mx.array(dec_ids.numpy()))
    )

    a, b = hf_logits.astype(np.float64), mlx_logits.astype(np.float64)
    max_abs = np.abs(a - b).max()
    mean_abs = np.abs(a - b).mean()
    scale = np.abs(a).mean()
    # do both pick the same argmax token at every position?
    same_argmax = bool((a.argmax(-1) == b.argmax(-1)).all())
    cos = float((a.flatten() @ b.flatten()) /
                (np.linalg.norm(a) * np.linalg.norm(b)))

    print("=" * 60)
    print(f"logits shape: hf={a.shape} mlx={b.shape}")
    print(f"max_abs_diff = {max_abs:.4e}")
    print(f"mean_abs_diff= {mean_abs:.4e}  (mean|logit|={scale:.3e}, "
          f"rel={mean_abs/scale:.2e})")
    print(f"cosine       = {cos:.8f}")
    print(f"argmax match = {same_argmax}")
    ok = same_argmax and (mean_abs / scale < 1e-2)
    print("PARITY PASS" if ok else "PARITY FAIL")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
