"""
Minimal t5-small (P5 variant) in MLX for the Stage-2 PoC.

Adapted from ml-explore/mlx-examples `t5/t5.py` (Apache-2.0), trimmed to the
forward + training path and extended with P5's whole-word embedding on the
encoder input. Generation is intentionally NOT ported (PoC boundary).

Fix vs. the reference: `DenseActivation` gating now keys off the "gated-"
prefix, so original T5 (t5-small, feed_forward_proj="relu") uses a single `wi`
and matches the HF weights.
"""
import json
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple

import numpy as np
import mlx.core as mx
import mlx.nn as nn


def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    relative_buckets = 0
    if bidirectional:
        num_buckets //= 2
        relative_buckets += (relative_position > 0).astype(mx.int16) * num_buckets
        relative_position = mx.abs(relative_position)
    else:
        relative_position = -mx.minimum(relative_position, mx.zeros_like(relative_position))
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact
    scale = (num_buckets - max_exact) / np.log(max_distance / max_exact)
    relative_position_if_large = max_exact + (
        mx.log(relative_position.astype(mx.float32) / max_exact) * scale
    ).astype(mx.int16)
    relative_position_if_large = mx.minimum(relative_position_if_large, num_buckets - 1)
    relative_buckets += mx.where(is_small, relative_position, relative_position_if_large)
    return relative_buckets


class RelativePositionBias(nn.Module):
    def __init__(self, config, bidirectional: bool):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_buckets = config.relative_attention_num_buckets
        self.max_distance = getattr(config, "relative_attention_max_distance", 128)
        self.n_heads = config.num_heads
        self.embeddings = nn.Embedding(config.relative_attention_num_buckets, config.num_heads)

    def __call__(self, query_length: int, key_length: int, offset: int = 0):
        context_position = mx.arange(offset, query_length)[:, None]
        memory_position = mx.arange(key_length)[None, :]
        relative_position = memory_position - context_position
        rp_bucket = _relative_position_bucket(
            relative_position, self.bidirectional, self.num_buckets, self.max_distance
        )
        values = self.embeddings(rp_bucket)          # (q, k, heads)
        return values.transpose(2, 0, 1)             # (heads, q, k)


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        inner_dim = config.d_kv * config.num_heads
        self.num_heads = config.num_heads
        self.query_proj = nn.Linear(config.d_model, inner_dim, bias=False)
        self.key_proj = nn.Linear(config.d_model, inner_dim, bias=False)
        self.value_proj = nn.Linear(config.d_model, inner_dim, bias=False)
        self.out_proj = nn.Linear(inner_dim, config.d_model, bias=False)

    def __call__(self, queries, keys, values, mask=None, cache=None):
        queries = self.query_proj(queries)
        keys = self.key_proj(keys)
        values = self.value_proj(values)
        num_heads = self.num_heads
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, S, num_heads, -1).transpose(0, 2, 3, 1)
        values = values.reshape(B, S, num_heads, -1).transpose(0, 2, 1, 3)
        if cache is not None:
            key_cache, value_cache = cache
            keys = mx.concatenate([key_cache, keys], axis=3)
            values = mx.concatenate([value_cache, values], axis=2)
        # NOTE: T5 does NOT scale attention scores by 1/sqrt(d).
        scores = queries @ keys
        if mask is not None:
            scores = scores + mask.astype(scores.dtype)
        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        values_hat = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(values_hat), (keys, values)


class DenseActivation(nn.Module):
    def __init__(self, config):
        super().__init__()
        mlp_dims = config.d_ff or config.d_model * 4
        ff = getattr(config, "feed_forward_proj", "relu")
        self.gated = ff.startswith("gated-")          # FIX: original T5 (relu) is NOT gated
        activation = ff.removeprefix("gated-")
        if self.gated:
            self.wi_0 = nn.Linear(config.d_model, mlp_dims, bias=False)
            self.wi_1 = nn.Linear(config.d_model, mlp_dims, bias=False)
        else:
            self.wi = nn.Linear(config.d_model, mlp_dims, bias=False)
        self.wo = nn.Linear(mlp_dims, config.d_model, bias=False)
        self.act = {"relu": nn.relu, "gelu": nn.gelu, "silu": nn.silu}[activation]

    def __call__(self, x):
        if self.gated:
            x = self.act(self.wi_0(x)) * self.wi_1(x)
        else:
            x = self.act(self.wi(x))
        return self.wo(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.ln1 = nn.RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.ln2 = nn.RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dense = DenseActivation(config)

    def __call__(self, x, mask):
        y = self.ln1(x)
        y, _ = self.attention(y, y, y, mask=mask)
        x = x + y
        y = self.ln2(x)
        y = self.dense(y)
        return x + y


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = [TransformerEncoderLayer(config) for _ in range(config.num_layers)]
        self.ln = nn.RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.relative_attention_bias = RelativePositionBias(config, bidirectional=True)

    def __call__(self, x, mask=None):
        pos_bias = self.relative_attention_bias(x.shape[1], x.shape[1])
        if mask is not None:
            pos_bias = pos_bias + mask
        for layer in self.layers:
            x = layer(x, mask=pos_bias)
        return self.ln(x)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attention = MultiHeadAttention(config)
        self.cross_attention = MultiHeadAttention(config)
        self.ln1 = nn.RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.ln2 = nn.RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.ln3 = nn.RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dense = DenseActivation(config)

    def __call__(self, x, memory, mask, memory_mask, cache=None):
        y = self.ln1(x)
        y, cache = self.self_attention(y, y, y, mask, cache)
        x = x + y
        y = self.ln2(x)
        y, _ = self.cross_attention(y, memory, memory, memory_mask)
        x = x + y
        y = self.ln3(x)
        y = self.dense(y)
        return x + y, cache


class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_layers = getattr(config, "num_decoder_layers", config.num_layers)
        self.layers = [TransformerDecoderLayer(config) for _ in range(n_layers)]
        self.ln = nn.RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.relative_attention_bias = RelativePositionBias(config, bidirectional=False)

    def __call__(self, x, memory, mask, memory_mask, cache=None):
        if cache is not None:
            offset = cache[0][0].shape[3]
        else:
            offset = 0
            cache = [None] * len(self.layers)
        T = offset + x.shape[1]
        pos_bias = self.relative_attention_bias(T, T, offset=offset)
        mask = pos_bias if mask is None else (mask + pos_bias)
        for e, layer in enumerate(self.layers):
            x, cache[e] = layer(x, memory, mask, memory_mask, cache=cache[e])
        return self.ln(x), cache


class T5P5(nn.Module):
    """t5-small with P5's additive whole-word embedding on the encoder input."""

    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.whole_word_embeddings = nn.Embedding(512, config.d_model)  # P5 addition
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)
        self.tie_word_embeddings = getattr(config, "tie_word_embeddings", True)
        if not self.tie_word_embeddings:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.model_dim = config.d_model

    def encode(self, inputs, whole_word_ids=None):
        h = self.wte(inputs)
        if whole_word_ids is not None:
            h = h + self.whole_word_embeddings(whole_word_ids)
        return self.encoder(h)

    def decode(self, inputs, memory, cache=None):
        h = self.wte(inputs)
        T = h.shape[1]
        if T > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(T).astype(h.dtype)
        else:
            mask = None
        y, cache = self.decoder(h, memory=memory, mask=mask, memory_mask=None, cache=cache)
        if not self.tie_word_embeddings:
            y = self.lm_head(y)
        else:
            y = (y * self.model_dim ** -0.5) @ self.wte.weight.T
        return y, cache

    def __call__(self, input_ids, decoder_input_ids, whole_word_ids=None):
        memory = self.encode(input_ids, whole_word_ids)
        return self.decode(decoder_input_ids, memory)[0]

    # --- weight loading (HF t5-small -> this module) ---
    @classmethod
    def sanitize(cls, weights):
        shared = [
            (".block.", ".layers."),
            (".k.", ".key_proj."), (".o.", ".out_proj."),
            (".q.", ".query_proj."), (".v.", ".value_proj."),
            ("shared.", "wte."), ("lm_head.", "lm_head."),
            (".layer.0.layer_norm.", ".ln1."),
            (".layer.1.layer_norm.", ".ln2."),
            (".layer.2.layer_norm.", ".ln3."),
            (".final_layer_norm.", ".ln."),
            ("layers.0.layer.0.SelfAttention.relative_attention_bias.",
             "relative_attention_bias.embeddings."),
        ]
        enc = [(".layer.0.SelfAttention.", ".attention."),
               (".layer.1.DenseReluDense.", ".dense.")]
        dec = [(".layer.0.SelfAttention.", ".self_attention."),
               (".layer.1.EncDecAttention.", ".cross_attention."),
               (".layer.2.DenseReluDense.", ".dense.")]
        ignored = ["decoder.layers.0.cross_attention.relative_attention_bias.weight"]

        def rk(key):
            for o, n in shared:
                key = key.replace(o, n)
            if key.startswith("encoder."):
                for o, n in enc:
                    key = key.replace(o, n)
            elif key.startswith("decoder."):
                for o, n in dec:
                    key = key.replace(o, n)
            return key

        weights = {rk(k): v for k, v in weights.items()}
        for k in ignored:
            weights.pop(k, None)
        return weights

    @classmethod
    def from_pretrained(cls, path_or_repo="t5-small", dtype=mx.float32):
        from huggingface_hub import snapshot_download
        p = Path(path_or_repo)
        if not p.exists():
            p = Path(snapshot_download(repo_id=path_or_repo,
                                       allow_patterns=["*.json", "*.safetensors", "*.model"]))
        with open(p / "config.json") as f:
            config = SimpleNamespace(**json.load(f))
        model = cls(config)
        weights = cls.sanitize(mx.load(str(p / "model.safetensors")))
        weights = {k: v.astype(dtype) for k, v in weights.items()}
        # whole_word_embeddings has no pretrained weight -> keep as-is (we zero it
        # in the parity check; P5 would train it).
        model.load_weights(list(weights.items()), strict=False)
        mx.eval(model.parameters())
        return model, config
