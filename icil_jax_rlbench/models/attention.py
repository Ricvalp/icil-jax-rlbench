from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class TransformerConfig:
    d_model: int = 512
    n_heads: int = 8
    mlp_mult: int = 4
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32


class MultiHeadAttention(nn.Module):
    cfg: TransformerConfig

    @nn.compact
    def __call__(
        self,
        q: jnp.ndarray,
        kv: Optional[jnp.ndarray] = None,
        *,
        kv_mask: Optional[jnp.ndarray] = None,
        train: bool = False,
        return_weights: bool = False,
    ):
        cfg = self.cfg
        kv = q if kv is None else kv
        d = int(cfg.d_model)
        h = int(cfg.n_heads)
        if d % h != 0:
            raise ValueError(f'd_model={d} must be divisible by n_heads={h}.')
        dh = d // h
        q_proj = nn.Dense(d, use_bias=False, dtype=cfg.dtype, param_dtype=cfg.param_dtype, name='q')(q)
        k_proj = nn.Dense(d, use_bias=False, dtype=cfg.dtype, param_dtype=cfg.param_dtype, name='k')(kv)
        v_proj = nn.Dense(d, use_bias=False, dtype=cfg.dtype, param_dtype=cfg.param_dtype, name='v')(kv)
        B, Tq, _ = q_proj.shape
        Tk = int(k_proj.shape[1])
        qh = q_proj.reshape(B, Tq, h, dh).transpose(0, 2, 1, 3)
        kh = k_proj.reshape(B, Tk, h, dh).transpose(0, 2, 1, 3)
        vh = v_proj.reshape(B, Tk, h, dh).transpose(0, 2, 1, 3)
        logits = jnp.einsum('bhqd,bhkd->bhqk', qh.astype(jnp.float32), kh.astype(jnp.float32)) / jnp.sqrt(jnp.asarray(dh, jnp.float32))
        if kv_mask is not None:
            mask = kv_mask.astype(jnp.bool_)[:, None, None, :]
            logits = jnp.where(mask, logits, jnp.asarray(-1e9, logits.dtype))
        weights = jax.nn.softmax(logits, axis=-1).astype(cfg.dtype)
        if float(cfg.dropout) > 0.0:
            weights = nn.Dropout(rate=float(cfg.dropout))(weights, deterministic=not train)
        out = jnp.einsum('bhqk,bhkd->bhqd', weights, vh).transpose(0, 2, 1, 3).reshape(B, Tq, d)
        out = nn.Dense(d, use_bias=False, dtype=cfg.dtype, param_dtype=cfg.param_dtype, name='out')(out)
        if return_weights:
            return out, weights
        return out


class MLP(nn.Module):
    cfg: TransformerConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool = False) -> jnp.ndarray:
        d = int(self.cfg.d_model)
        x = nn.Dense(int(self.cfg.mlp_mult) * d, dtype=self.cfg.dtype, param_dtype=self.cfg.param_dtype, name='fc1')(x)
        x = nn.gelu(x)
        if float(self.cfg.dropout) > 0.0:
            x = nn.Dropout(rate=float(self.cfg.dropout))(x, deterministic=not train)
        x = nn.Dense(d, dtype=self.cfg.dtype, param_dtype=self.cfg.param_dtype, name='fc2')(x)
        return x


class SelfAttentionBlock(nn.Module):
    cfg: TransformerConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, mask: Optional[jnp.ndarray] = None, train: bool = False) -> jnp.ndarray:
        y = nn.LayerNorm(dtype=self.cfg.dtype, param_dtype=self.cfg.param_dtype, name='ln1')(x)
        x = x + MultiHeadAttention(self.cfg, name='self_attn')(y, kv_mask=mask, train=train)
        y = nn.LayerNorm(dtype=self.cfg.dtype, param_dtype=self.cfg.param_dtype, name='ln2')(x)
        x = x + MLP(self.cfg, name='mlp')(y, train=train)
        return x


class CrossAttentionBlock(nn.Module):
    cfg: TransformerConfig

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        kv: jnp.ndarray,
        *,
        kv_mask: Optional[jnp.ndarray] = None,
        train: bool = False,
        return_weights: bool = False,
    ):
        y = nn.LayerNorm(dtype=self.cfg.dtype, param_dtype=self.cfg.param_dtype, name='ln_x')(x)
        if return_weights:
            attn_out, weights = MultiHeadAttention(self.cfg, name='cross_attn')(
                y, kv=kv, kv_mask=kv_mask, train=train, return_weights=True
            )
        else:
            attn_out = MultiHeadAttention(self.cfg, name='cross_attn')(y, kv=kv, kv_mask=kv_mask, train=train)
            weights = None
        x = x + attn_out
        y = nn.LayerNorm(dtype=self.cfg.dtype, param_dtype=self.cfg.param_dtype, name='ln_mlp')(x)
        x = x + MLP(self.cfg, name='mlp')(y, train=train)
        if return_weights:
            return x, weights
        return x
