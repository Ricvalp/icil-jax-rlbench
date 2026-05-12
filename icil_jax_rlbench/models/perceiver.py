from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import flax.linen as nn
import jax.numpy as jnp

from .attention import CrossAttentionBlock, SelfAttentionBlock, TransformerConfig


@dataclass(frozen=True)
class PerceiverConfig:
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 2
    num_latents: int = 128
    mlp_mult: int = 4
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def tx(self) -> TransformerConfig:
        return TransformerConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            mlp_mult=self.mlp_mult,
            dropout=self.dropout,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )


class LatentPerceiver(nn.Module):
    cfg: PerceiverConfig

    @nn.compact
    def __call__(self, tokens: jnp.ndarray, *, token_mask: Optional[jnp.ndarray] = None, train: bool = False) -> jnp.ndarray:
        B = int(tokens.shape[0])
        lat = self.param('latents', nn.initializers.normal(stddev=0.02), (int(self.cfg.num_latents), int(self.cfg.d_model)), self.cfg.param_dtype)
        z = jnp.broadcast_to(lat.astype(self.cfg.dtype)[None, :, :], (B, int(self.cfg.num_latents), int(self.cfg.d_model)))
        tx = self.cfg.tx()
        for i in range(int(self.cfg.n_layers)):
            z = CrossAttentionBlock(tx, name=f'cross_{i}')(z, tokens, kv_mask=token_mask, train=train)
        return z


class SelfAttentionStack(nn.Module):
    cfg: TransformerConfig
    n_layers: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, mask: Optional[jnp.ndarray] = None, train: bool = False) -> jnp.ndarray:
        for i in range(int(self.n_layers)):
            x = SelfAttentionBlock(self.cfg, name=f'block_{i}')(x, mask=mask, train=train)
        return x
