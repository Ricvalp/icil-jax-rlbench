from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

import flax.linen as nn
import jax.numpy as jnp

from .attention import CrossAttentionBlock, SelfAttentionBlock, TransformerConfig
from .encoders import ContextEncoder, EncoderConfig


@dataclass(frozen=True)
class DecoderConfig:
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    context_mode: str = 'single_ctx'  # single_ctx | two_ctx
    mlp_mult: int = 4
    dropout: float = 0.0

    def tx(self) -> TransformerConfig:
        return TransformerConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            mlp_mult=self.mlp_mult,
            dropout=self.dropout,
        )


@dataclass(frozen=True)
class PolicyConfig:
    encoder: EncoderConfig = EncoderConfig()
    decoder: DecoderConfig = DecoderConfig()
    H: int = 16

    def with_model_dim(self) -> 'PolicyConfig':
        return replace(self, decoder=replace(self.decoder, d_model=self.encoder.d_model, n_heads=self.encoder.n_heads))


class ActionDecoder(nn.Module):
    cfg: DecoderConfig
    H: int
    action_dim: int

    @nn.compact
    def __call__(
        self,
        query_tokens: jnp.ndarray,
        support_tokens: Optional[jnp.ndarray],
        *,
        query_mask: Optional[jnp.ndarray] = None,
        support_mask: Optional[jnp.ndarray] = None,
        train: bool = False,
        return_attn_stats: bool = False,
    ):
        d = int(self.cfg.d_model)
        B = int(query_tokens.shape[0])
        action_queries = self.param('action_queries', nn.initializers.normal(stddev=0.02), (int(self.H), d))
        x = jnp.broadcast_to(action_queries[None, :, :], (B, int(self.H), d))
        tx = self.cfg.tx()
        entropies = []
        max_probs = []
        raw_max_probs = []
        masses = []
        query_entropies = []
        mode = str(self.cfg.context_mode)
        if mode == 'single_ctx':
            query_len = int(query_tokens.shape[1])
            if support_tokens is None:
                context = query_tokens
                context_mask = query_mask
            else:
                context = jnp.concatenate([query_tokens, support_tokens], axis=1)
                if query_mask is None and support_mask is None:
                    context_mask = None
                else:
                    qmask = jnp.ones(query_tokens.shape[:2], dtype=jnp.bool_) if query_mask is None else query_mask
                    smask = jnp.ones(support_tokens.shape[:2], dtype=jnp.bool_) if support_mask is None else support_mask
                    context_mask = jnp.concatenate([qmask, smask], axis=1)
            for i in range(int(self.cfg.n_layers)):
                x = SelfAttentionBlock(tx, name=f'self_{i}')(x, train=train)
                if return_attn_stats:
                    x, weights = CrossAttentionBlock(tx, name=f'cross_{i}')(
                        x, context, kv_mask=context_mask, train=train, return_weights=True
                    )
                    query_stats = self._query_attention_stats(weights, query_len=query_len, query_mask=query_mask)
                    query_entropies.append(query_stats['attn_query_entropy'])
                    if support_tokens is None:
                        zero = jnp.asarray(0.0, dtype=jnp.float32)
                        entropies.append(zero)
                        max_probs.append(zero)
                        raw_max_probs.append(zero)
                        masses.append(zero)
                    else:
                        stats = self._memory_attention_stats(weights, query_len=query_len, support_mask=support_mask)
                        entropies.append(stats['attn_memory_entropy'])
                        max_probs.append(stats['attn_memory_max'])
                        raw_max_probs.append(stats['attn_memory_raw_max'])
                        masses.append(stats['attn_memory_mass'])
                else:
                    x = CrossAttentionBlock(tx, name=f'cross_{i}')(x, context, kv_mask=context_mask, train=train)
        elif mode == 'two_ctx':
            for i in range(int(self.cfg.n_layers)):
                x = SelfAttentionBlock(tx, name=f'self_{i}')(x, train=train)
                if return_attn_stats:
                    x, query_weights = CrossAttentionBlock(tx, name=f'query_cross_{i}')(
                        x, query_tokens, kv_mask=query_mask, train=train, return_weights=True
                    )
                    query_stats = self._context_attention_stats(query_weights, query_mask)
                    query_entropies.append(query_stats['attn_entropy'])
                    if support_tokens is None:
                        zero = jnp.asarray(0.0, dtype=jnp.float32)
                        entropies.append(zero)
                        max_probs.append(zero)
                        raw_max_probs.append(zero)
                        masses.append(zero)
                    else:
                        x, memory_weights = CrossAttentionBlock(tx, name=f'support_cross_{i}')(
                            x, support_tokens, kv_mask=support_mask, train=train, return_weights=True
                        )
                        stats = self._context_attention_stats(memory_weights, support_mask)
                        entropies.append(stats['attn_entropy'])
                        max_probs.append(stats['attn_max'])
                        raw_max_probs.append(stats['attn_raw_max'])
                        masses.append(stats['attn_mass'])
                else:
                    x = CrossAttentionBlock(tx, name=f'query_cross_{i}')(
                        x, query_tokens, kv_mask=query_mask, train=train
                    )
                    if support_tokens is not None:
                        x = CrossAttentionBlock(tx, name=f'support_cross_{i}')(
                            x, support_tokens, kv_mask=support_mask, train=train
                        )
        else:
            raise ValueError("decoder.context_mode must be 'single_ctx' or 'two_ctx'.")
        x = nn.LayerNorm(name='out_ln')(x)
        pred = nn.Dense(int(self.action_dim), name='action_head')(x)
        if return_attn_stats:
            stats = {
                'attn_memory_entropy': jnp.mean(jnp.stack(entropies)),
                'attn_memory_max': jnp.mean(jnp.stack(max_probs)),
                'attn_memory_raw_max': jnp.mean(jnp.stack(raw_max_probs)),
                'attn_memory_mass': jnp.mean(jnp.stack(masses)),
                'attn_query_entropy': jnp.mean(jnp.stack(query_entropies)),
            }
            return pred, stats
        return pred

    def _context_attention_stats(self, weights: jnp.ndarray, mask: Optional[jnp.ndarray]):
        # weights=[B, heads, action_queries, context_tokens].
        context_w = weights.astype(jnp.float32)
        if mask is not None:
            cmask = mask.astype(jnp.bool_)[:, None, None, :]
            context_w = jnp.where(cmask, context_w, 0.0)
            valid_count = jnp.sum(mask.astype(jnp.float32), axis=-1)
        else:
            valid_count = jnp.full((weights.shape[0],), weights.shape[-1], dtype=jnp.float32)
        mass = jnp.sum(context_w, axis=-1)
        context_dist = context_w / (mass[..., None] + 1e-8)
        entropy = -jnp.sum(jnp.where(context_dist > 0.0, context_dist * jnp.log(context_dist + 1e-8), 0.0), axis=-1)
        norm = jnp.log(jnp.maximum(valid_count, 2.0))[:, None, None]
        entropy = jnp.where(mass > 1e-8, entropy / norm, 0.0)
        return {
            'attn_entropy': jnp.mean(entropy),
            'attn_max': jnp.mean(jnp.max(context_dist, axis=-1)),
            'attn_raw_max': jnp.mean(jnp.max(context_w, axis=-1)),
            'attn_mass': jnp.mean(mass),
        }

    def _query_attention_stats(self, weights: jnp.ndarray, *, query_len: int, query_mask: Optional[jnp.ndarray]):
        # weights=[B, heads, action_queries, query_tokens + support_tokens].
        query_w = weights[..., : int(query_len)].astype(jnp.float32)
        if query_mask is not None:
            qmask = query_mask.astype(jnp.bool_)[:, None, None, :]
            query_w = jnp.where(qmask, query_w, 0.0)
            valid_count = jnp.sum(query_mask.astype(jnp.float32), axis=-1)
        else:
            valid_count = jnp.full((weights.shape[0],), int(query_len), dtype=jnp.float32)
        mass = jnp.sum(query_w, axis=-1)
        query_dist = query_w / (mass[..., None] + 1e-8)
        entropy = -jnp.sum(jnp.where(query_dist > 0.0, query_dist * jnp.log(query_dist + 1e-8), 0.0), axis=-1)
        norm = jnp.log(jnp.maximum(valid_count, 2.0))[:, None, None]
        entropy = jnp.where(mass > 1e-8, entropy / norm, 0.0)
        return {'attn_query_entropy': jnp.mean(entropy)}

    def _memory_attention_stats(self, weights: jnp.ndarray, *, query_len: int, support_mask: Optional[jnp.ndarray]):
        # weights=[B, heads, action_queries, query_tokens + support_tokens].
        mem_w = weights[..., int(query_len):].astype(jnp.float32)
        if support_mask is not None:
            smask = support_mask.astype(jnp.bool_)[:, None, None, :]
            mem_w = jnp.where(smask, mem_w, 0.0)
            valid_count = jnp.sum(support_mask.astype(jnp.float32), axis=-1)
        else:
            valid_count = jnp.full((weights.shape[0],), mem_w.shape[-1], dtype=jnp.float32)
        mass = jnp.sum(mem_w, axis=-1)
        mem_dist = mem_w / (mass[..., None] + 1e-8)
        entropy = -jnp.sum(jnp.where(mem_dist > 0.0, mem_dist * jnp.log(mem_dist + 1e-8), 0.0), axis=-1)
        norm = jnp.log(jnp.maximum(valid_count, 2.0))[:, None, None]
        entropy = jnp.where(mass > 1e-8, entropy / norm, 0.0)
        return {
            'attn_memory_entropy': jnp.mean(entropy),
            'attn_memory_max': jnp.mean(jnp.max(mem_dist, axis=-1)),
            'attn_memory_raw_max': jnp.mean(jnp.max(mem_w, axis=-1)),
            'attn_memory_mass': jnp.mean(mass),
        }


class DirectRegressionPolicy(nn.Module):
    cfg: PolicyConfig
    state_dim: int
    action_dim: int

    def setup(self) -> None:
        self.encoder = ContextEncoder(self.cfg.encoder, self.state_dim, self.action_dim, name='encoder')
        self.decoder = ActionDecoder(self.cfg.decoder, int(self.cfg.H), self.action_dim, name='decoder')

    def __call__(self, batch: Dict[str, jnp.ndarray], *, train: bool = False, return_attn_stats: bool = False):
        support_tokens, support_mask = self.encode_support(batch, train=train)
        if return_attn_stats:
            return self.predict_with_memory_and_stats(batch, support_tokens, support_mask=support_mask, train=train)
        return self.predict_with_memory(batch, support_tokens, support_mask=support_mask, train=train)

    def encode_support(self, batch: Dict[str, jnp.ndarray], *, train: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self.encoder.encode_support(batch, train=train)

    def encode_query(self, batch: Dict[str, jnp.ndarray], *, train: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self.encoder.encode_query(batch, train=train)

    def predict_with_memory(
        self,
        batch: Dict[str, jnp.ndarray],
        memory_tokens: Optional[jnp.ndarray],
        *,
        support_mask: Optional[jnp.ndarray] = None,
        train: bool = False,
    ) -> jnp.ndarray:
        q_tokens, q_mask = self.encode_query(batch, train=train)
        return self.decoder(q_tokens, memory_tokens, query_mask=q_mask, support_mask=support_mask, train=train)

    def predict_with_memory_and_stats(
        self,
        batch: Dict[str, jnp.ndarray],
        memory_tokens: Optional[jnp.ndarray],
        *,
        support_mask: Optional[jnp.ndarray] = None,
        train: bool = False,
    ):
        q_tokens, q_mask = self.encode_query(batch, train=train)
        return self.decoder(
            q_tokens,
            memory_tokens,
            query_mask=q_mask,
            support_mask=support_mask,
            train=train,
            return_attn_stats=True,
        )
