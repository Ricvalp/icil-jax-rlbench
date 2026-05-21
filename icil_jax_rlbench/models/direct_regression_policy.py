from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

import flax.linen as nn
import jax.numpy as jnp

from .attention import CrossAttentionBlock, MLP, MultiHeadAttention, SelfAttentionBlock, TransformerConfig
from .encoders import ContextEncoder, EncoderConfig


@dataclass(frozen=True)
class DecoderConfig:
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    context_mode: str = 'single_ctx'  # single_ctx | two_ctx | query_film_support
    mlp_mult: int = 4
    dropout: float = 0.0
    support_cross_layers: int = 2
    film_mlp_mult: int = 4

    def tx(self) -> TransformerConfig:
        return TransformerConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            mlp_mult=self.mlp_mult,
            dropout=self.dropout,
        )


@dataclass(frozen=True)
class ConditioningConfig:
    mode: str = 'support'  # support | none | task_variation | support_summary_film
    num_task_tokens: int = 1
    num_variation_tokens: int = 1
    num_tasks: int = 1
    num_task_variations: int = 1
    support_summary_source: str = 'traj_and_memory'  # traj_and_memory | traj | memory


@dataclass(frozen=True)
class PolicyConfig:
    encoder: EncoderConfig = EncoderConfig()
    decoder: DecoderConfig = DecoderConfig()
    conditioning: ConditioningConfig = ConditioningConfig()
    H: int = 16

    def with_model_dim(self) -> 'PolicyConfig':
        return replace(self, decoder=replace(self.decoder, d_model=self.encoder.d_model, n_heads=self.encoder.n_heads))


class ClassConditioner(nn.Module):
    cfg: ConditioningConfig
    d_model: int

    @nn.compact
    def __call__(self, batch: Dict[str, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        B = int(batch['query_xyz'].shape[0])
        d = int(self.d_model)
        pieces = []
        if int(self.cfg.num_task_tokens) > 0:
            table = self.param(
                'task_tokens',
                nn.initializers.normal(stddev=0.02),
                (max(1, int(self.cfg.num_tasks)), int(self.cfg.num_task_tokens), d),
            )
            task_id = jnp.asarray(batch['task_id'], dtype=jnp.int32).reshape((B,))
            task_id = jnp.clip(task_id, 0, max(0, int(self.cfg.num_tasks) - 1))
            pieces.append(table[task_id])
        if int(self.cfg.num_variation_tokens) > 0:
            table = self.param(
                'variation_tokens',
                nn.initializers.normal(stddev=0.02),
                (max(1, int(self.cfg.num_task_variations)), int(self.cfg.num_variation_tokens), d),
            )
            variation_id = jnp.asarray(batch['task_variation_id'], dtype=jnp.int32).reshape((B,))
            variation_id = jnp.clip(variation_id, 0, max(0, int(self.cfg.num_task_variations) - 1))
            pieces.append(table[variation_id])
        if not pieces:
            tokens = jnp.zeros((B, 0, d), dtype=jnp.float32)
        else:
            tokens = jnp.concatenate(pieces, axis=1)
        mask = jnp.ones(tokens.shape[:2], dtype=jnp.bool_)
        return tokens, mask


class AdaptiveLayerNorm(nn.Module):
    cfg: TransformerConfig
    film_mlp_mult: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray, summary: jnp.ndarray) -> jnp.ndarray:
        d = int(self.cfg.d_model)
        y = nn.LayerNorm(
            use_scale=False,
            use_bias=False,
            dtype=self.cfg.dtype,
            param_dtype=self.cfg.param_dtype,
            name='ln',
        )(x)
        h = nn.Dense(
            max(d, int(self.film_mlp_mult) * d),
            dtype=self.cfg.dtype,
            param_dtype=self.cfg.param_dtype,
            name='summary_fc1',
        )(summary.astype(jnp.float32))
        h = nn.gelu(h)
        film = nn.Dense(
            2 * d,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            dtype=self.cfg.dtype,
            param_dtype=self.cfg.param_dtype,
            name='summary_to_scale_shift',
        )(h)
        scale, shift = jnp.split(film, 2, axis=-1)
        return y * (1.0 + scale[:, None, :]) + shift[:, None, :]


class FiLMSupportDecoderBlock(nn.Module):
    cfg: TransformerConfig
    film_mlp_mult: int
    use_support_cross: bool

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        query_tokens: jnp.ndarray,
        support_tokens: Optional[jnp.ndarray],
        summary: jnp.ndarray,
        *,
        query_mask: Optional[jnp.ndarray],
        support_mask: Optional[jnp.ndarray],
        train: bool = False,
        return_weights: bool = False,
    ):
        y = AdaptiveLayerNorm(self.cfg, self.film_mlp_mult, name='self_film')(x, summary)
        x = x + MultiHeadAttention(self.cfg, name='self_attn')(y, kv_mask=None, train=train)

        y = AdaptiveLayerNorm(self.cfg, self.film_mlp_mult, name='query_film')(x, summary)
        if return_weights:
            query_out, query_weights = MultiHeadAttention(self.cfg, name='query_attn')(
                y, kv=query_tokens, kv_mask=query_mask, train=train, return_weights=True
            )
        else:
            query_out = MultiHeadAttention(self.cfg, name='query_attn')(
                y, kv=query_tokens, kv_mask=query_mask, train=train
            )
            query_weights = None
        x = x + query_out

        support_weights = None
        if bool(self.use_support_cross) and support_tokens is not None:
            y = AdaptiveLayerNorm(self.cfg, self.film_mlp_mult, name='support_film')(x, summary)
            if return_weights:
                support_out, support_weights = MultiHeadAttention(self.cfg, name='support_attn')(
                    y, kv=support_tokens, kv_mask=support_mask, train=train, return_weights=True
                )
            else:
                support_out = MultiHeadAttention(self.cfg, name='support_attn')(
                    y, kv=support_tokens, kv_mask=support_mask, train=train
                )
            x = x + support_out

        y = AdaptiveLayerNorm(self.cfg, self.film_mlp_mult, name='mlp_film')(x, summary)
        x = x + MLP(self.cfg, name='mlp')(y, train=train)
        if return_weights:
            return x, query_weights, support_weights
        return x


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
        support_summary: Optional[jnp.ndarray] = None,
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
        zero = jnp.asarray(0.0, dtype=jnp.float32)
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
        elif mode == 'query_film_support':
            if support_summary is None:
                support_summary = jnp.zeros((B, d), dtype=query_tokens.dtype)
            support_cross_layers = max(0, min(int(self.cfg.support_cross_layers), int(self.cfg.n_layers)))
            first_support_layer = int(self.cfg.n_layers) - support_cross_layers
            for i in range(int(self.cfg.n_layers)):
                use_support_cross = bool(support_tokens is not None and i >= first_support_layer)
                if return_attn_stats:
                    x, query_weights, support_weights = FiLMSupportDecoderBlock(
                        tx,
                        film_mlp_mult=int(self.cfg.film_mlp_mult),
                        use_support_cross=use_support_cross,
                        name=f'film_block_{i}',
                    )(
                        x,
                        query_tokens,
                        support_tokens,
                        support_summary,
                        query_mask=query_mask,
                        support_mask=support_mask,
                        train=train,
                        return_weights=True,
                    )
                    query_stats = self._context_attention_stats(query_weights, query_mask)
                    query_entropies.append(query_stats['attn_entropy'])
                    if support_weights is not None:
                        stats = self._context_attention_stats(support_weights, support_mask)
                        entropies.append(stats['attn_entropy'])
                        max_probs.append(stats['attn_max'])
                        raw_max_probs.append(stats['attn_raw_max'])
                        masses.append(stats['attn_mass'])
                else:
                    x = FiLMSupportDecoderBlock(
                        tx,
                        film_mlp_mult=int(self.cfg.film_mlp_mult),
                        use_support_cross=use_support_cross,
                        name=f'film_block_{i}',
                    )(
                        x,
                        query_tokens,
                        support_tokens,
                        support_summary,
                        query_mask=query_mask,
                        support_mask=support_mask,
                        train=train,
                    )
        else:
            raise ValueError("decoder.context_mode must be 'single_ctx', 'two_ctx', or 'query_film_support'.")
        if mode == 'query_film_support':
            if support_summary is None:
                support_summary = jnp.zeros((B, d), dtype=query_tokens.dtype)
            x = AdaptiveLayerNorm(tx, int(self.cfg.film_mlp_mult), name='out_film')(x, support_summary)
        else:
            x = nn.LayerNorm(name='out_ln')(x)
        pred = nn.Dense(int(self.action_dim), name='action_head')(x)
        if return_attn_stats:
            stats = {
                'attn_memory_entropy': self._mean_or_zero(entropies),
                'attn_memory_max': self._mean_or_zero(max_probs),
                'attn_memory_raw_max': self._mean_or_zero(raw_max_probs),
                'attn_memory_mass': self._mean_or_zero(masses),
                'attn_query_entropy': self._mean_or_zero(query_entropies),
            }
            return pred, stats
        return pred

    def _mean_or_zero(self, values):
        if not values:
            return jnp.asarray(0.0, dtype=jnp.float32)
        return jnp.mean(jnp.stack(values))

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
        self.class_conditioner = ClassConditioner(
            self.cfg.conditioning,
            int(self.cfg.encoder.d_model),
            name='class_conditioner',
        )

    def _conditioning_mode(self) -> str:
        mode = str(self.cfg.conditioning.mode)
        if mode not in ('support', 'none', 'task_variation', 'support_summary_film'):
            raise ValueError("conditioning.mode must be 'support', 'none', 'task_variation', or 'support_summary_film'.")
        return mode

    def _uses_support_conditioning(self) -> bool:
        return self._conditioning_mode() in ('support', 'support_summary_film') and bool(self.cfg.encoder.use_support_tokens)

    def _uses_support_summary(self) -> bool:
        mode = self._conditioning_mode()
        return mode == 'support_summary_film' or (mode == 'support' and str(self.cfg.decoder.context_mode) == 'query_film_support')

    def __call__(self, batch: Dict[str, jnp.ndarray], *, train: bool = False, return_attn_stats: bool = False):
        support_stats = {}
        support_summary = None
        mode = self._conditioning_mode()
        if mode == 'task_variation':
            support_tokens, support_mask = self.class_conditioner(batch)
            if return_attn_stats:
                zero = jnp.asarray(0.0, dtype=jnp.float32)
                support_stats = {
                    'attn_support_input_mass': zero,
                    'attn_support_input_entropy': zero,
                    'attn_support_input_max': zero,
                    'attn_traj_input_mass': zero,
                    'attn_traj_input_entropy': zero,
                    'attn_traj_input_max': zero,
                }
        elif self._uses_support_summary() and bool(self.cfg.encoder.use_support_tokens):
            if return_attn_stats:
                support_tokens, support_mask, support_summary, support_stats = self.encoder.encode_support(
                    batch,
                    train=train,
                    return_attn_stats=True,
                    return_summary=True,
                    summary_source=str(self.cfg.conditioning.support_summary_source),
                )
            else:
                support_tokens, support_mask, support_summary = self.encoder.encode_support(
                    batch,
                    train=train,
                    return_summary=True,
                    summary_source=str(self.cfg.conditioning.support_summary_source),
                )
        elif mode == 'support' and bool(self.cfg.encoder.use_support_tokens):
            if return_attn_stats:
                support_tokens, support_mask, support_stats = self.encoder.encode_support(
                    batch, train=train, return_attn_stats=True
                )
            else:
                support_tokens, support_mask = self.encode_support(batch, train=train)
        else:
            support_tokens, support_mask = None, None
            if return_attn_stats:
                zero = jnp.asarray(0.0, dtype=jnp.float32)
                support_stats = {
                    'attn_support_input_mass': zero,
                    'attn_support_input_entropy': zero,
                    'attn_support_input_max': zero,
                    'attn_traj_input_mass': zero,
                    'attn_traj_input_entropy': zero,
                    'attn_traj_input_max': zero,
                }
        if return_attn_stats:
            pred, decoder_stats = self.predict_with_memory_and_stats(
                batch,
                support_tokens,
                support_mask=support_mask,
                support_summary=support_summary,
                train=train,
            )
            return pred, {**support_stats, **decoder_stats}
        return self.predict_with_memory(
            batch,
            support_tokens,
            support_mask=support_mask,
            support_summary=support_summary,
            train=train,
        )

    def encode_support(self, batch: Dict[str, jnp.ndarray], *, train: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self.encoder.encode_support(batch, train=train)

    def encode_support_with_stats(self, batch: Dict[str, jnp.ndarray], *, train: bool = False):
        return self.encoder.encode_support(batch, train=train, return_attn_stats=True)

    def encode_support_conditioning(self, batch: Dict[str, jnp.ndarray], *, train: bool = False):
        return self.encoder.encode_support(
            batch,
            train=train,
            return_summary=True,
            summary_source=str(self.cfg.conditioning.support_summary_source),
        )

    def encode_query(self, batch: Dict[str, jnp.ndarray], *, train: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self.encoder.encode_query(batch, train=train)

    def predict_with_memory(
        self,
        batch: Dict[str, jnp.ndarray],
        memory_tokens: Optional[jnp.ndarray],
        *,
        support_mask: Optional[jnp.ndarray] = None,
        support_summary: Optional[jnp.ndarray] = None,
        train: bool = False,
    ) -> jnp.ndarray:
        q_tokens, q_mask = self.encode_query(batch, train=train)
        return self.decoder(
            q_tokens,
            memory_tokens,
            query_mask=q_mask,
            support_mask=support_mask,
            support_summary=support_summary,
            train=train,
        )

    def predict_with_memory_and_stats(
        self,
        batch: Dict[str, jnp.ndarray],
        memory_tokens: Optional[jnp.ndarray],
        *,
        support_mask: Optional[jnp.ndarray] = None,
        support_summary: Optional[jnp.ndarray] = None,
        train: bool = False,
    ):
        q_tokens, q_mask = self.encode_query(batch, train=train)
        return self.decoder(
            q_tokens,
            memory_tokens,
            query_mask=q_mask,
            support_mask=support_mask,
            support_summary=support_summary,
            train=train,
            return_attn_stats=True,
        )
