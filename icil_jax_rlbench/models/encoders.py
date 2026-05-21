from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from .attention import TransformerConfig
from .perceiver import LatentPerceiver, PerceiverConfig, SelfAttentionStack


@dataclass(frozen=True)
class EncoderConfig:
    encoder_type: str = 'perceiver'  # perceiver | supernode
    d_model: int = 256
    n_heads: int = 4
    dropout: float = 0.0
    mlp_mult: int = 4
    frame_num_latents: int = 8
    frame_layers: int = 2
    query_num_latents: int = 16
    support_num_latents: int = 64
    support_layers: int = 2
    query_layers: int = 1
    support_tokenizer: str = 'frame'  # frame | spacetime_supernode
    supernodes: int = 64
    supernode_temperature: float = 0.005
    supernode_center_sampling: str = 'linspace'  # linspace | mask_balanced
    supernode_layers: int = 2
    spacetime_supernodes: int = 256
    spacetime_temperature_xyz: float = 0.005
    spacetime_temperature_t: float = 0.04
    spacetime_layers: int = 2
    traj_layers: int = 1
    max_positions: int = 0
    mask_id_vocab: int = 256
    use_support_tokens: bool = True
    use_rgb: bool = True
    use_mask_id: bool = False
    use_traj_tokens: bool = True

    def tx(self) -> TransformerConfig:
        return TransformerConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            mlp_mult=self.mlp_mult,
            dropout=self.dropout,
        )

    def perceiver(self, *, num_latents: int, n_layers: int) -> PerceiverConfig:
        return PerceiverConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=n_layers,
            num_latents=num_latents,
            mlp_mult=self.mlp_mult,
            dropout=self.dropout,
        )



class PointFeatureEmbed(nn.Module):
    cfg: EncoderConfig

    @nn.compact
    def __call__(
        self,
        xyz: jnp.ndarray,
        *,
        rgb: Optional[jnp.ndarray] = None,
        mask_id: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        d = int(self.cfg.d_model)
        pieces = [xyz.astype(jnp.float32)]
        if bool(self.cfg.use_rgb) and rgb is not None:
            pieces.append(rgb.astype(jnp.float32))
        x = jnp.concatenate(pieces, axis=-1)
        x = nn.Dense(d, name='xyz_rgb_proj')(x)
        if bool(self.cfg.use_mask_id) and mask_id is not None:
            vocab = int(self.cfg.mask_id_vocab)
            mid = jnp.clip(mask_id.astype(jnp.int32), 0, vocab - 1)
            x = x + nn.Embed(vocab, d, name='mask_embed')(mid)
        return x


class PerceiverFrameTokenizer(nn.Module):
    cfg: EncoderConfig

    @nn.compact
    def __call__(
        self,
        xyz: jnp.ndarray,
        state: jnp.ndarray,
        valid: jnp.ndarray,
        *,
        rgb: Optional[jnp.ndarray] = None,
        mask_id: Optional[jnp.ndarray] = None,
        train: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Inputs are flattened frames: xyz=[Bf,N,3], state=[Bf,S].
        d = int(self.cfg.d_model)
        point_tokens = PointFeatureEmbed(self.cfg, name='point_embed')(xyz, rgb=rgb, mask_id=mask_id)
        state_token = nn.Dense(d, name='state_proj')(state.astype(jnp.float32))[:, None, :]
        tokens = jnp.concatenate([point_tokens, state_token], axis=1)
        mask = jnp.concatenate([valid.astype(jnp.bool_), jnp.ones((valid.shape[0], 1), dtype=jnp.bool_)], axis=1)
        latents = LatentPerceiver(
            self.cfg.perceiver(num_latents=int(self.cfg.frame_num_latents), n_layers=int(self.cfg.frame_layers)),
            name='frame_perceiver',
        )(tokens, token_mask=mask, train=train)
        return latents, jnp.ones((latents.shape[0], latents.shape[1]), dtype=jnp.bool_)


def _linspace_center_indices(batch_size: int, num_points: int, num_centers: int) -> jnp.ndarray:
    idx = jnp.linspace(0, max(int(num_points) - 1, 0), int(num_centers)).round().astype(jnp.int32)
    return jnp.broadcast_to(idx[None, :], (int(batch_size), int(num_centers)))


def _mask_balanced_center_indices(valid: jnp.ndarray, mask_id: jnp.ndarray, num_centers: int) -> jnp.ndarray:
    valid = valid.astype(jnp.bool_)
    mask_id = mask_id.astype(jnp.int32)
    B, N = int(valid.shape[0]), int(valid.shape[1])
    M = int(num_centers)

    def one_row(row_valid: jnp.ndarray, row_mask: jnp.ndarray) -> jnp.ndarray:
        pos = jnp.arange(N, dtype=jnp.int32)
        invalid_mask = jnp.asarray(jnp.iinfo(jnp.int32).max, dtype=jnp.int32)
        safe_mask = jnp.where(row_valid, row_mask, invalid_mask)
        order = jnp.argsort(safe_mask, stable=True)
        sorted_valid = row_valid[order]
        sorted_mask = safe_mask[order]

        prev_mask = jnp.concatenate([jnp.asarray([-1], dtype=jnp.int32), sorted_mask[:-1]], axis=0)
        group_start = (sorted_mask != prev_mask) & sorted_valid
        group_id = jnp.cumsum(group_start.astype(jnp.int32), axis=0) - 1
        safe_group_id = jnp.maximum(group_id, 0)

        sorted_pos = jnp.arange(N, dtype=jnp.int32)
        start_pos = jnp.where(group_start, sorted_pos, jnp.asarray(0, dtype=jnp.int32))
        group_start_pos = jax.lax.associative_scan(jnp.maximum, start_pos, axis=0)
        rank_in_group = sorted_pos - group_start_pos

        group_counts = jnp.bincount(
            safe_group_id,
            weights=sorted_valid.astype(jnp.float32),
            length=N,
        )
        count = jnp.maximum(group_counts[safe_group_id], 1.0)
        rank_fraction = rank_in_group.astype(jnp.float32) / count

        # Sort first by within-mask rank fraction, then by mask-group order. Taking the
        # first M indices gives one point per visible mask before taking second points.
        group_tie = safe_group_id.astype(jnp.float32) / jnp.asarray(max(N, 1), dtype=jnp.float32)
        balanced_key = rank_fraction + 1e-3 * group_tie
        invalid_key = 2.0 + sorted_pos.astype(jnp.float32) / jnp.asarray(max(N, 1), dtype=jnp.float32)
        balanced_key = jnp.where(sorted_valid, balanced_key, invalid_key)
        balanced_order = jnp.argsort(balanced_key, stable=True)
        take_pos = jnp.arange(M, dtype=jnp.int32) % max(N, 1)
        return order[balanced_order[take_pos]]

    return jax.vmap(one_row)(valid, mask_id).reshape(B, M)


def _supernode_center_indices(
    *,
    valid: jnp.ndarray,
    mask_id: Optional[jnp.ndarray],
    num_centers: int,
    center_sampling: str,
) -> jnp.ndarray:
    B, N = int(valid.shape[0]), int(valid.shape[1])
    if str(center_sampling) == 'mask_balanced' and mask_id is not None:
        return _mask_balanced_center_indices(valid, mask_id, int(num_centers))
    if str(center_sampling) != 'linspace':
        raise ValueError("encoder.supernode_center_sampling must be 'linspace' or 'mask_balanced'.")
    return _linspace_center_indices(B, N, int(num_centers))


class SupernodeFrameTokenizer(nn.Module):
    cfg: EncoderConfig

    @nn.compact
    def __call__(
        self,
        xyz: jnp.ndarray,
        state: jnp.ndarray,
        valid: jnp.ndarray,
        *,
        rgb: Optional[jnp.ndarray] = None,
        mask_id: Optional[jnp.ndarray] = None,
        train: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Deterministic soft supernode pooling around selected point centers.
        d = int(self.cfg.d_model)
        M = int(self.cfg.supernodes)
        N = int(xyz.shape[1])
        idx = _supernode_center_indices(
            valid=valid,
            mask_id=mask_id,
            num_centers=M,
            center_sampling=str(self.cfg.supernode_center_sampling),
        )
        point_tokens = PointFeatureEmbed(self.cfg, name='point_embed')(xyz, rgb=rgb, mask_id=mask_id)
        centers = jnp.take_along_axis(xyz.astype(jnp.float32), idx[:, :, None], axis=1)
        dist2 = jnp.sum((centers[:, :, None, :] - xyz[:, None, :, :].astype(jnp.float32)) ** 2, axis=-1)
        logits = -dist2 / max(float(self.cfg.supernode_temperature), 1e-6)
        logits = jnp.where(valid[:, None, :].astype(jnp.bool_), logits, jnp.asarray(-1e9, logits.dtype))
        weights = nn.softmax(logits, axis=-1)
        super_tokens = jnp.einsum('bmn,bnd->bmd', weights.astype(point_tokens.dtype), point_tokens)
        state_token = nn.Dense(d, name='state_proj')(state.astype(jnp.float32))[:, None, :]
        tokens = jnp.concatenate([super_tokens, state_token], axis=1)
        mask = jnp.ones((tokens.shape[0], tokens.shape[1]), dtype=jnp.bool_)
        tokens = SelfAttentionStack(self.cfg.tx(), int(self.cfg.supernode_layers), name='supernode_refine')(tokens, mask=mask, train=train)
        return tokens, mask


class SpacetimeSupportTokenizer(nn.Module):
    cfg: EncoderConfig

    @nn.compact
    def __call__(
        self,
        xyz: jnp.ndarray,
        time: jnp.ndarray,
        state: jnp.ndarray,
        valid: jnp.ndarray,
        *,
        rgb: Optional[jnp.ndarray] = None,
        mask_id: Optional[jnp.ndarray] = None,
        train: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Inputs are support unions: xyz=[B,K,P,3], time=[B,K,P], state=[B,K,P,S].
        B, K, P = int(xyz.shape[0]), int(xyz.shape[1]), int(xyz.shape[2])
        M = int(self.cfg.spacetime_supernodes)
        d = int(self.cfg.d_model)
        if M <= 0:
            return (
                jnp.zeros((B, 0, d), dtype=jnp.float32),
                jnp.zeros((B, 0), dtype=jnp.bool_),
            )
        if int(self.cfg.max_positions) < max(M, K):
            raise ValueError(
                f'encoder.max_positions={int(self.cfg.max_positions)} is too small for spacetime support '
                f'(spacetime_supernodes={M}, K={K}). Set it to 0 to infer automatically or increase it.'
            )

        flat_xyz = xyz.reshape(B * K, P, xyz.shape[-1]).astype(jnp.float32)
        flat_time = time.reshape(B * K, P).astype(jnp.float32)
        flat_state = state.reshape(B * K, P, state.shape[-1]).astype(jnp.float32)
        flat_valid = valid.reshape(B * K, P).astype(jnp.bool_)
        flat_rgb = None if rgb is None else rgb.reshape(B * K, P, rgb.shape[-1])
        flat_mask = None if mask_id is None else mask_id.reshape(B * K, P)

        pieces = [flat_xyz, flat_time[:, :, None], flat_state]
        if bool(self.cfg.use_rgb) and flat_rgb is not None:
            pieces.append(flat_rgb.astype(jnp.float32))
        point_features = jnp.concatenate(pieces, axis=-1)
        point_tokens = nn.Dense(d, name='spacetime_point_proj')(point_features)
        if bool(self.cfg.use_mask_id) and flat_mask is not None:
            vocab = int(self.cfg.mask_id_vocab)
            mid = jnp.clip(flat_mask.astype(jnp.int32), 0, vocab - 1)
            point_tokens = point_tokens + nn.Embed(vocab, d, name='spacetime_mask_embed')(mid)

        idx = _supernode_center_indices(
            valid=flat_valid,
            mask_id=flat_mask,
            num_centers=M,
            center_sampling=str(self.cfg.supernode_center_sampling),
        )
        centers_xyz = jnp.take_along_axis(flat_xyz, idx[:, :, None], axis=1)
        centers_time = jnp.take_along_axis(flat_time, idx, axis=1)
        xyz_dist2 = jnp.sum((centers_xyz[:, :, None, :] - flat_xyz[:, None, :, :]) ** 2, axis=-1)
        time_dist2 = (centers_time[:, :, None] - flat_time[:, None, :]) ** 2
        logits = -(
            xyz_dist2 / max(float(self.cfg.spacetime_temperature_xyz), 1e-6)
            + time_dist2 / max(float(self.cfg.spacetime_temperature_t), 1e-6)
        )
        logits = jnp.where(flat_valid[:, None, :], logits, jnp.asarray(-1e9, logits.dtype))
        weights = nn.softmax(logits, axis=-1)
        super_tokens = jnp.einsum('bmn,bnd->bmd', weights.astype(point_tokens.dtype), point_tokens)
        super_tokens = super_tokens.reshape(B, K, M, d)

        st_pos = self.param('spacetime_pos', nn.initializers.normal(stddev=0.02), (int(self.cfg.max_positions), d))[:M]
        demo_pos = self.param('spacetime_demo_pos', nn.initializers.normal(stddev=0.02), (int(self.cfg.max_positions), d))[:K]
        tokens = super_tokens + st_pos[None, None, :, :] + demo_pos[None, :, None, :]
        tokens = tokens.reshape(B, K * M, d)
        demo_valid = jnp.any(valid.astype(jnp.bool_), axis=-1)
        mask = jnp.broadcast_to(demo_valid[:, :, None], (B, K, M)).reshape(B, K * M)
        if int(self.cfg.spacetime_layers) > 0:
            tokens = SelfAttentionStack(self.cfg.tx(), int(self.cfg.spacetime_layers), name='spacetime_refine')(
                tokens, mask=mask, train=train
            )
        return tokens, mask


class TrajectoryTokenizer(nn.Module):
    cfg: EncoderConfig
    action_dim: int

    @nn.compact
    def __call__(self, traj: jnp.ndarray, traj_mask: jnp.ndarray, *, train: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # traj=[B,K,T,A]. This gives the model direct support action-shape evidence.
        B, K, T, _ = traj.shape
        d = int(self.cfg.d_model)
        x = nn.Dense(d, name='action_proj')(traj.astype(jnp.float32))
        max_positions = int(self.cfg.max_positions)
        if max_positions < max(int(T), int(K)):
            raise ValueError(
                f'encoder.max_positions={max_positions} is too small for trajectory tokens '
                f'(traj_len={int(T)}, K={int(K)}). Set it to 0 to infer automatically or increase it.'
            )
        tpos = self.param('traj_pos', nn.initializers.normal(stddev=0.02), (int(self.cfg.max_positions), d))[:T]
        dpos = self.param('traj_demo_pos', nn.initializers.normal(stddev=0.02), (int(self.cfg.max_positions), d))[:K]
        x = x + tpos[None, None, :, :] + dpos[None, :, None, :]
        x = x.reshape(B, K * T, d)
        mask = traj_mask.reshape(B, K * T).astype(jnp.bool_)
        if int(self.cfg.traj_layers) > 0:
            x = SelfAttentionStack(self.cfg.tx(), int(self.cfg.traj_layers), name='traj_refine')(x, mask=mask, train=train)
        return x, mask


class SupportSummaryHead(nn.Module):
    cfg: EncoderConfig
    source: str = 'traj_and_memory'

    @nn.compact
    def __call__(
        self,
        visual_tokens: jnp.ndarray,
        visual_mask: jnp.ndarray,
        traj_tokens: Optional[jnp.ndarray],
        traj_mask: Optional[jnp.ndarray],
    ) -> jnp.ndarray:
        d = int(self.cfg.d_model)
        source = str(self.source)
        visual_summary = self._masked_mean(visual_tokens, visual_mask)
        if traj_tokens is None or traj_mask is None:
            traj_summary = jnp.zeros_like(visual_summary)
        else:
            traj_summary = self._masked_mean(traj_tokens, traj_mask)
        if source == 'memory':
            x = visual_summary
        elif source == 'traj':
            x = traj_summary
        elif source == 'traj_and_memory':
            x = jnp.concatenate([visual_summary, traj_summary], axis=-1)
        else:
            raise ValueError(
                "conditioning.support_summary_source must be 'traj_and_memory', 'traj', or 'memory'."
            )
        x = nn.Dense(d, name='summary_fc1')(x.astype(jnp.float32))
        x = nn.gelu(x)
        x = nn.Dense(d, name='summary_fc2')(x)
        return x

    def _masked_mean(self, tokens: jnp.ndarray, mask: Optional[jnp.ndarray]) -> jnp.ndarray:
        if mask is None:
            return jnp.mean(tokens.astype(jnp.float32), axis=1)
        weights = mask.astype(jnp.float32)
        denom = jnp.maximum(jnp.sum(weights, axis=1, keepdims=True), 1.0)
        return jnp.sum(tokens.astype(jnp.float32) * weights[:, :, None], axis=1) / denom


class ContextEncoder(nn.Module):
    cfg: EncoderConfig
    state_dim: int
    action_dim: int

    def _segment_attention_stats(
        self,
        weights: jnp.ndarray,
        mask: Optional[jnp.ndarray],
        *,
        prefix: str,
    ) -> Dict[str, jnp.ndarray]:
        # weights=[layers,B,heads,latents,tokens].
        if int(weights.shape[-1]) == 0:
            zero = jnp.asarray(0.0, dtype=jnp.float32)
            return {
                f'attn_{prefix}_input_mass': zero,
                f'attn_{prefix}_input_entropy': zero,
                f'attn_{prefix}_input_max': zero,
            }
        w = weights.astype(jnp.float32)
        if mask is not None:
            m = mask.astype(jnp.bool_)[None, :, None, None, :]
            w = jnp.where(m, w, 0.0)
            valid_count = jnp.sum(mask.astype(jnp.float32), axis=-1)
        else:
            valid_count = jnp.full((weights.shape[1],), int(weights.shape[-1]), dtype=jnp.float32)
        mass = jnp.sum(w, axis=-1)
        dist = w / (mass[..., None] + 1e-8)
        entropy = -jnp.sum(jnp.where(dist > 0.0, dist * jnp.log(dist + 1e-8), 0.0), axis=-1)
        norm = jnp.log(jnp.maximum(valid_count, 2.0))[None, :, None, None]
        entropy = jnp.where(mass > 1e-8, entropy / norm, 0.0)
        return {
            f'attn_{prefix}_input_mass': jnp.mean(mass),
            f'attn_{prefix}_input_entropy': jnp.mean(entropy),
            f'attn_{prefix}_input_max': jnp.mean(jnp.max(dist, axis=-1)),
        }

    def _support_traj_attention_stats(
        self,
        weights: jnp.ndarray,
        mask: jnp.ndarray,
        *,
        support_token_count: int,
    ) -> Dict[str, jnp.ndarray]:
        support_len = int(support_token_count)
        support_stats = self._segment_attention_stats(
            weights[..., :support_len],
            None if mask is None else mask[:, :support_len],
            prefix='support',
        )
        traj_stats = self._segment_attention_stats(
            weights[..., support_len:],
            None if mask is None else mask[:, support_len:],
            prefix='traj',
        )
        return {**support_stats, **traj_stats}

    def _tokenize_frames(
        self,
        xyz: jnp.ndarray,
        state: jnp.ndarray,
        valid: jnp.ndarray,
        *,
        rgb: Optional[jnp.ndarray],
        mask_id: Optional[jnp.ndarray],
        role: str,
        train: bool,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # xyz=[B,F,N,3], state=[B,F,S]. Returns [B,F*T,d].
        B, F = int(xyz.shape[0]), int(xyz.shape[1])
        flat_xyz = xyz.reshape(B * F, xyz.shape[-2], xyz.shape[-1])
        flat_state = state.reshape(B * F, state.shape[-1])
        flat_valid = valid.reshape(B * F, valid.shape[-1])
        flat_rgb = None if rgb is None else rgb.reshape(B * F, rgb.shape[-2], rgb.shape[-1])
        flat_mask = None if mask_id is None else mask_id.reshape(B * F, mask_id.shape[-1])
        if self.cfg.encoder_type == 'perceiver':
            frame_tokens, frame_mask = PerceiverFrameTokenizer(self.cfg, name=f'{role}_perceiver_frame')(
                flat_xyz, flat_state, flat_valid, rgb=flat_rgb, mask_id=flat_mask, train=train
            )
        elif self.cfg.encoder_type == 'supernode':
            frame_tokens, frame_mask = SupernodeFrameTokenizer(self.cfg, name=f'{role}_supernode_frame')(
                flat_xyz, flat_state, flat_valid, rgb=flat_rgb, mask_id=flat_mask, train=train
            )
        else:
            raise ValueError(f'Unknown encoder_type={self.cfg.encoder_type!r}')
        Ttok = int(frame_tokens.shape[1])
        tokens = frame_tokens.reshape(B, F * Ttok, int(self.cfg.d_model))
        mask = frame_mask.reshape(B, F * Ttok)
        needed_positions = F * Ttok
        if int(self.cfg.max_positions) < int(needed_positions):
            raise ValueError(
                f'encoder.max_positions={int(self.cfg.max_positions)} is too small for {role} tokens '
                f'({needed_positions}=frames {F} * tokens_per_frame {Ttok}). '
                'Set it to 0 to infer automatically or increase it.'
            )
        pos = self.param(f'{role}_pos', nn.initializers.normal(stddev=0.02), (int(self.cfg.max_positions), int(self.cfg.d_model)))[: F * Ttok]
        tokens = tokens + pos[None, :, :]
        return tokens, mask

    @nn.compact
    def encode_support(
        self,
        batch: Dict[str, jnp.ndarray],
        *,
        train: bool = False,
        return_attn_stats: bool = False,
        return_summary: bool = False,
        summary_source: str = 'traj_and_memory',
    ):
        if not bool(self.cfg.use_support_tokens):
            B = int(batch['query_xyz'].shape[0]) if 'query_xyz' in batch else int(batch['cond_xyz'].shape[0])
            summary = jnp.zeros((B, int(self.cfg.d_model)), dtype=jnp.float32)
            if return_attn_stats:
                zero = jnp.asarray(0.0, dtype=jnp.float32)
                stats = {
                    'attn_support_input_mass': zero,
                    'attn_support_input_entropy': zero,
                    'attn_support_input_max': zero,
                    'attn_traj_input_mass': zero,
                    'attn_traj_input_entropy': zero,
                    'attn_traj_input_max': zero,
                }
                if return_summary:
                    return None, None, summary, stats
                return None, None, stats
            if return_summary:
                return None, None, summary
            return None, None
        support_tokenizer = str(self.cfg.support_tokenizer)
        if support_tokenizer == 'spacetime_supernode':
            if 'cond_st_xyz' not in batch:
                raise ValueError(
                    'support_tokenizer=spacetime_supernode requires cond_st_* fields. '
                    'Set data.support_spacetime_points > 0.'
                )
            xyz = batch['cond_st_xyz']
            B, K = int(xyz.shape[0]), int(xyz.shape[1])
            visual_tokens, visual_mask = SpacetimeSupportTokenizer(self.cfg, name='support_spacetime')(
                xyz,
                batch['cond_st_time'],
                batch['cond_st_state'],
                batch['cond_st_valid'],
                rgb=batch.get('cond_st_rgb'),
                mask_id=batch.get('cond_st_mask_id'),
                train=train,
            )
        elif support_tokenizer == 'frame':
            xyz = batch['cond_xyz']
            B, K, L = int(xyz.shape[0]), int(xyz.shape[1]), int(xyz.shape[2])
            rgb = batch.get('cond_rgb')
            mask_id = batch.get('cond_mask_id')
            visual_tokens, visual_mask = self._tokenize_frames(
                xyz.reshape(B, K * L, xyz.shape[-2], xyz.shape[-1]),
                batch['cond_state'].reshape(B, K * L, batch['cond_state'].shape[-1]),
                batch['cond_valid'].reshape(B, K * L, batch['cond_valid'].shape[-1]),
                rgb=None if rgb is None else rgb.reshape(B, K * L, rgb.shape[-2], rgb.shape[-1]),
                mask_id=None if mask_id is None else mask_id.reshape(B, K * L, mask_id.shape[-1]),
                role='support',
                train=train,
            )
        else:
            raise ValueError("encoder.support_tokenizer must be 'frame' or 'spacetime_supernode'.")
        tokens, mask = visual_tokens, visual_mask
        support_token_count = int(visual_tokens.shape[1])
        traj_tokens, traj_mask = None, None
        if bool(self.cfg.use_traj_tokens) and 'cond_traj' in batch:
            traj_tokens, traj_mask = TrajectoryTokenizer(self.cfg, self.action_dim, name='traj_tokenizer')(
                batch['cond_traj'], batch['cond_traj_mask'], train=train
            )
            tokens = jnp.concatenate([tokens, traj_tokens], axis=1)
            mask = jnp.concatenate([mask, traj_mask], axis=1)
        summary = None
        if bool(return_summary):
            summary = SupportSummaryHead(self.cfg, source=str(summary_source), name='support_summary')(
                visual_tokens,
                visual_mask,
                traj_tokens,
                traj_mask,
            )
        stats: Dict[str, jnp.ndarray] = {}
        if int(self.cfg.support_num_latents) > 0:
            compressor = LatentPerceiver(
                self.cfg.perceiver(num_latents=int(self.cfg.support_num_latents), n_layers=int(self.cfg.support_layers)),
                name='support_compressor',
            )
            if return_attn_stats:
                tokens, weights = compressor(tokens, token_mask=mask, train=train, return_attn_weights=True)
                stats = self._support_traj_attention_stats(weights, mask, support_token_count=support_token_count)
            else:
                tokens = compressor(tokens, token_mask=mask, train=train)
            mask = jnp.ones((tokens.shape[0], tokens.shape[1]), dtype=jnp.bool_)
        elif return_attn_stats:
            zero = jnp.asarray(0.0, dtype=jnp.float32)
            stats = {
                'attn_support_input_mass': zero,
                'attn_support_input_entropy': zero,
                'attn_support_input_max': zero,
                'attn_traj_input_mass': zero,
                'attn_traj_input_entropy': zero,
                'attn_traj_input_max': zero,
            }
        if return_attn_stats:
            if return_summary:
                return tokens, mask, summary, stats
            return tokens, mask, stats
        if return_summary:
            return tokens, mask, summary
        return tokens, mask

    @nn.compact
    def encode_query(self, batch: Dict[str, jnp.ndarray], *, train: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        rgb = batch.get('query_rgb')
        mask_id = batch.get('query_mask_id')
        tokens, mask = self._tokenize_frames(
            batch['query_xyz'],
            batch['query_state'],
            batch['query_valid'],
            rgb=rgb,
            mask_id=mask_id,
            role='query',
            train=train,
        )
        if int(self.cfg.query_layers) > 0:
            tokens = SelfAttentionStack(self.cfg.tx(), int(self.cfg.query_layers), name='query_refine')(tokens, mask=mask, train=train)
        if int(self.cfg.query_num_latents) > 0:
            tokens = LatentPerceiver(
                self.cfg.perceiver(num_latents=int(self.cfg.query_num_latents), n_layers=1),
                name='query_compressor',
            )(tokens, token_mask=mask, train=train)
            mask = jnp.ones((tokens.shape[0], tokens.shape[1]), dtype=jnp.bool_)
        return tokens, mask
