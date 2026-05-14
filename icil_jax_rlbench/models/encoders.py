from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import flax.linen as nn
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
    supernodes: int = 64
    supernode_temperature: float = 0.02
    supernode_layers: int = 2
    traj_layers: int = 1
    max_positions: int = 0
    mask_id_vocab: int = 256
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
        # Deterministic soft supernode pooling around evenly spaced point centers.
        d = int(self.cfg.d_model)
        M = int(self.cfg.supernodes)
        N = int(xyz.shape[1])
        idx = jnp.linspace(0, max(N - 1, 0), M).round().astype(jnp.int32)
        point_tokens = PointFeatureEmbed(self.cfg, name='point_embed')(xyz, rgb=rgb, mask_id=mask_id)
        centers = jnp.take(xyz.astype(jnp.float32), idx, axis=1)
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


class ContextEncoder(nn.Module):
    cfg: EncoderConfig
    state_dim: int
    action_dim: int

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
    def encode_support(self, batch: Dict[str, jnp.ndarray], *, train: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        xyz = batch['cond_xyz']
        B, K, L = int(xyz.shape[0]), int(xyz.shape[1]), int(xyz.shape[2])
        rgb = batch.get('cond_rgb')
        mask_id = batch.get('cond_mask_id')
        tokens, mask = self._tokenize_frames(
            xyz.reshape(B, K * L, xyz.shape[-2], xyz.shape[-1]),
            batch['cond_state'].reshape(B, K * L, batch['cond_state'].shape[-1]),
            batch['cond_valid'].reshape(B, K * L, batch['cond_valid'].shape[-1]),
            rgb=None if rgb is None else rgb.reshape(B, K * L, rgb.shape[-2], rgb.shape[-1]),
            mask_id=None if mask_id is None else mask_id.reshape(B, K * L, mask_id.shape[-1]),
            role='support',
            train=train,
        )
        if bool(self.cfg.use_traj_tokens) and 'cond_traj' in batch:
            traj_tokens, traj_mask = TrajectoryTokenizer(self.cfg, self.action_dim, name='traj_tokenizer')(
                batch['cond_traj'], batch['cond_traj_mask'], train=train
            )
            tokens = jnp.concatenate([tokens, traj_tokens], axis=1)
            mask = jnp.concatenate([mask, traj_mask], axis=1)
        if int(self.cfg.support_num_latents) > 0:
            tokens = LatentPerceiver(
                self.cfg.perceiver(num_latents=int(self.cfg.support_num_latents), n_layers=int(self.cfg.support_layers)),
                name='support_compressor',
            )(tokens, token_mask=mask, train=train)
            mask = jnp.ones((tokens.shape[0], tokens.shape[1]), dtype=jnp.bool_)
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
