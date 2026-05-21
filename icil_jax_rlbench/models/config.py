from __future__ import annotations

from dataclasses import replace
from typing import Any

from .direct_regression_policy import ConditioningConfig, DecoderConfig, PolicyConfig
from .encoders import EncoderConfig


def get_value(cfg: Any, name: str, default):
    return getattr(cfg, name, default) if cfg is not None else default


def encoder_config_from(cfg: Any) -> EncoderConfig:
    base = EncoderConfig()
    vals = {field: get_value(cfg, field, getattr(base, field)) for field in base.__dataclass_fields__}
    return EncoderConfig(**vals)


def decoder_config_from(cfg: Any, d_model: int, n_heads: int) -> DecoderConfig:
    base = DecoderConfig(d_model=d_model, n_heads=n_heads)
    vals = {field: get_value(cfg, field, getattr(base, field)) for field in base.__dataclass_fields__}
    vals['d_model'] = d_model
    vals['n_heads'] = n_heads
    return DecoderConfig(**vals)


def conditioning_config_from(cfg: Any) -> ConditioningConfig:
    base = ConditioningConfig()
    vals = {field: get_value(cfg, field, getattr(base, field)) for field in base.__dataclass_fields__}
    return ConditioningConfig(**vals)


def _infer_frame_tokens_per_frame(enc: EncoderConfig) -> int:
    if str(enc.encoder_type) == 'supernode':
        return int(enc.supernodes) + 1
    if str(enc.encoder_type) == 'perceiver':
        return int(enc.frame_num_latents)
    raise ValueError(f'Unknown encoder_type={enc.encoder_type!r}')


def _infer_max_positions(enc: EncoderConfig, data_cfg: Any) -> int:
    if data_cfg is None:
        raise ValueError('encoder.max_positions=0 requires data_cfg so positional length can be inferred.')
    per_frame = _infer_frame_tokens_per_frame(enc)
    K = int(get_value(data_cfg, 'K', 1))
    L = int(get_value(data_cfg, 'L', 1))
    T_obs = int(get_value(data_cfg, 'T_obs', 1))
    traj_len = int(get_value(data_cfg, 'traj_len', 0))
    if bool(enc.use_support_tokens) and str(enc.support_tokenizer) == 'spacetime_supernode':
        support_positions = K * int(enc.spacetime_supernodes)
    else:
        support_positions = K * L * per_frame if bool(enc.use_support_tokens) else 0
    query_positions = T_obs * per_frame
    traj_positions = traj_len if bool(enc.use_support_tokens) and bool(enc.use_traj_tokens) else 0
    traj_demo_positions = K if bool(enc.use_support_tokens) and bool(enc.use_traj_tokens) else 0
    return max(1, support_positions, query_positions, traj_positions, traj_demo_positions)


def policy_config_from(cfg: Any, H: int, data_cfg: Any = None) -> PolicyConfig:
    enc = encoder_config_from(get_value(cfg, 'encoder', None))
    if int(enc.max_positions) <= 0:
        enc = replace(enc, max_positions=_infer_max_positions(enc, data_cfg))
    dec = decoder_config_from(get_value(cfg, 'decoder', None), d_model=enc.d_model, n_heads=enc.n_heads)
    conditioning = conditioning_config_from(get_value(cfg, 'conditioning', None))
    return PolicyConfig(encoder=enc, decoder=dec, conditioning=conditioning, H=int(H))
