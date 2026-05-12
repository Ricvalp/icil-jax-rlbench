from __future__ import annotations

from typing import Any

from .direct_regression_policy import DecoderConfig, PolicyConfig
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


def policy_config_from(cfg: Any, H: int) -> PolicyConfig:
    enc = encoder_config_from(get_value(cfg, 'encoder', None))
    dec = decoder_config_from(get_value(cfg, 'decoder', None), d_model=enc.d_model, n_heads=enc.n_heads)
    return PolicyConfig(encoder=enc, decoder=dec, H=int(H))
