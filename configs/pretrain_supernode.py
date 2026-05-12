from icil_jax_rlbench.configs.base import get_config as _base

def get_config():
    return _base(mode='pretrain', encoder_type='supernode')
