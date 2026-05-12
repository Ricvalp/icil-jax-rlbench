from icil_jax_rlbench.configs.base import get_config as _base

def get_config():
    return _base(mode='memory_maml', encoder_type='supernode')
