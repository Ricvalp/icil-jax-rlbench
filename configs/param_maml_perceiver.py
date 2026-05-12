from icil_jax_rlbench.configs.base import get_config as _base

def get_config():
    return _base(mode='param_maml', encoder_type='perceiver')
