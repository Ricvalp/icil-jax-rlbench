from icil_jax_rlbench.configs.base import get_config as _base


def get_config():
    cfg = _base(mode='param_maml', encoder_type='supernode')
    cfg.model.encoder.support_tokenizer = 'spacetime_supernode'
    cfg.model.encoder.spacetime_supernodes = 256
    cfg.model.encoder.spacetime_temperature_xyz = 0.005
    cfg.model.encoder.spacetime_temperature_t = 0.04
    cfg.model.encoder.spacetime_layers = 2
    cfg.data.L = 30
    cfg.data.support_spacetime_points = 8192
    cfg.data.support_spacetime_sampling = 'mask_balanced'
    return cfg
