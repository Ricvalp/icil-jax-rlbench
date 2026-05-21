from icil_jax_rlbench.configs.base import get_config as _base


def get_config():
    cfg = _base(mode='param_maml', encoder_type='supernode')
    cfg.model.conditioning.mode = 'support_summary_film'
    cfg.model.conditioning.support_summary_source = 'traj_and_memory'
    cfg.model.decoder.context_mode = 'query_film_support'
    cfg.model.decoder.support_cross_layers = 2
    cfg.model.decoder.film_mlp_mult = 4
    cfg.maml.fast_param_preset = 'film_top'
    cfg.maml.fast_param_top_layers = 2
    return cfg
