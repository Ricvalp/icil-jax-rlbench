from icil_jax_rlbench.configs.param_maml_supernode_spacetime import get_config as _spacetime


def get_config():
    cfg = _spacetime()
    cfg.model.conditioning.mode = 'support_summary_film'
    cfg.model.conditioning.support_summary_source = 'traj_and_memory'
    cfg.model.decoder.context_mode = 'query_film_support'
    cfg.model.decoder.support_cross_layers = 2
    cfg.model.decoder.film_mlp_mult = 4
    cfg.maml.fast_param_preset = 'film_top'
    cfg.maml.fast_param_top_layers = 2
    cfg.maml.inner_param_include = ()
    cfg.maml.inner_param_exclude = ()
    return cfg
