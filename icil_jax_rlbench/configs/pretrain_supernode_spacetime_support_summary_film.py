from icil_jax_rlbench.configs.pretrain_supernode_spacetime import get_config as _spacetime


def get_config():
    cfg = _spacetime()
    cfg.model.conditioning.mode = 'support_summary_film'
    cfg.model.conditioning.support_summary_source = 'traj_and_memory'
    cfg.model.decoder.context_mode = 'query_film_support'
    cfg.model.decoder.support_cross_layers = 2
    cfg.model.decoder.film_mlp_mult = 4
    return cfg
