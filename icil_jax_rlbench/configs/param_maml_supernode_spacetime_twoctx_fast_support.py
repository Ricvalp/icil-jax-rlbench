from icil_jax_rlbench.configs.param_maml_supernode_spacetime import get_config as _spacetime


def get_config():
    cfg = _spacetime()
    cfg.maml.fast_param_preset = 'name'
    cfg.maml.fast_param_top_layers = 2
    cfg.maml.inner_param_include = (
        'decoder/support_cross_2',
        'decoder/support_cross_3',
        'decoder/action_head',
    )
    cfg.maml.inner_param_exclude = ()
    return cfg
