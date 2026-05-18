from icil_jax_rlbench.configs.base import get_config as _base


def get_config():
    cfg = _base(mode='pretrain', encoder_type='supernode')
    cfg.data.K = 1
    cfg.data.L = 1
    cfg.data.traj_len = 0
    cfg.model.encoder.support_layers = 0
    cfg.model.encoder.traj_layers = 0
    cfg.model.encoder.support_num_latents = 0
    cfg.model.encoder.use_support_tokens = False
    cfg.model.encoder.use_traj_tokens = False
    cfg.model.encoder.query_num_latents = 64
    cfg.model.conditioning.mode = 'task_variation'
    cfg.model.conditioning.num_task_tokens = 1
    cfg.model.conditioning.num_variation_tokens = 1
    cfg.model.decoder.context_mode = 'single_ctx'
    return cfg
