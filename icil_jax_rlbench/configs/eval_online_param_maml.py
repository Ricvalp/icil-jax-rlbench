from __future__ import annotations

from ml_collections import ConfigDict

from icil_jax_rlbench.configs.eval_online_pretrained import get_config as _base


def get_config():
    cfg = _base()
    cfg.task.max_env_steps = 220
    cfg.dataset.query_stride_mode = 'dataset'
    cfg.control.execute_actions_per_plan = 8

    cfg.adaptation = ConfigDict()
    cfg.adaptation.inner_steps_override = -1
    cfg.adaptation.num_inner_queries = 0
    cfg.adaptation.inner_lr = 0.0
    cfg.adaptation.grad_clip_norm = 0.0
    cfg.adaptation.regenerate_each_episode = False

    return cfg
