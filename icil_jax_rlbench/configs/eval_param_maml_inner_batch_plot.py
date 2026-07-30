from __future__ import annotations

import os

from ml_collections import ConfigDict


def get_config():
    cfg = ConfigDict()

    cfg.seed = 0
    cfg.checkpoint_path = ''

    cfg.task = ConfigDict()
    cfg.task.name = 'eval_contextual_slide_pick_sequence'
    cfg.task.variation = 0

    cfg.support = ConfigDict()
    cfg.support.episodes = ''

    cfg.dataset = ConfigDict()
    cfg.dataset.use_checkpoint_dataset_config = True
    cfg.dataset.K = 2
    cfg.dataset.L = 8
    cfg.dataset.T_obs = 2
    cfg.dataset.H = 16
    cfg.dataset.stride = 2
    cfg.dataset.traj_len = 512
    cfg.dataset.action_representation = 'absolute'
    cfg.dataset.support_spacetime_points = 0
    cfg.dataset.support_spacetime_sampling = 'mask_balanced'

    cfg.conditioning = ConfigDict()
    cfg.conditioning.cache_root = os.environ.get('ICIL_CACHE_ROOT', '')
    cfg.conditioning.keep_open = True

    cfg.inner_batch = ConfigDict()
    cfg.inner_batch.batch_sizes = '4,64'
    cfg.inner_batch.step_index = 0

    cfg.plot = ConfigDict()
    cfg.plot.max_points_per_sample = 0
    cfg.plot.marker_size = 1.0
    cfg.plot.point_opacity = 0.18
    cfg.plot.current_marker_size = 4.0
    cfg.plot.action_marker_size = 2.5
    cfg.plot.action_line_width = 4.0
    cfg.plot.support_line_width = 5.0
    cfg.plot.support_opacity = 0.35

    cfg.output = ConfigDict()
    cfg.output.root_dir = os.environ.get('ICIL_EVAL_OUTPUT_DIR', 'eval_outputs/param_maml_inner_batch_plots')

    return cfg
