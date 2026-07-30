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

    cfg.query = ConfigDict()
    cfg.query.episodes = ''
    cfg.query.t0s = '0'
    cfg.query.stride_mode = 'dataset'  # dataset | consecutive
    cfg.query.window_mode = 'checkpoint'  # checkpoint | online_history | forward

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

    cfg.adaptation = ConfigDict()
    cfg.adaptation.inner_steps_override = -1
    cfg.adaptation.num_inner_queries = 0
    cfg.adaptation.inner_lr = 0.0
    cfg.adaptation.grad_clip_norm = 0.0

    cfg.plot = ConfigDict()
    cfg.plot.marker_size = 1.5
    cfg.plot.point_opacity = 0.72
    cfg.plot.show_supernodes = True
    cfg.plot.edge_top_k = 8
    cfg.plot.max_edge_supernodes = 64
    cfg.plot.skip_self_edges = True
    cfg.plot.edge_line_width = 2.5
    cfg.plot.edge_opacity = 0.55
    cfg.plot.edge_min_length = 0.0
    cfg.plot.edge_candidate_multiplier = 1

    cfg.output = ConfigDict()
    cfg.output.root_dir = os.environ.get('ICIL_EVAL_OUTPUT_DIR', 'eval_outputs/param_maml_adaptation_action_chunks')

    return cfg
