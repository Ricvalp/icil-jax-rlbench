from __future__ import annotations

import os

from ml_collections import ConfigDict


def get_config():
    cfg = ConfigDict()

    cfg.seed = 0
    cfg.checkpoint_path = ''

    cfg.task = ConfigDict()
    cfg.task.name = 'push_button'
    cfg.task.variation = 0
    cfg.task.num_eval_episodes = 10
    cfg.task.max_env_steps = 80

    cfg.dataset = ConfigDict()
    cfg.dataset.use_checkpoint_dataset_config = True
    cfg.dataset.K = 2
    cfg.dataset.L = 8
    cfg.dataset.T_obs = 2
    cfg.dataset.H = 16
    cfg.dataset.stride = 2
    cfg.dataset.traj_len = 512
    cfg.dataset.action_representation = 'absolute'
    cfg.dataset.query_stride_mode = 'dataset' # 'dataset' or 'consecutive'
    cfg.dataset.support_spacetime_points = 0
    cfg.dataset.support_spacetime_sampling = 'mask_balanced'

    cfg.conditioning = ConfigDict()
    cfg.conditioning.cache_root = os.environ.get('ICIL_CACHE_ROOT', '')
    cfg.conditioning.regenerate_demos_each_episode = False
    cfg.conditioning.use_rgb = True
    cfg.conditioning.use_mask_id = False
    cfg.conditioning.num_points = 0
    cfg.conditioning.filter_workspace_bounds = True
    cfg.conditioning.workspace_bounds = ((-1.0, 1.0), (-1.0, 1.0), (0.0, 2.5))

    cfg.sim = ConfigDict()
    cfg.sim.headless = True
    cfg.sim.renderer = 'opengl'
    cfg.sim.image_size = (128, 128)
    cfg.sim.arm_max_velocity = 1.0
    cfg.sim.arm_max_acceleration = 4.0
    cfg.sim.collision_checking = False
    cfg.sim.static_positions = False

    cfg.control = ConfigDict()
    cfg.control.execute_actions_per_plan = 8
    cfg.control.normalize_quaternion = True
    cfg.control.discretize_gripper = True
    cfg.control.reject_out_of_bounds_actions = True
    cfg.control.action_position_bounds = ((-1.0, 1.0), (-1.0, 1.0), (0.0, 2.5))
    cfg.control.max_position_delta = 0.0

    cfg.video = ConfigDict()
    cfg.video.enable = True
    cfg.video.camera = 'front'
    cfg.video.fps = 10
    cfg.video.format = 'mp4'

    cfg.ground_truth_video = ConfigDict()
    cfg.ground_truth_video.enable = True
    cfg.ground_truth_video.num_demos = 1
    cfg.ground_truth_video.camera = ''
    cfg.ground_truth_video.fps = 10
    cfg.ground_truth_video.format = 'mp4'
    cfg.ground_truth_video.max_attempts = 10

    cfg.action_chunk_viz = ConfigDict()
    cfg.action_chunk_viz.enable = False
    cfg.action_chunk_viz.every_n_plans = 1
    cfg.action_chunk_viz.max_plots_per_episode = 16
    cfg.action_chunk_viz.edge_top_k = 8
    cfg.action_chunk_viz.max_edge_supernodes = 64
    cfg.action_chunk_viz.skip_self_edges = False
    cfg.action_chunk_viz.edge_min_length = 0.0
    cfg.action_chunk_viz.edge_candidate_multiplier = 1
    cfg.action_chunk_viz.edge_line_width = 3.0
    cfg.action_chunk_viz.edge_opacity = 1.0
    cfg.action_chunk_viz.marker_size = 1.5

    cfg.output = ConfigDict()
    cfg.output.root_dir = os.environ.get('ICIL_EVAL_OUTPUT_DIR', 'eval_outputs')

    cfg.wandb = ConfigDict()
    cfg.wandb.enable = False
    cfg.wandb.project = os.environ.get('WANDB_PROJECT', 'icil-jax-rlbench-eval')
    cfg.wandb.entity = os.environ.get('WANDB_ENTITY', '')
    cfg.wandb.name = ''
    cfg.wandb.mode = os.environ.get('WANDB_MODE', 'online')

    return cfg
