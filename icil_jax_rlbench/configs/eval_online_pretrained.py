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
    cfg.dataset.query_stride_mode = 'consecutive' # 'dataset' or 'consecutive'

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

    cfg.control = ConfigDict()
    cfg.control.execute_actions_per_plan = 16
    cfg.control.normalize_quaternion = True
    cfg.control.discretize_gripper = True

    cfg.video = ConfigDict()
    cfg.video.enable = True
    cfg.video.camera = 'front'
    cfg.video.fps = 10
    cfg.video.format = 'mp4'

    cfg.output = ConfigDict()
    cfg.output.root_dir = os.environ.get('ICIL_EVAL_OUTPUT_DIR', 'eval_outputs')

    cfg.wandb = ConfigDict()
    cfg.wandb.enable = False
    cfg.wandb.project = os.environ.get('WANDB_PROJECT', 'icil-jax-rlbench-eval')
    cfg.wandb.entity = os.environ.get('WANDB_ENTITY', '')
    cfg.wandb.name = ''
    cfg.wandb.mode = os.environ.get('WANDB_MODE', 'online')

    return cfg
