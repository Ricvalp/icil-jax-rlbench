from __future__ import annotations

import os
from ml_collections import ConfigDict


def get_config(mode: str = 'pretrain', encoder_type: str = 'perceiver') -> ConfigDict:
    cfg = ConfigDict()
    cfg.mode = mode

    cfg.data = ConfigDict()
    cfg.data.cache_root = os.environ.get('ICIL_CACHE_ROOT', '/mnt/external_storage/robotics/rlbench/icil_rlbench/.rlbench_cache_dense_v4')
    cfg.data.tasks = ()
    cfg.data.exclude_tasks = (
        'slide_block_to_target',
        'close_laptop_lid',
        'close_box',
        'open_jar',
        'toilet_seat_up',
        'push_button',
        'basketball_in_hoop',
        'meat_on_grill',
        'put_umbrella_in_umbrella_stand',
        'lamp_on',
    )
    cfg.data.K = 4
    cfg.data.L = 10
    cfg.data.T_obs = 2
    cfg.data.H = 16
    cfg.data.stride = 2
    cfg.data.action_representation = 'absolute'
    cfg.data.task_sampling = 'task_uniform'
    cfg.data.task_sampling_alpha = 1.0
    cfg.data.traj_len = 64
    cfg.data.keep_open = True
    cfg.data.preload_to_memory = False

    cfg.model = ConfigDict()
    cfg.model.encoder = ConfigDict()
    cfg.model.encoder.encoder_type = encoder_type
    cfg.model.encoder.d_model = 256
    cfg.model.encoder.n_heads = 4
    cfg.model.encoder.dropout = 0.0
    cfg.model.encoder.mlp_mult = 4
    cfg.model.encoder.frame_num_latents = 8
    cfg.model.encoder.frame_layers = 2
    cfg.model.encoder.query_num_latents = 16
    cfg.model.encoder.support_num_latents = 64
    cfg.model.encoder.support_layers = 2
    cfg.model.encoder.query_layers = 1
    cfg.model.encoder.supernodes = 64
    cfg.model.encoder.supernode_temperature = 0.02
    cfg.model.encoder.supernode_layers = 2
    cfg.model.encoder.traj_layers = 1
    cfg.model.encoder.max_positions = 0
    cfg.model.encoder.mask_id_vocab = 256
    cfg.model.encoder.use_rgb = True
    cfg.model.encoder.use_mask_id = False
    cfg.model.encoder.use_traj_tokens = True

    cfg.model.decoder = ConfigDict()
    cfg.model.decoder.n_layers = 4
    cfg.model.decoder.context_mode = 'single_ctx'  # single_ctx | two_ctx
    cfg.model.decoder.mlp_mult = 4
    cfg.model.decoder.dropout = 0.0

    cfg.train = ConfigDict()
    cfg.train.seed = 0
    cfg.train.num_steps = 100000
    cfg.train.batch_size = 8
    cfg.train.lr = 1e-4
    cfg.train.weight_decay = 1e-4
    cfg.train.loss_type = 'mse'
    cfg.train.log_attention_stats = False
    cfg.train.grad_clip_norm = 1.0
    cfg.train.log_every = 100
    cfg.train.ckpt_every = 10000
    cfg.train.prefetch_workers = 2
    cfg.train.prefetch_batches = 2
    cfg.train.resume_path = ''
    cfg.train.resume_optimizer = True
    cfg.train.resume_rng = True
    cfg.train.checkpoint_dir = os.path.join(
        os.environ.get('ICIL_JAX_RLBENCH_CHECKPOINT_DIR', '/mnt/external_storage/robotics/rlbench/icil_jax_rlbench_runs/checkpoints'),
        mode,
    )

    cfg.maml = ConfigDict()
    cfg.maml.inner_steps = 1
    cfg.maml.num_inner_queries = 4
    cfg.maml.num_query_loss_samples = 4
    cfg.maml.inner_lr = 1e-2
    cfg.maml.first_order = False
    cfg.maml.inner_grad_clip_norm = 1.0
    cfg.maml.memory_grad_clip_norm = 1.0
    cfg.maml.memory_update_clip_norm = 0.0
    cfg.maml.inner_param_include = ()
    cfg.maml.inner_param_exclude = ()

    cfg.wandb = ConfigDict()
    cfg.wandb.enable = False
    cfg.wandb.project = os.environ.get('ICIL_JAX_RLBENCH_WANDB_PROJECT', 'icil-jax-rlbench')
    cfg.wandb.entity = os.environ.get('WANDB_ENTITY', '')
    cfg.wandb.mode = os.environ.get('WANDB_MODE', 'online')
    cfg.wandb.name = ''
    cfg.wandb.prediction_log_every = 200
    cfg.wandb.prediction_num_samples = 64
    cfg.wandb.prediction_num_plots = 4

    if mode in ('param_maml', 'memory_maml'):
        cfg.train.batch_size = 4
        cfg.train.ckpt_every = 5000
        cfg.train.checkpoint_dir = os.path.join(
            os.environ.get('ICIL_JAX_RLBENCH_CHECKPOINT_DIR', '/mnt/external_storage/robotics/rlbench/icil_jax_rlbench_runs/checkpoints'),
            mode,
        )
    return cfg
