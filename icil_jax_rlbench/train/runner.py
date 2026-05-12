from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from absl import logging
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
import numpy as np
import optax

from icil_jax_rlbench.data.action_representation import decode_action_chunk
from icil_jax_rlbench.data.h5_cache import RLBenchCacheStore, build_keys
from icil_jax_rlbench.data.sampler import ICILDataConfig, ICILSampler, shard_batch
from icil_jax_rlbench.models.config import policy_config_from
from icil_jax_rlbench.models.direct_regression_policy import DirectRegressionPolicy
from icil_jax_rlbench.train.checkpoints import load_checkpoint, save_checkpoint
from icil_jax_rlbench.train.trajectory_plots import ActionChunkPlotData, make_action_chunk_figures
from icil_jax_rlbench.train.step import (
    StepConfig,
    TrainState,
    create_memory_maml_step,
    create_param_maml_step,
    create_pretrain_step,
    make_name_mask,
)


def _maybe_wandb(cfg: ConfigDict):
    if not bool(getattr(cfg.wandb, 'enable', False)):
        return None
    import wandb

    entity = getattr(cfg.wandb, 'entity', os.environ.get('WANDB_ENTITY', '')) or None
    name = getattr(cfg.wandb, 'name', '') or None
    wandb.init(
        project=str(getattr(cfg.wandb, 'project', os.environ.get('WANDB_PROJECT', 'icil-jax-rlbench'))),
        entity=entity,
        name=name,
        mode=str(getattr(cfg.wandb, 'mode', os.environ.get('WANDB_MODE', 'online'))),
        config=cfg.to_dict(),
    )
    return wandb


def _tasks(value: Any) -> Optional[Sequence[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        return None if value == '' else tuple(x.strip() for x in value.split(',') if x.strip())
    if len(value) == 0:
        return None
    return tuple(str(x) for x in value)


def _data_config(cfg: ConfigDict) -> ICILDataConfig:
    return ICILDataConfig(
        K=int(cfg.data.K),
        L=int(cfg.data.L),
        T_obs=int(cfg.data.T_obs),
        H=int(cfg.data.H),
        stride=int(cfg.data.stride),
        action_representation=str(cfg.data.action_representation),
        task_sampling=str(cfg.data.task_sampling),
        task_sampling_alpha=float(cfg.data.task_sampling_alpha),
        traj_len=int(cfg.data.traj_len),
    )


def _step_config(cfg: ConfigDict) -> StepConfig:
    return StepConfig(
        loss_type=str(cfg.train.loss_type),
        grad_clip_norm=float(cfg.train.grad_clip_norm),
        first_order=bool(getattr(cfg.maml, 'first_order', True)),
        inner_lr=float(getattr(cfg.maml, 'inner_lr', 1e-2)),
        inner_grad_clip_norm=float(getattr(cfg.maml, 'inner_grad_clip_norm', 1.0)),
        memory_grad_clip_norm=float(getattr(cfg.maml, 'memory_grad_clip_norm', 1.0)),
        memory_update_clip_norm=float(getattr(cfg.maml, 'memory_update_clip_norm', 0.0)),
        inner_param_include=tuple(getattr(cfg.maml, 'inner_param_include', ())),
        inner_param_exclude=tuple(getattr(cfg.maml, 'inner_param_exclude', ())),
        log_attention_stats=bool(getattr(cfg.train, 'log_attention_stats', False)),
    )


def _init_state(model: DirectRegressionPolicy, init_batch: Dict[str, Any], cfg: ConfigDict, seed: int) -> TrainState:
    rng = jax.random.PRNGKey(int(seed))
    variables = model.init({'params': rng, 'dropout': rng}, init_batch, train=True)
    tx = optax.adamw(learning_rate=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay))
    state = TrainState.create(apply_fn=None, params=variables['params'], tx=tx, rng=rng)
    resume = str(getattr(cfg.train, 'resume_path', '') or '')
    if resume:
        ckpt = load_checkpoint(resume)
        state = state.replace(params=ckpt['params'])
        if bool(getattr(cfg.train, 'resume_optimizer', True)) and 'opt_state' in ckpt:
            state = state.replace(opt_state=ckpt['opt_state'])
        if bool(getattr(cfg.train, 'resume_rng', True)) and 'rng' in ckpt:
            state = state.replace(rng=ckpt['rng'])
        logging.info('Resumed checkpoint: %s', resume)
    return state


def _metric_value(x):
    x = jax.device_get(x)
    if hasattr(x, 'shape') and len(x.shape) > 0:
        x = x[0]
    return float(x)


def _log_metrics(prefix: str, step: int, metrics: Dict[str, Any], wandb_mod=None) -> None:
    flat = {f'{prefix}/{k}': _metric_value(v) for k, v in metrics.items()}
    pieces = ' | '.join(f'{k.split("/", 1)[1]} {v:.6f}' for k, v in flat.items())
    logging.info('step %d | %s', step, pieces)
    if wandb_mod is not None:
        wandb_mod.log({**flat, f'{prefix}/step': step}, step=step)


def _unreplicate_params(replicated_state: TrainState) -> Any:
    return jax.tree_util.tree_map(lambda x: jax.device_get(x[0]), replicated_state.params)


def _replicate_to_local_devices(tree: Any, num_devices: int) -> Any:
    def _replicate_leaf(x):
        x = jnp.asarray(x)
        return jax.device_put(jnp.broadcast_to(x, (num_devices,) + tuple(x.shape)))

    return jax.tree_util.tree_map(_replicate_leaf, tree)



def _prediction_examples(
    *,
    split: str,
    batch: Dict[str, np.ndarray],
    pred: np.ndarray,
    action_representation: str,
    num_plots: int,
) -> Sequence[ActionChunkPlotData]:
    target = np.asarray(batch['target_action'], dtype=np.float32)
    query_state = np.asarray(batch['query_state'], dtype=np.float32)
    pred_abs = decode_action_chunk(pred, query_state=query_state, representation=action_representation)
    target_abs = decode_action_chunk(target, query_state=query_state, representation=action_representation)
    count = min(int(num_plots), int(pred.shape[0]))
    out = []
    for i in range(count):
        mse = float(np.mean(np.square(pred[i].astype(np.float32) - target[i].astype(np.float32))))
        out.append(
            ActionChunkPlotData(
                name=f'chunk_{i}',
                split=split,
                pred_xyz=pred_abs[i, :, :3],
                target_xyz=target_abs[i, :, :3],
                mse=mse,
            )
        )
    return out


def _evaluate_prediction_split(
    *,
    split: str,
    sampler: ICILSampler,
    eval_predict,
    params: Any,
    policy_cfg: Any,
    cfg: ConfigDict,
) -> tuple[float, Sequence[ActionChunkPlotData]]:
    num_samples = int(getattr(cfg.wandb, 'prediction_num_samples', 64))
    num_plots = int(getattr(cfg.wandb, 'prediction_num_plots', 4))
    if num_samples <= 0:
        return 0.0, ()
    batch = sampler.build_pretrain_batch(
        num_samples,
        load_rgb=bool(policy_cfg.encoder.use_rgb),
        load_mask_id=bool(policy_cfg.encoder.use_mask_id),
    )
    pred = np.asarray(jax.device_get(eval_predict(params, batch)), dtype=np.float32)
    target = np.asarray(batch['target_action'], dtype=np.float32)
    mse = float(np.mean(np.square(pred - target)))
    examples = _prediction_examples(
        split=split,
        batch=batch,
        pred=pred,
        action_representation=str(cfg.data.action_representation),
        num_plots=num_plots,
    )
    return mse, examples


def _maybe_log_prediction_eval(
    *,
    step: int,
    cfg: ConfigDict,
    wandb_mod,
    replicated_state: TrainState,
    train_eval_sampler: ICILSampler,
    excluded_eval_sampler: Optional[ICILSampler],
    policy_cfg: Any,
    eval_predict,
) -> None:
    if wandb_mod is None:
        return
    every = int(getattr(cfg.wandb, 'prediction_log_every', 0))
    if every <= 0 or step % every != 0:
        return

    params = _unreplicate_params(replicated_state)
    payload: Dict[str, Any] = {}
    plot_examples = []
    train_mse, train_examples = _evaluate_prediction_split(
        split='train',
        sampler=train_eval_sampler,
        eval_predict=eval_predict,
        params=params,
        policy_cfg=policy_cfg,
        cfg=cfg,
    )
    payload['eval/train_mse'] = train_mse
    plot_examples.extend(train_examples)

    if excluded_eval_sampler is not None:
        excluded_mse, excluded_examples = _evaluate_prediction_split(
            split='excluded',
            sampler=excluded_eval_sampler,
            eval_predict=eval_predict,
            params=params,
            policy_cfg=policy_cfg,
            cfg=cfg,
        )
        payload['eval/excluded_mse'] = excluded_mse
        plot_examples.extend(excluded_examples)

    try:
        figures = make_action_chunk_figures(plot_examples)
    except ImportError as exc:
        logging.warning('Skipping prediction trajectory plots because plotly is unavailable: %s', exc)
        figures = {}
    for name, fig in figures.items():
        payload[f'eval/action_chunk/{name}'] = wandb_mod.Plotly(fig) if hasattr(wandb_mod, 'Plotly') else fig
    wandb_mod.log(payload, step=step)


def train(mode: str, cfg: ConfigDict) -> None:
    if mode not in ('pretrain', 'param_maml', 'memory_maml'):
        raise ValueError(f'Unknown mode={mode!r}')
    excluded_tasks = _tasks(getattr(cfg.data, 'exclude_tasks', ())) or ()
    keys, selected_tasks = build_keys(Path(cfg.data.cache_root), tasks=_tasks(getattr(cfg.data, 'tasks', ())), exclude_tasks=excluded_tasks)
    store = RLBenchCacheStore(keys, keep_open=bool(cfg.data.keep_open), preload_to_memory=bool(cfg.data.preload_to_memory))
    if store.preload_to_memory:
        logging.info('Preloaded RLBench cache into RAM: %.2f GiB', store.preloaded_bytes / (1024 ** 3))
    num_points, state_dim, action_dim = store.infer_dims()
    logging.info('RLBench cache: root=%s tasks=%d variations=%d points=%d state_dim=%d action_dim=%d', cfg.data.cache_root, len(selected_tasks), len(keys), num_points, state_dim, action_dim)

    data_cfg = _data_config(cfg)
    sampler = ICILSampler(store, data_cfg, seed=int(cfg.train.seed))
    train_eval_sampler = ICILSampler(store, data_cfg, seed=int(cfg.train.seed) + 100003)
    excluded_eval_sampler = None
    excluded_store = None
    if excluded_tasks:
        excluded_keys, excluded_selected = build_keys(Path(cfg.data.cache_root), tasks=excluded_tasks, exclude_tasks=())
        excluded_store = RLBenchCacheStore(excluded_keys, keep_open=bool(cfg.data.keep_open), preload_to_memory=False)
        excluded_eval_sampler = ICILSampler(excluded_store, data_cfg, seed=int(cfg.train.seed) + 200003)
        logging.info('Prediction eval excluded tasks: tasks=%d variations=%d', len(excluded_selected), len(excluded_keys))
    else:
        logging.info('No data.exclude_tasks configured; skipping excluded-task prediction eval.')
    policy_cfg = policy_config_from(cfg.model, H=data_cfg.H)
    model = DirectRegressionPolicy(policy_cfg, state_dim=state_dim, action_dim=action_dim)
    eval_predict = jax.jit(lambda params, batch: model.apply({'params': params}, batch, train=False))

    init_batch = sampler.build_pretrain_batch(
        max(1, min(int(cfg.train.batch_size), 2)),
        load_rgb=bool(policy_cfg.encoder.use_rgb),
        load_mask_id=bool(policy_cfg.encoder.use_mask_id),
    )
    state = _init_state(model, init_batch, cfg, int(cfg.train.seed))
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    logging.info('Model parameters: %.3f M', n_params / 1e6)

    num_devices = jax.local_device_count()
    if int(cfg.train.batch_size) % num_devices != 0:
        raise ValueError(f'train.batch_size={cfg.train.batch_size} must be divisible by local_device_count={num_devices}.')
    replicated_state = _replicate_to_local_devices(state, num_devices)
    step_cfg = _step_config(cfg)
    if mode == 'pretrain':
        p_train_step = create_pretrain_step(model, step_cfg)
    elif mode == 'param_maml':
        inner_mask = make_name_mask(state.params, include=step_cfg.inner_param_include, exclude=step_cfg.inner_param_exclude)
        inner_mask = _replicate_to_local_devices(inner_mask, num_devices)
        # The mask is static in value but a pytree constant in the closure; use host copy for pmap tracing.
        inner_mask_host = jax.tree_util.tree_map(lambda x: bool(jax.device_get(x[0])) if hasattr(x, 'shape') and x.shape else bool(jax.device_get(x)), inner_mask)
        p_train_step = create_param_maml_step(model, step_cfg, inner_mask=inner_mask_host)
    else:
        p_train_step = create_memory_maml_step(model, step_cfg)

    wandb_mod = _maybe_wandb(cfg)
    ckpt_dir = Path(cfg.train.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_time = time.time()
    for step in range(1, int(cfg.train.num_steps) + 1):
        t0 = time.time()
        if mode == 'pretrain':
            batch = sampler.build_pretrain_batch(int(cfg.train.batch_size), load_rgb=bool(policy_cfg.encoder.use_rgb), load_mask_id=bool(policy_cfg.encoder.use_mask_id))
        elif mode == 'param_maml':
            batch = sampler.build_param_maml_batch(
                int(cfg.train.batch_size),
                inner_steps=int(cfg.maml.inner_steps),
                num_inner_queries=int(cfg.maml.num_inner_queries),
                num_query_loss_samples=int(cfg.maml.num_query_loss_samples),
                load_rgb=bool(policy_cfg.encoder.use_rgb),
                load_mask_id=bool(policy_cfg.encoder.use_mask_id),
            )
        else:
            batch = sampler.build_memory_maml_batch(
                int(cfg.train.batch_size),
                inner_steps=int(cfg.maml.inner_steps),
                num_inner_queries=int(cfg.maml.num_inner_queries),
                num_query_loss_samples=int(cfg.maml.num_query_loss_samples),
                load_rgb=bool(policy_cfg.encoder.use_rgb),
                load_mask_id=bool(policy_cfg.encoder.use_mask_id),
            )
        data_wait_s = time.time() - t0
        replicated_state, metrics = p_train_step(replicated_state, shard_batch(batch, num_devices))
        if step == 1:
            jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x, metrics)
        now = time.time()
        if step == 1 or step % int(cfg.train.log_every) == 0:
            metrics = dict(metrics)
            metrics['data_wait_s'] = jnp.asarray(data_wait_s, dtype=jnp.float32)
            metrics['step_s'] = jnp.asarray((now - last_time) / max(1, int(cfg.train.log_every)), dtype=jnp.float32)
            _log_metrics('train', step, metrics, wandb_mod)
            last_time = now
        _maybe_log_prediction_eval(
            step=step,
            cfg=cfg,
            wandb_mod=wandb_mod,
            replicated_state=replicated_state,
            train_eval_sampler=train_eval_sampler,
            excluded_eval_sampler=excluded_eval_sampler,
            policy_cfg=policy_cfg,
            eval_predict=eval_predict,
        )
        if step % int(cfg.train.ckpt_every) == 0:
            path = ckpt_dir / f'step_{step:07d}.pkl'
            save_checkpoint(path, state=replicated_state, step=step, config=cfg, replicated=True)
            logging.info('Saved checkpoint: %s', path)
    path = ckpt_dir / 'last.pkl'
    save_checkpoint(path, state=replicated_state, step=int(cfg.train.num_steps), config=cfg, replicated=True)
    logging.info('Training complete. Final checkpoint: %s', path)
    if wandb_mod is not None:
        wandb_mod.finish()
    if excluded_store is not None:
        excluded_store.close()
