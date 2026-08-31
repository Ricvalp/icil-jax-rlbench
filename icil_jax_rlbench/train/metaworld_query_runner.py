from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from ml_collections import ConfigDict

from icil_jax_rlbench.data.metaworld_ml1_reach import (
    ML1ReachTaskDataset,
    ML1ReachTaskSampler,
)
from icil_jax_rlbench.models.fast_weight_ttt import (
    FastWeightTTTConfig,
    init_fast_weight_ttt_params,
)
from icil_jax_rlbench.train.checkpoints import load_checkpoint, save_checkpoint
from icil_jax_rlbench.train.provenance import (
    collect_experiment_provenance,
    config_to_dict,
    write_experiment_ledger,
)
from icil_jax_rlbench.train.query_only_step import (
    create_query_only_train_step,
    query_only_objective,
)
from icil_jax_rlbench.train.ttt_step import create_ttt_train_state

CHECKPOINT_TYPE = 'metaworld_ml1_reach_query_only'
_LOGGER = logging.getLogger(__name__)


def validate_metaworld_query_config(cfg: ConfigDict) -> None:
    if str(cfg.mode) != 'metaworld_ml1_reach_query_only':
        raise ValueError("Expected mode='metaworld_ml1_reach_query_only'.")
    if not str(cfg.dataset.cache_root):
        raise ValueError(
            'dataset.cache_root is required. Point it at a processed phi-mujoco cache.'
        )
    if str(cfg.action.translation_loss) != 'huber':
        raise ValueError('ML1 Reach uses Huber loss for Cartesian actions.')
    if str(cfg.action.gripper_loss) != 'huber':
        raise ValueError('ML1 Reach uses Huber loss for its continuous gripper action.')
    if int(cfg.train.batch_size) < 1:
        raise ValueError('train.batch_size must be positive.')
    if int(cfg.train.query_episodes_per_task) < 1:
        raise ValueError('train.query_episodes_per_task must be positive.')


def metaworld_model_config_from(
    cfg: ConfigDict,
    dataset: ML1ReachTaskDataset,
) -> FastWeightTTTConfig:
    return FastWeightTTTConfig(
        observation_dim=int(dataset.observation_dim),
        action_dim=int(dataset.action_dim),
        translation_dim=3,
        hidden_dim=int(cfg.model.hidden_dim),
        fast_dim=int(cfg.model.fast_dim),
        fast_hidden_dim=int(cfg.model.fast_hidden_dim),
        fast_model=str(cfg.model.fast_model),
        gate_init=float(cfg.model.gate_init),
        inner_lr_init=float(cfg.model.inner_lr_init),
        inner_lr_min=float(cfg.model.inner_lr_min),
        translation_output='linear',
        translation_loss_weight=float(cfg.action.translation_loss_weight),
        gripper_loss_weight=float(cfg.action.gripper_loss_weight),
        translation_huber_delta=float(cfg.action.translation_huber_delta),
        gripper_loss='huber',
        gripper_huber_delta=float(cfg.action.gripper_huber_delta),
    )


def _weight_decay_mask(params: Mapping[str, Any]) -> Mapping[str, Any]:
    trained_groups = {'query_encoder', 'translation_head', 'gripper_head'}

    def make(path, value):
        top_level = str(getattr(path[0], 'key', path[0])) if path else ''
        return bool(top_level in trained_groups and getattr(value, 'ndim', 0) >= 2)

    return jax.tree_util.tree_map_with_path(make, params)


def _optimizer(cfg: ConfigDict, params: Mapping[str, Any]):
    return optax.adamw(
        learning_rate=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
        mask=_weight_decay_mask(params),
    )


def _maybe_wandb(cfg: ConfigDict):
    if not bool(cfg.wandb.enable):
        return None
    import wandb

    kwargs = {
        'project': str(cfg.wandb.project),
        'config': config_to_dict(cfg),
        'mode': str(cfg.wandb.mode),
    }
    if str(cfg.wandb.entity):
        kwargs['entity'] = str(cfg.wandb.entity)
    if str(cfg.wandb.name):
        kwargs['name'] = str(cfg.wandb.name)
    wandb.init(**kwargs)
    return wandb


def _run_id(wandb_mod) -> str:
    if wandb_mod is not None and wandb_mod.run is not None:
        return str(wandb_mod.run.id)
    return f'ml1_reach_query_only_{datetime.now(UTC).strftime("%Y%m%d-%H%M%S")}'


def _write_json(path: Path, value: Any) -> None:
    with path.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write('\n')


def _host_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {
        name: float(np.asarray(value))
        for name, value in jax.device_get(metrics).items()
    }


def _query_to_jax(query: Mapping[str, np.ndarray]) -> dict[str, jax.Array]:
    return {
        name: jnp.asarray(query[name])
        for name in ('observation', 'action', 'outer_loss_mask')
    }


def _log_metrics(
    prefix: str,
    step: int,
    metrics: Mapping[str, float],
    wandb_mod,
) -> None:
    selected = {
        name: value
        for name, value in metrics.items()
        if name
        in {
            'loss',
            'translation_loss',
            'gripper_loss',
            'translation_l1',
            'gripper_l1',
            'slow_grad_norm',
            'step_s',
        }
    }
    _LOGGER.info(
        '%s step %d | %s',
        prefix,
        int(step),
        ' | '.join(
            f'{name} {value:.6f}' for name, value in sorted(selected.items())
        ),
    )
    if wandb_mod is not None:
        wandb_mod.log(
            {f'{prefix}/{name}': value for name, value in metrics.items()},
            step=int(step),
        )


def _restore_state(
    checkpoint_path: str,
    optimizer,
    dataset: ML1ReachTaskDataset,
    model_cfg: FastWeightTTTConfig,
):
    payload = load_checkpoint(checkpoint_path)
    extra = payload.get('extra', {})
    if extra.get('checkpoint_type') != CHECKPOINT_TYPE:
        raise ValueError('Resume checkpoint is not an ML1 Reach query-only checkpoint.')
    if extra.get('cache_data_sha256') != dataset.bundle.data_sha256:
        raise ValueError('Resume checkpoint was trained from a different processed cache.')
    if extra.get('normalizer_id') != dataset.normalization_id:
        raise ValueError('Resume checkpoint normalization differs from the current cache.')
    if extra.get('model_config') != asdict(model_cfg):
        raise ValueError('Resume checkpoint model differs from the requested model config.')
    return create_ttt_train_state(
        jax.tree_util.tree_map(jnp.asarray, payload['params']),
        optimizer,
        jnp.asarray(payload['rng']),
        step=int(payload['step']),
        opt_state=jax.tree_util.tree_map(jnp.asarray, payload['opt_state']),
    )


def _save_state(
    path: Path,
    state,
    *,
    step: int,
    cfg: ConfigDict,
    dataset: ML1ReachTaskDataset,
    model_cfg: FastWeightTTTConfig,
    provenance: Mapping[str, Any],
) -> None:
    save_checkpoint(
        path,
        state=state,
        step=int(step),
        config=cfg,
        extra={
            'checkpoint_type': CHECKPOINT_TYPE,
            'normalization': dataset.normalization.to_dict(),
            'normalizer_id': dataset.normalization_id,
            'cache_data_sha256': dataset.bundle.data_sha256,
            'model_config': asdict(model_cfg),
            'experiment_id': provenance['experiment_id'],
            'transient_fast_state_saved': False,
        },
        replicated=False,
    )


def train_metaworld_query_only(cfg: ConfigDict) -> Path:
    validate_metaworld_query_config(cfg)
    dataset = ML1ReachTaskDataset(
        str(cfg.dataset.cache_root),
        normalization_eps=float(cfg.dataset.normalization_eps),
        cache_prepared_episodes=bool(cfg.dataset.cache_prepared_episodes),
    )
    integrity = dataset.integrity_report()
    if not bool(integrity['normalizer_uses_exact_training_task_episodes']):
        raise RuntimeError('Train-task-only normalization check failed.')
    if not bool(integrity['unique_episode_seeds']):
        raise RuntimeError('Cache contains duplicate episode seeds.')
    if float(integrity['expert_success_rate']) != 1.0:
        raise RuntimeError('Cache contains unsuccessful expert episodes.')

    model_cfg = metaworld_model_config_from(cfg, dataset)
    params = init_fast_weight_ttt_params(
        jax.random.key(int(cfg.train.seed)), model_cfg
    )
    optimizer = _optimizer(cfg, params)
    state = create_ttt_train_state(
        params, optimizer, jax.random.key(int(cfg.train.seed) + 1)
    )
    if str(cfg.train.resume_path):
        checkpoint = load_checkpoint(str(cfg.train.resume_path))
        checkpoint_normalization = checkpoint.get('extra', {}).get('normalization')
        if checkpoint_normalization is None:
            raise ValueError('Resume checkpoint does not contain normalization statistics.')
        dataset = ML1ReachTaskDataset(
            str(cfg.dataset.cache_root),
            normalization=checkpoint_normalization,
            cache_prepared_episodes=bool(cfg.dataset.cache_prepared_episodes),
        )
        state = _restore_state(
            str(cfg.train.resume_path), optimizer, dataset, model_cfg
        )

    train_step = create_query_only_train_step(
        optimizer,
        model_cfg,
        slow_grad_clip_norm=float(cfg.train.slow_grad_clip_norm),
    )
    train_sampler = ML1ReachTaskSampler(
        dataset, split='train', seed=int(cfg.train.seed) + 1001
    )
    validation_sampler = ML1ReachTaskSampler(
        dataset, split='validation', seed=int(cfg.train.seed) + 2001
    )

    wandb_mod = _maybe_wandb(cfg)
    run_id = _run_id(wandb_mod)
    run_dir = Path(cfg.train.output_dir).expanduser().resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    provenance = collect_experiment_provenance(
        repo_root=Path(__file__).resolve().parents[2],
        config=cfg,
        experiment_id=run_id,
        dataset=dataset.provenance(),
        parent_checkpoint=str(cfg.train.resume_path),
        adaptation_mode='query_only_no_support',
        reset_policy='not_applicable',
    )
    write_experiment_ledger(run_dir, config=cfg, provenance=provenance)
    _write_json(run_dir / 'normalization.json', dataset.normalization.to_dict())
    _write_json(run_dir / 'dataset_integrity.json', integrity)
    _write_json(
        run_dir / 'task_splits.json',
        {
            split: list(dataset.task_ids(split))
            for split in ('train', 'validation', 'test')
        },
    )
    _LOGGER.info('ML1 Reach query-only run directory: %s', run_dir)

    start_step = int(jax.device_get(state.step))
    target_step = int(cfg.train.num_steps)
    if start_step >= target_step:
        raise ValueError(
            f'Checkpoint step {start_step} is already >= train.num_steps {target_step}.'
        )
    last_log_time = time.time()
    last_log_step = start_step
    try:
        for step in range(start_step + 1, target_step + 1):
            batch = train_sampler.build_query_batch(
                int(cfg.train.batch_size),
                query_episodes=int(cfg.train.query_episodes_per_task),
            )
            query = _query_to_jax(batch['query'])
            state, metrics = train_step(state, query)
            if step == start_step + 1:
                jax.block_until_ready(metrics['loss'])
            if step == start_step + 1 or step % int(cfg.train.log_every) == 0:
                now = time.time()
                host_metrics = _host_metrics(metrics)
                host_metrics['step_s'] = (now - last_log_time) / max(
                    1, step - last_log_step
                )
                _log_metrics('train', step, host_metrics, wandb_mod)
                last_log_time = now
                last_log_step = step

            if step % int(cfg.train.eval_every) == 0:
                values = []
                for _ in range(int(cfg.train.eval_batches)):
                    validation_batch = validation_sampler.build_query_batch(
                        int(cfg.train.batch_size),
                        query_episodes=int(cfg.train.query_episodes_per_task),
                    )
                    _, validation_metrics = query_only_objective(
                        state.params,
                        _query_to_jax(validation_batch['query']),
                        model_cfg,
                    )
                    values.append(_host_metrics(validation_metrics))
                averaged = {
                    name: float(np.mean([value[name] for value in values]))
                    for name in values[0]
                }
                _log_metrics('validation', step, averaged, wandb_mod)

            if step % int(cfg.train.ckpt_every) == 0:
                _save_state(
                    run_dir / f'step_{step:07d}.pkl',
                    state,
                    step=step,
                    cfg=cfg,
                    dataset=dataset,
                    model_cfg=model_cfg,
                    provenance=provenance,
                )
        _save_state(
            run_dir / 'last.pkl',
            state,
            step=target_step,
            cfg=cfg,
            dataset=dataset,
            model_cfg=model_cfg,
            provenance=provenance,
        )
        _LOGGER.info(
            'ML1 Reach query-only training complete: %s', run_dir / 'last.pkl'
        )
        return run_dir / 'last.pkl'
    finally:
        if wandb_mod is not None:
            wandb_mod.finish()


__all__ = [
    'CHECKPOINT_TYPE',
    'metaworld_model_config_from',
    'train_metaworld_query_only',
    'validate_metaworld_query_config',
]
