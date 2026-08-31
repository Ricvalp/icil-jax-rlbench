from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict
from phi_mujoco.evaluation import EvaluationConfig, EvaluationRunner
from phi_mujoco.integrations.metaworld_ml1_reach import (
    MetaWorldML1ReachIntegration,
)

from icil_jax_rlbench.data.metaworld_ml1_reach import (
    ML1ReachTaskDataset,
    ML1ReachTaskSampler,
)
from icil_jax_rlbench.eval.metaworld_policy import ML1ReachJaxPolicy
from icil_jax_rlbench.eval.support_controls import (
    SUPPORTED_CONDITIONS,
    bootstrap_mean_confidence_interval,
    condition_support,
    confidence_interval,
    random_fast_state_with_matched_delta,
)
from icil_jax_rlbench.models.fast_weight_ttt import (
    adapt_fast_state,
    initial_fast_state,
    named_fast_tensor_metrics,
    query_imitation_loss,
    tree_difference_norm,
)
from icil_jax_rlbench.train.checkpoints import load_checkpoint
from icil_jax_rlbench.train.metaworld_query_runner import (
    metaworld_model_config_from,
)
from icil_jax_rlbench.train.metaworld_ttt_runner import (
    CHECKPOINT_TYPE,
    validate_metaworld_ttt_config,
)
from icil_jax_rlbench.train.provenance import config_to_dict
from icil_jax_rlbench.train.ttt_config import adaptation_config_from


_LOGGER = logging.getLogger(__name__)
_SUPPORT_FIELDS = ('observation', 'action', 'next_observation', 'write_mask')
_QUERY_FIELDS = ('observation', 'action', 'outer_loss_mask')


def _write_json(path: Path, value: Any) -> None:
    with path.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write('\n')


def _remove_task_axis(
    section: Mapping[str, np.ndarray],
    fields: Sequence[str],
) -> dict[str, np.ndarray]:
    return {name: np.asarray(section[name][0]) for name in fields}


def _fresh_evaluation_seed(
    requested_seed: int,
    episodes: int,
    dataset: ML1ReachTaskDataset,
) -> int:
    used = {episode.seed for episode in dataset.bundle.episodes}
    candidate = int(requested_seed)
    while any(candidate + offset in used for offset in range(int(episodes))):
        candidate += int(episodes)
    if candidate + int(episodes) - 1 > 2**32 - 1:
        raise ValueError('Could not choose fresh closed-loop evaluation seeds.')
    return candidate


def _trace_summary(trace: Mapping[str, jax.Array]) -> dict[str, dict[str, float]]:
    host = jax.device_get(trace)
    result = {}
    for name, value in host.items():
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        if array.size:
            result[name] = {
                'first': float(array[0]),
                'last': float(array[-1]),
                'mean': float(np.mean(array)),
                'maximum': float(np.max(array)),
            }
    return result


def _host_scalars(values: Mapping[str, Any]) -> dict[str, float]:
    return {
        name: float(np.asarray(value))
        for name, value in jax.device_get(values).items()
    }


def _paired_gain_summary(
    records: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    task_ids = [str(record['task_id']) for record in records]
    baseline_ids = [str(record['task_id']) for record in baseline]
    if task_ids != baseline_ids:
        raise ValueError('Paired MetaWorld records are not aligned by task ID.')
    values = {
        'success_rate': np.asarray(
            [
                float(record['closed_loop']['success_rate'])
                - float(base['closed_loop']['success_rate'])
                for record, base in zip(records, baseline)
            ]
        ),
        'offline_loss_reduction': np.asarray(
            [
                float(base['offline_query_loss'])
                - float(record['offline_query_loss'])
                for record, base in zip(records, baseline)
            ]
        ),
    }
    summary: dict[str, Any] = {'task_count': len(records)}
    for index, (name, array) in enumerate(values.items()):
        mean, low, high = bootstrap_mean_confidence_interval(
            array, seed=int(seed) + index
        )
        summary[name] = {
            'mean': mean,
            'ci95': [low, high],
            'per_task': array.tolist(),
        }
    return summary


def _aggregate(
    records: Mapping[str, Mapping[str, list[dict[str, Any]]]],
    *,
    seed: int,
) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for support_index, (count_key, by_condition) in enumerate(records.items()):
        aggregate[count_key] = {}
        for condition, condition_records in by_condition.items():
            successes = sum(
                int(record['closed_loop']['successful_episodes'])
                for record in condition_records
            )
            attempts = sum(
                int(record['closed_loop']['attempted_episodes'])
                for record in condition_records
            )
            low, high = confidence_interval(successes, attempts)
            aggregate[count_key][condition] = {
                'success_rate': successes / max(1, attempts),
                'successful_episodes': successes,
                'attempted_episodes': attempts,
                'success_ci95': [low, high],
                'mean_offline_query_loss': float(
                    np.mean(
                        [record['offline_query_loss'] for record in condition_records]
                    )
                ),
                'mean_fast_delta_norm': float(
                    np.mean(
                        [record['fast_delta_norm'] for record in condition_records]
                    )
                ),
            }
        baseline = by_condition.get('no_update')
        if baseline is not None:
            aggregate[count_key]['paired_gain_over_no_update'] = {
                condition: _paired_gain_summary(
                    condition_records,
                    baseline,
                    seed=int(seed) + support_index * 100 + condition_index,
                )
                for condition_index, (condition, condition_records) in enumerate(
                    by_condition.items()
                )
                if condition != 'no_update'
            }
        correct = aggregate[count_key].get('correct_support')
        wrong = aggregate[count_key].get('wrong_task_support')
        no_update = aggregate[count_key].get('no_update')
        if correct is not None and no_update is not None:
            aggregate[count_key]['correct_support_gain'] = {
                'success_rate': correct['success_rate'] - no_update['success_rate'],
                'offline_loss_reduction': (
                    no_update['mean_offline_query_loss']
                    - correct['mean_offline_query_loss']
                ),
            }
        if correct is not None and wrong is not None:
            aggregate[count_key]['correct_vs_wrong_gap'] = {
                'success_rate': correct['success_rate'] - wrong['success_rate'],
                'offline_loss': (
                    wrong['mean_offline_query_loss']
                    - correct['mean_offline_query_loss']
                ),
            }
    return aggregate


def evaluate_metaworld_ttt(cfg: ConfigDict) -> Path:
    if not str(cfg.checkpoint_path):
        raise ValueError('checkpoint_path is required.')
    payload = load_checkpoint(str(cfg.checkpoint_path))
    extra = payload.get('extra', {})
    if extra.get('checkpoint_type') != CHECKPOINT_TYPE:
        raise ValueError('Evaluation requires a MetaWorld fast-weight TTT checkpoint.')
    train_cfg = ConfigDict(payload['config'])
    validate_metaworld_ttt_config(train_cfg)

    split = str(cfg.split)
    if split not in ('validation', 'test'):
        raise ValueError("Held-out TTT evaluation split must be 'validation' or 'test'.")
    conditions = tuple(str(condition) for condition in cfg.conditions)
    unknown = sorted(set(conditions) - set(SUPPORTED_CONDITIONS))
    if unknown:
        raise ValueError(f'Unknown support conditions: {unknown}.')
    if 'no_update' not in conditions or 'correct_support' not in conditions:
        raise ValueError("Evaluation requires 'no_update' and 'correct_support'.")
    support_counts = tuple(int(value) for value in cfg.support_counts)
    if not support_counts or any(value < 1 for value in support_counts):
        raise ValueError('support_counts must contain positive values.')
    if int(cfg.offline_query_episodes) < 1:
        raise ValueError('offline_query_episodes must be positive.')
    if int(cfg.closed_loop_episodes) < 1:
        raise ValueError('closed_loop_episodes must be positive.')

    cache_root = str(cfg.cache_root) or str(train_cfg.dataset.cache_root)
    normalization = extra.get('normalization')
    if not isinstance(normalization, Mapping):
        raise ValueError('Checkpoint lacks normalization statistics.')
    dataset = ML1ReachTaskDataset(
        cache_root,
        normalization=normalization,
        cache_prepared_episodes=bool(cfg.cache_prepared_episodes),
    )
    if dataset.bundle.data_sha256 != extra.get('cache_data_sha256'):
        raise ValueError('Evaluation cache differs from the checkpoint cache.')
    if dataset.normalization_id != extra.get('normalizer_id'):
        raise ValueError('Evaluation normalization differs from the checkpoint.')
    if max(support_counts) + int(cfg.offline_query_episodes) > int(
        dataset.task_index.episodes_per_task
    ):
        raise ValueError('Requested support/query episodes exceed episodes per task.')

    model_cfg = metaworld_model_config_from(train_cfg, dataset)
    if asdict(model_cfg) != extra.get('model_config'):
        raise ValueError('Checkpoint model metadata differs from its config.')
    checkpoint_adapt_cfg = adaptation_config_from(train_cfg)
    if asdict(checkpoint_adapt_cfg) != extra.get('adaptation_config'):
        raise ValueError('Checkpoint adaptation metadata differs from its config.')
    adapt_cfg = checkpoint_adapt_cfg
    if int(cfg.write_steps_per_segment_override) > 0:
        adapt_cfg = replace(
            adapt_cfg,
            write_steps_per_segment=int(cfg.write_steps_per_segment_override),
        )
    params = jax.tree_util.tree_map(jnp.asarray, payload['params'])

    all_task_ids = dataset.task_ids(split)
    task_ids = (
        all_task_ids
        if int(cfg.max_tasks) <= 0
        else all_task_ids[: int(cfg.max_tasks)]
    )
    if not task_ids:
        raise ValueError(f'No tasks selected from split {split!r}.')
    evaluation_seed = _fresh_evaluation_seed(
        int(cfg.closed_loop_base_seed),
        int(cfg.closed_loop_episodes),
        dataset,
    )
    checkpoint_name = Path(cfg.checkpoint_path).stem
    run_dir = (
        Path(cfg.output_dir).expanduser().resolve()
        / f'{checkpoint_name}_{datetime.now(UTC).strftime("%Y%m%d-%H%M%S")}'
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    sampler = ML1ReachTaskSampler(
        dataset, split=split, seed=int(cfg.seed) + 8001
    )
    adapt = jax.jit(
        lambda value, support: adapt_fast_state(
            value, support, model_cfg, adapt_cfg
        )
    )
    offline_query = jax.jit(
        lambda value, fast_state, query: query_imitation_loss(
            value, fast_state, query, model_cfg, adapt_cfg
        )
    )
    base_integration = MetaWorldML1ReachIntegration(
        catalog_seed=dataset.task_index.catalog.catalog_seed
    )
    records: dict[str, dict[str, list[dict[str, Any]]]] = {
        str(count): {condition: [] for condition in conditions}
        for count in support_counts
    }

    for support_index, support_count in enumerate(support_counts):
        count_key = str(support_count)
        for task_position, task_id in enumerate(task_ids):
            correct_batch = sampler.build_batch(
                1,
                support_episodes=support_count,
                query_episodes=int(cfg.offline_query_episodes),
                task_ids=[task_id],
            )
            wrong_task_id = all_task_ids[
                (all_task_ids.index(task_id) + 1) % len(all_task_ids)
            ]
            wrong_batch = sampler.build_batch(
                1,
                support_episodes=support_count,
                query_episodes=1,
                task_ids=[wrong_task_id],
            )
            correct_support_np = _remove_task_axis(
                correct_batch['support'], _SUPPORT_FIELDS
            )
            wrong_support_np = _remove_task_axis(
                wrong_batch['support'], _SUPPORT_FIELDS
            )
            query = {
                name: jnp.asarray(value)
                for name, value in _remove_task_axis(
                    correct_batch['query'], _QUERY_FIELDS
                ).items()
            }
            correct_support = jax.tree_util.tree_map(
                jnp.asarray, correct_support_np
            )
            correct_adapted, correct_trace = adapt(params, correct_support)
            task_goal = dataset.task_index.catalog.task(task_id).goal

            for condition in conditions:
                condition_index = SUPPORTED_CONDITIONS.index(condition)
                control_rng = np.random.default_rng(
                    int(cfg.seed)
                    + 100_000 * support_index
                    + 1_000 * task_position
                    + condition_index
                )
                source_task_id = task_id
                source_episode_ids = np.asarray(
                    correct_batch['support']['episode_id'][0]
                ).astype(int).tolist()
                if condition == 'no_update':
                    fast_state = initial_fast_state(params)
                    trace = {}
                    source_task_id = ''
                    source_episode_ids = []
                elif condition == 'correct_support':
                    fast_state = correct_adapted
                    trace = correct_trace
                elif condition == 'random_update_matched_norm':
                    fast_state = random_fast_state_with_matched_delta(
                        params, correct_adapted, control_rng
                    )
                    trace = {}
                else:
                    conditioned = condition_support(
                        condition,
                        correct_support_np,
                        wrong_support_np,
                        control_rng,
                    )
                    fast_state, trace = adapt(
                        params, jax.tree_util.tree_map(jnp.asarray, conditioned)
                    )
                    if condition == 'wrong_task_support':
                        source_task_id = wrong_task_id
                        source_episode_ids = np.asarray(
                            wrong_batch['support']['episode_id'][0]
                        ).astype(int).tolist()

                offline_loss, offline_metrics = offline_query(
                    params, fast_state, query
                )
                integration = base_integration.for_task(task_id)
                policy = ML1ReachJaxPolicy(
                    integration=integration,
                    params=params,
                    fast_state=fast_state,
                    read_enabled=bool(adapt_cfg.read_enabled),
                    model_cfg=model_cfg,
                    normalization=dataset.normalization,
                )
                save_rollouts = bool(cfg.save_rollout_artifacts) or bool(
                    cfg.record_video
                )
                rollout_directory = (
                    run_dir
                    / 'rollouts'
                    / f'support-{support_count}'
                    / task_id
                    / condition
                    if save_rollouts
                    else None
                )
                closed_loop = EvaluationRunner(
                    integration,
                    policy,
                    EvaluationConfig(
                        episodes=int(cfg.closed_loop_episodes),
                        base_seed=evaluation_seed,
                        max_steps=(
                            None
                            if int(cfg.closed_loop_max_steps) <= 0
                            else int(cfg.closed_loop_max_steps)
                        ),
                        run_directory=rollout_directory,
                        record_video=bool(cfg.record_video),
                        render_width=int(cfg.render_width),
                        render_height=int(cfg.render_height),
                    ),
                ).run()
                fast_metrics = named_fast_tensor_metrics(
                    params, fast_state, model_cfg
                )
                records[count_key][condition].append(
                    {
                        'task_id': task_id,
                        'privileged_task_goal': list(task_goal),
                        'support_source_task_id': source_task_id or None,
                        'support_episode_ids': source_episode_ids,
                        'offline_query_episode_ids': np.asarray(
                            correct_batch['query']['episode_id'][0]
                        ).astype(int).tolist(),
                        'offline_query_loss': float(jax.device_get(offline_loss)),
                        'offline_query_metrics': _host_scalars(offline_metrics),
                        'fast_delta_norm': float(
                            jax.device_get(
                                tree_difference_norm(
                                    fast_state, initial_fast_state(params)
                                )
                            )
                        ),
                        'correct_reference_delta_norm': float(
                            jax.device_get(
                                tree_difference_norm(
                                    correct_adapted, initial_fast_state(params)
                                )
                            )
                        ),
                        'write_trace': _trace_summary(trace),
                        'fast_tensor_metrics': _host_scalars(fast_metrics),
                        'closed_loop': closed_loop.as_dict(),
                    }
                )
            _LOGGER.info(
                'MetaWorld TTT support=%d task %d/%d complete: %s',
                support_count,
                task_position + 1,
                len(task_ids),
                task_id,
            )

    aggregate = _aggregate(records, seed=int(cfg.seed) + 50_000)
    summary = {
        'gate': 'metaworld_hidden_goal_adaptation_only_generalization',
        'checkpoint_path': str(Path(cfg.checkpoint_path).expanduser().resolve()),
        'checkpoint_step': int(payload['step']),
        'checkpoint_adaptation_config': asdict(checkpoint_adapt_cfg),
        'evaluation_adaptation_config': asdict(adapt_cfg),
        'dataset': dataset.provenance(),
        'split': split,
        'task_ids': list(task_ids),
        'support_counts': list(support_counts),
        'offline_query_episodes': int(cfg.offline_query_episodes),
        'closed_loop_episodes': int(cfg.closed_loop_episodes),
        'closed_loop_base_seed': evaluation_seed,
        'closed_loop_seeds_fresh_relative_to_cache': True,
        'matched_closed_loop_seeds_across_conditions': True,
        'support_query_episode_overlap': False,
        'goal_provided_to_policy': False,
        'reset_policy': 'reset_to_w0_for_each_task_and_condition',
        'query_fast_state_policy': 'freeze_after_support_and_reuse_across_query_episodes',
        'paired_inference_unit': 'held_out_goal',
        'paired_ci_method': '10000-sample percentile bootstrap over held-out goals',
        'aggregate': aggregate,
        'per_support_count': records,
    }
    _write_json(run_dir / 'resolved_eval_config.json', config_to_dict(cfg))
    _write_json(run_dir / 'summary.json', summary)
    _LOGGER.info('MetaWorld fast-weight TTT evaluation complete: %s', run_dir)
    return run_dir


__all__ = ['evaluate_metaworld_ttt']
