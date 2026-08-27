from __future__ import annotations

from typing import Any, Dict, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from icil_jax_rlbench.data.hidden_goal import HiddenGoalEnvironment, StateNormalizer
from icil_jax_rlbench.models.fast_weight_ttt import (
    executable_action,
    initial_fast_state,
    predict_action,
    tree_difference_norm,
    tree_l2_norm,
)


def to_jax(value: Any) -> Any:
    return jax.tree_util.tree_map(jnp.asarray, value)


def remove_task_axis(section: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {name: np.asarray(value[0]) for name, value in section.items()}


def _shuffle_first_two_axes(value: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    flat = value.reshape((-1,) + value.shape[2:])
    return flat[permutation].reshape(value.shape)


def condition_support(
    condition: str,
    correct_support: Mapping[str, np.ndarray],
    wrong_support: Mapping[str, np.ndarray],
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    source = wrong_support if condition == 'wrong_task_support' else correct_support
    support = {name: np.array(value, copy=True) for name, value in source.items()}
    if condition in ('no_update', 'correct_support', 'wrong_task_support'):
        return support
    if condition == 'shuffled_actions':
        count = int(np.prod(support['action'].shape[:2]))
        permutation = rng.permutation(count)
        support['action'] = _shuffle_first_two_axes(support['action'], permutation)
        return support
    if condition == 'shuffled_time':
        count = int(np.prod(support['write_mask'].shape[:2]))
        permutation = rng.permutation(count)
        for name in ('observation', 'action', 'next_observation', 'write_mask'):
            support[name] = _shuffle_first_two_axes(support[name], permutation)
        return support
    if condition == 'observations_only':
        support['action'].fill(0.0)
        return support
    if condition == 'actions_only':
        support['observation'].fill(0.0)
        support['next_observation'].fill(0.0)
        return support
    if condition == 'duplicated_support':
        for name in ('observation', 'action', 'next_observation', 'write_mask'):
            support[name] = np.repeat(
                support[name][:1], support[name].shape[0], axis=0
            )
        return support
    if condition == 'random_update_matched_norm':
        return support
    raise ValueError(f'Unknown support condition {condition!r}.')


def random_fast_state_with_matched_delta(
    params: Mapping[str, Any],
    correct_adapted: Any,
    rng: np.random.Generator,
) -> Any:
    initial = initial_fast_state(params)
    target_norm = tree_difference_norm(correct_adapted, initial)
    random_direction = jax.tree_util.tree_map(
        lambda value: jnp.asarray(rng.normal(size=value.shape), dtype=value.dtype),
        initial,
    )
    direction_norm = tree_l2_norm(random_direction)
    scale = target_norm / (direction_norm + 1e-8)
    return jax.tree_util.tree_map(
        lambda value, direction: value + scale.astype(direction.dtype) * direction,
        initial,
        random_direction,
    )


def confidence_interval(successes: int, count: int) -> tuple[float, float]:
    if count <= 0:
        return 0.0, 0.0
    probability = successes / count
    standard_error = np.sqrt(max(probability * (1.0 - probability), 1e-12) / count)
    return (
        float(max(0.0, probability - 1.96 * standard_error)),
        float(min(1.0, probability + 1.96 * standard_error)),
    )


def rollout(
    params: Mapping[str, Any],
    fast_state: Any,
    *,
    goal: np.ndarray,
    episode_id: int,
    benchmark_cfg,
    normalizer: StateNormalizer,
    model_cfg,
    read_enabled: bool,
) -> Dict[str, Any]:
    env = HiddenGoalEnvironment(benchmark_cfg, goal, int(episode_id))
    action_trace = []
    for _ in range(int(benchmark_cfg.horizon)):
        normalized_observation = normalizer.normalize_observation(env.observation())
        prediction = predict_action(
            params,
            fast_state,
            jnp.asarray(normalized_observation),
            model_cfg,
            read_enabled=bool(read_enabled),
        )
        action = np.asarray(jax.device_get(executable_action(prediction, model_cfg)))
        action_trace.append(action)
        _, done = env.step(action)
        if done:
            break
    return {
        'success': bool(env.success()),
        'final_distance': float(env.final_distance()),
        'rollout_length': len(action_trace),
    }
