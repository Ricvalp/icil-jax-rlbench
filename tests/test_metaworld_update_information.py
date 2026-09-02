from __future__ import annotations

import numpy as np

from icil_jax_rlbench.analysis.metaworld_update_information import (
    raw_support_statistics,
    ridge_predict,
)
from icil_jax_rlbench.configs.analyze_metaworld_ml10_information import get_config


def test_raw_support_statistics_preserve_demo_time_contract_and_ignore_padding() -> None:
    observation = np.zeros((2, 4, 3), dtype=np.float32)
    action = np.zeros((2, 4, 2), dtype=np.float32)
    next_observation = np.zeros_like(observation)
    mask = np.asarray(
        [[True, True, False, False], [True, True, True, False]], dtype=np.bool_
    )
    observation[mask] = np.arange(15, dtype=np.float32).reshape(5, 3)
    next_observation[mask] = observation[mask] + 1.0
    action[mask] = 0.5
    support = {
        'observation': observation,
        'action': action,
        'next_observation': next_observation,
        'write_mask': mask,
    }
    first = raw_support_statistics(support)
    support['observation'][~mask] = 1e6
    support['action'][~mask] = -1e6
    second = raw_support_statistics(support)

    evidence_width = observation.shape[-1] * 2 + action.shape[-1]
    assert first.shape == ((2 + 4) * evidence_width + 2,)
    np.testing.assert_array_equal(first, second)
    assert np.isfinite(first).all()


def test_ridge_probe_recovers_linear_targets_in_high_dimensional_dual_case() -> None:
    rng = np.random.default_rng(4)
    train_latent = rng.normal(size=(20, 4))
    test_latent = rng.normal(size=(5, 4))
    feature_map = rng.normal(size=(4, 40))
    target_map = rng.normal(size=(4, 3))
    train = train_latent @ feature_map
    test = test_latent @ feature_map
    target = train_latent @ target_map
    expected = test_latent @ target_map

    prediction = ridge_predict(train, target, test, ridge=1e-8)

    np.testing.assert_allclose(prediction, expected, atol=2e-4, rtol=2e-4)


def test_ml10_information_config_requests_matched_controls_and_full_vectors() -> None:
    cfg = get_config()
    assert cfg.support_counts == (1, 2)
    assert cfg.samples_per_task == 2
    assert 'correct_support' in cfg.conditions
    assert 'same_family_wrong_instance' in cfg.conditions
    assert 'different_family_support' in cfg.conditions
    assert 'shuffled_actions' in cfg.conditions
    assert 'shuffled_time' in cfg.conditions
