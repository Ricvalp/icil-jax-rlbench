from __future__ import annotations

from icil_jax_rlbench.configs.eval_metaworld_ml45_ttt import (
    get_config as get_eval_config,
)
from icil_jax_rlbench.configs.metaworld_ml45_action_bc import (
    get_config as get_action_bc_config,
)
from icil_jax_rlbench.configs.metaworld_ml45_fomaml import (
    get_config as get_fomaml_config,
)
from icil_jax_rlbench.configs.metaworld_ml45_kvb import (
    get_config as get_kvb_config,
)
from icil_jax_rlbench.configs.metaworld_ml45_query_only import (
    get_config as get_query_config,
)
from icil_jax_rlbench.data.metaworld_hidden_goal import (
    benchmark_for_integration,
    benchmark_from_config,
)
from icil_jax_rlbench.eval.metaworld_hidden_goal_ttt import _grouped_aggregate


def test_ml45_benchmark_uses_the_shared_hidden_goal_policy_contract() -> None:
    benchmark = benchmark_for_integration('metaworld_ml45')
    assert benchmark.label == 'ML45'
    assert benchmark.query_mode == 'metaworld_ml45_query_only'
    assert benchmark.ttt_mode == 'metaworld_ml45_ttt'
    assert benchmark.protocols == ('development', 'final')
    assert benchmark.family_aware
    assert benchmark.split_names == (
        'train',
        'latent_validation',
        'family_validation',
        'validation',
        'test',
    )


def test_ml45_query_and_main_kvb_configs_select_delta_read() -> None:
    query = get_query_config()
    kvb = get_kvb_config()
    assert benchmark_from_config(query).integration_name == 'metaworld_ml45'
    assert query.mode == 'metaworld_ml45_query_only'
    assert query.dataset.protocol == 'development'
    assert query.train.num_steps == 100_000
    assert kvb.mode == 'metaworld_ml45_ttt'
    assert kvb.dataset.integration == 'metaworld_ml45'
    assert kvb.adaptation.write_objective == 'kvb'
    assert kvb.adaptation.read_mode == 'delta'
    assert not kvb.adaptation.first_order
    assert kvb.train.num_steps == 100_000


def test_ml45_ablation_and_evaluation_configs_remain_explicit() -> None:
    fomaml = get_fomaml_config()
    action_bc = get_action_bc_config()
    evaluation = get_eval_config()
    assert fomaml.adaptation.first_order
    assert fomaml.adaptation.write_objective == 'kvb'
    assert action_bc.adaptation.write_objective == 'action_bc'
    assert not action_bc.adaptation.first_order
    assert action_bc.adaptation.read_mode == 'delta'
    assert evaluation.integration == 'metaworld_ml45'
    assert evaluation.split == 'family_validation'
    assert 'same_family_wrong_instance' in evaluation.conditions
    assert 'different_family_support' in evaluation.conditions


def test_ml45_evaluation_can_aggregate_families_and_shared_motion_phases() -> None:
    def record(task_id: str, family: str, phases: list[str], loss: float, success: int):
        return {
            'task_id': task_id,
            'task_family': family,
            'task_motion_phases': phases,
            'offline_query_loss': loss,
            'fast_delta_norm': 0.2,
            'closed_loop': {
                'success_rate': float(success),
                'successful_episodes': success,
                'attempted_episodes': 1,
            },
        }

    records = {
        '2': {
            'no_update': [
                record('a', 'pick-place-wall-v3', ['reach', 'grasp'], 2.0, 0),
                record('b', 'coffee-pull-v3', ['reach', 'grasp'], 1.0, 0),
            ],
            'correct_support': [
                record('a', 'pick-place-wall-v3', ['reach', 'grasp'], 1.0, 1),
                record('b', 'coffee-pull-v3', ['reach', 'grasp'], 0.5, 1),
            ],
        }
    }
    by_family = _grouped_aggregate(
        records, metadata_key='task_family', seed=0
    )
    by_phase = _grouped_aggregate(
        records, metadata_key='task_motion_phases', seed=0
    )
    assert set(by_family) == {'coffee-pull-v3', 'pick-place-wall-v3'}
    assert set(by_phase) == {'grasp', 'reach'}
    assert by_phase['reach']['2']['correct_support_gain']['success_rate'] == 1.0
