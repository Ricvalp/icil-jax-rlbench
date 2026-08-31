from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from phi_mujoco.evaluation import PolicyInput
from phi_mujoco.offline import StandardNormalization

from icil_jax_rlbench.models.fast_weight_ttt import (
    FastWeightTTTConfig,
    initial_fast_state,
    predict_action,
)


@partial(jax.jit, static_argnames=('model_cfg', 'read_enabled'))
def _predict_normalized_action(
    params: Mapping[str, Any],
    fast_state: Any,
    observation: jax.Array,
    *,
    model_cfg: FastWeightTTTConfig,
    read_enabled: bool,
) -> jax.Array:
    return predict_action(
        params,
        fast_state,
        observation,
        model_cfg,
        read_enabled=bool(read_enabled),
    )


class ML1ReachJaxPolicy:
    """Expose a frozen JAX query policy to the public phi policy interface."""

    def __init__(
        self,
        *,
        integration,
        params: Mapping[str, Any],
        model_cfg: FastWeightTTTConfig,
        normalization: StandardNormalization,
        fast_state: Any | None = None,
        read_enabled: bool = False,
    ) -> None:
        self.integration = integration
        self.params = jax.tree_util.tree_map(jnp.asarray, params)
        self.model_cfg = model_cfg
        self.normalization = normalization
        self.fast_state = jax.tree_util.tree_map(
            jnp.asarray,
            initial_fast_state(self.params) if fast_state is None else fast_state,
        )
        self.read_enabled = bool(read_enabled)

    def reset(self, *, integration, seed: int) -> None:
        del seed
        if integration != self.integration.spec:
            raise ValueError('Evaluation runner integration changed during a rollout.')

    def predict(self, inputs: PolicyInput) -> np.ndarray:
        if inputs.integration != self.integration.spec:
            raise ValueError('Policy input does not match the bound ML1 Reach task.')
        if set(inputs.observations) != {'state'}:
            raise ValueError("ML1 Reach policy expects exactly the 'state' observation.")
        state = np.asarray(inputs.observations['state'], dtype=np.float32)
        if state.shape != (self.model_cfg.observation_dim,):
            raise ValueError(
                f'Expected state shape {(self.model_cfg.observation_dim,)}, got {state.shape}.'
            )
        normalized_state = self.normalization.normalize_observations(state)
        normalized_action = _predict_normalized_action(
            self.params,
            self.fast_state,
            jnp.asarray(normalized_state),
            model_cfg=self.model_cfg,
            read_enabled=self.read_enabled,
        )
        action = self.normalization.denormalize_actions(
            np.asarray(jax.device_get(normalized_action))
        )
        return self.integration.project_action(
            np.asarray(action, dtype=np.float32),
            observations=inputs.observations,
        )


__all__ = ['ML1ReachJaxPolicy']
