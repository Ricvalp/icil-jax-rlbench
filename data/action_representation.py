from __future__ import annotations

import numpy as np

SUPPORTED_ACTION_REPRESENTATIONS = ('absolute', 'delta_xyz')


def normalize_action_representation(value: str) -> str:
    value = str(value).strip().lower()
    if value not in SUPPORTED_ACTION_REPRESENTATIONS:
        raise ValueError(f'action_representation must be one of {SUPPORTED_ACTION_REPRESENTATIONS}, got {value!r}.')
    return value


def encode_action_chunk(action_chunk: np.ndarray, *, query_state: np.ndarray, representation: str) -> np.ndarray:
    representation = normalize_action_representation(representation)
    action_chunk = np.asarray(action_chunk, dtype=np.float32)
    if representation == 'absolute':
        return action_chunk
    if action_chunk.ndim < 2 or action_chunk.shape[-1] < 3:
        raise ValueError(f'action_chunk must be [..., H, A>=3], got {action_chunk.shape}.')
    if query_state.ndim < 2 or query_state.shape[-1] < 3:
        raise ValueError(f'query_state must be [..., T_obs, S>=3], got {query_state.shape}.')
    out = np.array(action_chunk, copy=True)
    anchor = query_state[..., -1, :3]
    out[..., 0, :3] = action_chunk[..., 0, :3] - anchor
    if action_chunk.shape[-2] > 1:
        out[..., 1:, :3] = action_chunk[..., 1:, :3] - action_chunk[..., :-1, :3]
    return out


def encode_support_traj(traj: np.ndarray, *, representation: str) -> np.ndarray:
    representation = normalize_action_representation(representation)
    traj = np.asarray(traj, dtype=np.float32)
    if representation == 'absolute':
        return traj
    out = np.array(traj, copy=True)
    out[..., 0, :3] = 0.0
    if traj.shape[-2] > 1:
        out[..., 1:, :3] = traj[..., 1:, :3] - traj[..., :-1, :3]
    return out


def decode_action_chunk(action_chunk: np.ndarray, *, query_state: np.ndarray, representation: str) -> np.ndarray:
    representation = normalize_action_representation(representation)
    action_chunk = np.asarray(action_chunk, dtype=np.float32)
    if representation == 'absolute':
        return action_chunk
    out = np.array(action_chunk, copy=True)
    anchor = query_state[..., -1:, :3]
    out[..., :, :3] = np.cumsum(action_chunk[..., :, :3], axis=-2) + anchor
    return out
