from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
from ml_collections import ConfigDict
import numpy as np

from icil_jax_rlbench.data.action_representation import decode_action_chunk, encode_action_chunk, encode_support_traj
from icil_jax_rlbench.data.h5_cache import RLBenchCacheStore, build_variation_keys
from icil_jax_rlbench.data.sampler import ICILDataConfig
from icil_jax_rlbench.eval.pointcloud_diagnostics import (
    _add_frame_traces,
    _parse_int_list,
    _supernode_geometry,
)
from icil_jax_rlbench.models.config import policy_config_from
from icil_jax_rlbench.models.direct_regression_policy import DirectRegressionPolicy
from icil_jax_rlbench.train.checkpoints import load_checkpoint


def _parse_optional_int_list(value: str) -> Tuple[int, ...]:
    value = str(value or '').strip()
    if not value:
        return ()
    return _parse_int_list(value)


def _checkpoint_config(ckpt: Dict[str, Any]) -> ConfigDict:
    return ConfigDict(ckpt.get('config', {}) or {})


def _data_config_from_checkpoint(ckpt_cfg: ConfigDict, args: argparse.Namespace) -> ICILDataConfig:
    data = getattr(ckpt_cfg, 'data', ConfigDict())

    def value(name: str, default: Any) -> Any:
        override = getattr(args, name, None)
        if override is not None and int(override) > 0:
            return override
        return getattr(data, name, default)

    return ICILDataConfig(
        K=int(value('K', 2)),
        L=int(value('L', 8)),
        T_obs=int(value('T_obs', 2)),
        H=int(value('H', 16)),
        stride=int(value('stride', 2)),
        action_representation=str(getattr(data, 'action_representation', 'absolute')),
        task_sampling='variation_uniform',
        task_sampling_alpha=1.0,
        traj_len=int(value('traj_len', 64)),
        query_window_mode=str(getattr(data, 'query_window_mode', 'online_history')),
        support_spacetime_points=int(
            args.support_spacetime_points
            if int(getattr(args, 'support_spacetime_points', 0)) > 0
            else getattr(data, 'support_spacetime_points', 0)
        ),
        support_spacetime_sampling=str(
            getattr(args, 'support_spacetime_sampling', '') or getattr(data, 'support_spacetime_sampling', 'mask_balanced')
        ),
    )


def _fixed_task_variation_ids(ckpt_cfg: ConfigDict, task: str, variation: int) -> Optional[Dict[str, np.ndarray]]:
    mode = str(getattr(getattr(ckpt_cfg, 'model', ConfigDict()).get('conditioning', ConfigDict()), 'mode', 'support'))
    if mode != 'task_variation':
        return None
    data = getattr(ckpt_cfg, 'data', ConfigDict())
    task_names = list(getattr(data, 'task_id_names', ()))
    variation_keys = list(getattr(data, 'task_variation_keys', ()))
    variation_key = f'{task}:{int(variation)}'
    if task not in task_names:
        raise ValueError(f'Task {task!r} is not in checkpoint task-token vocabulary.')
    if variation_key not in variation_keys:
        raise ValueError(f'Variation {variation_key!r} is not in checkpoint task-variation-token vocabulary.')
    return {
        'task_id': np.asarray(task_names.index(task), dtype=np.int32),
        'task_variation_id': np.asarray(variation_keys.index(variation_key), dtype=np.int32),
    }


def _obs_act_indices(
    t0: int,
    T: int,
    cfg: ICILDataConfig,
    *,
    query_stride_mode: str,
    query_window_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    qstep = int(cfg.stride) if str(query_stride_mode) == 'dataset' else 1
    if str(query_window_mode) == 'online_history':
        current = int(np.clip(int(t0), 0, max(0, int(T) - 1)))
        offsets = (int(cfg.T_obs) - 1 - np.arange(int(cfg.T_obs), dtype=np.int64)) * qstep
        obs = np.maximum(0, current - offsets).astype(np.int64)
        act_start = current + int(cfg.stride)
    elif str(query_window_mode) == 'forward':
        obs = int(t0) + np.arange(0, int(cfg.T_obs) * qstep, qstep, dtype=np.int64)
        if obs[-1] >= int(T):
            raise RuntimeError(f'Query window exceeds episode: t0={t0} obs_last={int(obs[-1])} T={T}.')
        act_start = int(obs[-1]) + int(cfg.stride)
    else:
        raise ValueError("query_window_mode must be 'online_history' or 'forward'.")
    act = act_start + np.arange(0, int(cfg.H) * int(cfg.stride), int(cfg.stride), dtype=np.int64)
    act = np.minimum(act, int(T) - 1)
    return obs, act


def _traj_indices(T: int, cfg: ICILDataConfig) -> Tuple[np.ndarray, np.ndarray]:
    M = int(cfg.traj_len)
    if M <= 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.bool_)
    raw = np.arange(0, int(T), max(1, int(cfg.stride)), dtype=np.int64)
    if len(raw) >= M:
        return raw[:M], np.ones((M,), dtype=np.bool_)
    pad = np.full((M - len(raw),), int(T) - 1, dtype=np.int64)
    mask = np.zeros((M,), dtype=np.bool_)
    mask[: len(raw)] = True
    return np.concatenate([raw, pad], axis=0), mask


def _even_keyframes(T: int, L: int) -> np.ndarray:
    if int(T) <= 0:
        raise RuntimeError('Cannot sample keyframes from an empty episode.')
    if int(L) <= 1:
        return np.asarray([0], dtype=np.int64)
    return np.linspace(0, int(T) - 1, int(L)).round().astype(np.int64)


def _support_conditioning(
    store: RLBenchCacheStore,
    *,
    vidx: int,
    support_ids: Sequence[int],
    cfg: ICILDataConfig,
    load_rgb: bool,
    load_mask_id: bool,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    cond_xyz: List[np.ndarray] = []
    cond_state: List[np.ndarray] = []
    cond_valid: List[np.ndarray] = []
    cond_rgb: List[np.ndarray] = []
    cond_mask_id: List[np.ndarray] = []
    st_xyz: List[np.ndarray] = []
    st_time: List[np.ndarray] = []
    st_state: List[np.ndarray] = []
    st_valid: List[np.ndarray] = []
    st_rgb: List[np.ndarray] = []
    st_mask_id: List[np.ndarray] = []
    cond_traj: List[np.ndarray] = []
    cond_traj_mask: List[np.ndarray] = []
    has_rgb = bool(load_rgb)
    has_mask = bool(load_mask_id)
    has_st_rgb = bool(load_rgb)
    has_st_mask = bool(load_mask_id)
    has_traj = int(cfg.traj_len) > 0
    has_spacetime = int(cfg.support_spacetime_points) > 0
    for eid in support_ids:
        T = store.episode_length(vidx, int(eid))
        frame_idx = _even_keyframes(T, int(cfg.L))
        item = store.load_episode_slices(vidx, int(eid), frame_idx, load_rgb=load_rgb, load_mask_id=load_mask_id)
        cond_xyz.append(item['xyz'])
        cond_state.append(item['state'])
        cond_valid.append(item['valid'])
        if load_rgb and 'rgb' in item:
            cond_rgb.append(item['rgb'])
        else:
            has_rgb = False
        if load_mask_id and 'mask_id' in item:
            cond_mask_id.append(item['mask_id'])
        else:
            has_mask = False
        if has_spacetime:
            st_item = _spacetime_support_item(item, frame_idx=frame_idx, episode_length=T, cfg=cfg, rng=rng)
            st_xyz.append(st_item['xyz'])
            st_time.append(st_item['time'])
            st_state.append(st_item['state'])
            st_valid.append(st_item['valid'])
            if load_rgb and 'rgb' in st_item:
                st_rgb.append(st_item['rgb'])
            else:
                has_st_rgb = False
            if load_mask_id and 'mask_id' in st_item:
                st_mask_id.append(st_item['mask_id'])
            else:
                has_st_mask = False
        if has_traj:
            tidx, tmask = _traj_indices(T, cfg)
            traj = store.load_episode_slices(vidx, int(eid), tidx, load_rgb=False, load_mask_id=False)['action']
            cond_traj.append(encode_support_traj(traj, representation=cfg.action_representation))
            cond_traj_mask.append(tmask)

    out: Dict[str, np.ndarray] = {
        'cond_xyz': np.stack(cond_xyz, axis=0).astype(np.float32),
        'cond_state': np.stack(cond_state, axis=0).astype(np.float32),
        'cond_valid': np.stack(cond_valid, axis=0).astype(np.bool_),
    }
    if has_rgb:
        out['cond_rgb'] = np.stack(cond_rgb, axis=0).astype(np.float32)
    if has_mask:
        out['cond_mask_id'] = np.stack(cond_mask_id, axis=0).astype(np.int32)
    if has_spacetime:
        out['cond_st_xyz'] = np.stack(st_xyz, axis=0).astype(np.float32)
        out['cond_st_time'] = np.stack(st_time, axis=0).astype(np.float32)
        out['cond_st_state'] = np.stack(st_state, axis=0).astype(np.float32)
        out['cond_st_valid'] = np.stack(st_valid, axis=0).astype(np.bool_)
        if has_st_rgb:
            out['cond_st_rgb'] = np.stack(st_rgb, axis=0).astype(np.float32)
        if has_st_mask:
            out['cond_st_mask_id'] = np.stack(st_mask_id, axis=0).astype(np.int32)
    if has_traj:
        out['cond_traj'] = np.stack(cond_traj, axis=0).astype(np.float32)
        out['cond_traj_mask'] = np.stack(cond_traj_mask, axis=0).astype(np.bool_)
    return out


def _sample_spacetime_indices(
    valid: np.ndarray,
    *,
    mask_id: Optional[np.ndarray],
    count: int,
    sampling: str,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    flat_valid = np.asarray(valid, dtype=np.bool_).reshape(-1)
    valid_idx = np.flatnonzero(flat_valid)
    if int(count) <= 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.bool_)
    if valid_idx.size == 0:
        return np.zeros((int(count),), dtype=np.int64), np.zeros((int(count),), dtype=np.bool_)
    use_mask_balance = (
        str(sampling) == 'mask_balanced'
        and mask_id is not None
        and np.asarray(mask_id).size == flat_valid.size
    )
    if not use_mask_balance:
        return (
            rng.choice(valid_idx, size=int(count), replace=valid_idx.size < int(count)).astype(np.int64),
            np.ones((int(count),), dtype=np.bool_),
        )
    masks = np.asarray(mask_id).reshape(-1).astype(np.int64)
    values = np.unique(masks[valid_idx])
    groups = [valid_idx[masks[valid_idx] == int(value)] for value in values.tolist()]
    group_choice = rng.integers(0, len(groups), size=int(count))
    chosen = np.empty((int(count),), dtype=np.int64)
    for gidx, group in enumerate(groups):
        slots = np.flatnonzero(group_choice == int(gidx))
        if slots.size:
            chosen[slots] = rng.choice(group, size=int(slots.size), replace=group.size < slots.size)
    rng.shuffle(chosen)
    return chosen, np.ones((int(count),), dtype=np.bool_)


def _spacetime_support_item(
    item: Dict[str, np.ndarray],
    *,
    frame_idx: np.ndarray,
    episode_length: int,
    cfg: ICILDataConfig,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    P = int(cfg.support_spacetime_points)
    xyz = np.asarray(item['xyz'], dtype=np.float32)
    valid = np.asarray(item['valid'], dtype=np.bool_)
    mask_id = np.asarray(item['mask_id'], dtype=np.int32) if 'mask_id' in item else None
    chosen, chosen_valid = _sample_spacetime_indices(
        valid,
        mask_id=mask_id,
        count=P,
        sampling=str(cfg.support_spacetime_sampling),
        rng=rng,
    )
    L, N = int(xyz.shape[0]), int(xyz.shape[1])
    frame_numbers = np.asarray(frame_idx, dtype=np.float32)
    denom = max(1.0, float(int(episode_length) - 1))
    out: Dict[str, np.ndarray] = {
        'xyz': xyz.reshape((L * N, -1))[chosen].astype(np.float32),
        'time': np.repeat(frame_numbers / denom, N).astype(np.float32)[chosen],
        'state': np.repeat(np.asarray(item['state'], dtype=np.float32), N, axis=0)[chosen],
        'valid': chosen_valid.astype(np.bool_),
    }
    if 'rgb' in item:
        out['rgb'] = np.asarray(item['rgb'], dtype=np.float32).reshape((L * N, -1))[chosen].astype(np.float32)
    if mask_id is not None:
        out['mask_id'] = mask_id.reshape(-1)[chosen].astype(np.int32)
    return out


def _query_sample(
    store: RLBenchCacheStore,
    *,
    vidx: int,
    episode_id: int,
    t0: int,
    cfg: ICILDataConfig,
    query_stride_mode: str,
    query_window_mode: str,
    load_rgb: bool,
    load_mask_id: bool,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    T = store.episode_length(vidx, int(episode_id))
    obs_idx, act_idx = _obs_act_indices(
        int(t0),
        T,
        cfg,
        query_stride_mode=query_stride_mode,
        query_window_mode=query_window_mode,
    )
    obs = store.load_episode_slices(vidx, int(episode_id), obs_idx, load_rgb=load_rgb, load_mask_id=load_mask_id)
    act = store.load_episode_slices(vidx, int(episode_id), act_idx, load_rgb=False, load_mask_id=False)
    task_id, task_variation_id = store.class_ids_for_vidx(vidx)
    sample: Dict[str, np.ndarray] = {
        'query_xyz': obs['xyz'].astype(np.float32),
        'query_state': obs['state'].astype(np.float32),
        'query_valid': obs['valid'].astype(np.bool_),
        'target_action': encode_action_chunk(
            act['action'],
            query_state=obs['state'],
            representation=cfg.action_representation,
        ).astype(np.float32),
        'chunk_start': np.asarray(float(t0), dtype=np.float32),
        'task_id': np.asarray(task_id, dtype=np.int32),
        'task_variation_id': np.asarray(task_variation_id, dtype=np.int32),
    }
    if load_rgb and 'rgb' in obs:
        sample['query_rgb'] = obs['rgb'].astype(np.float32)
    if load_mask_id and 'mask_id' in obs:
        sample['query_mask_id'] = obs['mask_id'].astype(np.int32)
    meta = {
        'T': int(T),
        'query_window_mode': str(query_window_mode),
        'obs_idx': [int(x) for x in obs_idx.tolist()],
        'act_idx': [int(x) for x in act_idx.tolist()],
    }
    return sample, meta


def _batchify(sample: Dict[str, np.ndarray], fixed_conditioning: Optional[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    out = {k: np.expand_dims(v, axis=0) for k, v in sample.items()}
    if fixed_conditioning is not None:
        for key, value in fixed_conditioning.items():
            out[key] = np.asarray(value, dtype=np.int32).reshape((1,))
    return out


def _choose_support_ids(all_episode_ids: Sequence[int], query_episode: int, K: int, explicit: Sequence[int]) -> Tuple[int, ...]:
    if explicit:
        return tuple(int(x) for x in explicit[: int(K)])
    candidates = [int(eid) for eid in all_episode_ids if int(eid) != int(query_episode)]
    if len(candidates) < int(K):
        raise RuntimeError(f'Need {K} support episodes besides query episode {query_episode}, found {len(candidates)}.')
    return tuple(candidates[: int(K)])


def _action_scene_ranges(frame: Dict[str, Any], pred_xyz: np.ndarray, target_xyz: np.ndarray) -> Dict[str, Tuple[float, float]]:
    xyz = np.asarray(frame['xyz'], dtype=np.float32).reshape((-1, 3))
    valid = np.asarray(frame.get('valid', np.ones((xyz.shape[0],), dtype=np.bool_))).reshape((-1,)).astype(np.bool_)
    points = [xyz[valid], np.asarray(pred_xyz, dtype=np.float32).reshape((-1, 3)), np.asarray(target_xyz, dtype=np.float32).reshape((-1, 3))]
    all_pts = np.concatenate([p for p in points if p.size > 0], axis=0)
    mins = np.nanmin(all_pts, axis=0)
    maxs = np.nanmax(all_pts, axis=0)
    center = 0.5 * (mins + maxs)
    span = max(float(np.nanmax(maxs - mins)), 1e-3)
    half = 0.55 * span
    return {
        'x': (float(center[0] - half), float(center[0] + half)),
        'y': (float(center[1] - half), float(center[1] + half)),
        'z': (float(center[2] - half), float(center[2] + half)),
    }


def _add_action_traces(fig: Any, *, pred_xyz: np.ndarray, target_xyz: np.ndarray, current_xyz: np.ndarray) -> None:
    import plotly.graph_objects as go

    pred = np.asarray(pred_xyz, dtype=np.float32)
    target = np.asarray(target_xyz, dtype=np.float32)
    current = np.asarray(current_xyz, dtype=np.float32).reshape(3)
    fig.add_trace(
        go.Scatter3d(
            x=target[:, 0],
            y=target[:, 1],
            z=target[:, 2],
            mode='lines+markers',
            line={'color': '#2563eb', 'width': 8},
            marker={'size': 4, 'color': '#2563eb'},
            name='ground truth action chunk',
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=pred[:, 0],
            y=pred[:, 1],
            z=pred[:, 2],
            mode='lines+markers',
            line={'color': '#dc2626', 'width': 8},
            marker={'size': 4, 'color': '#dc2626'},
            name='predicted action chunk',
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=[float(current[0])],
            y=[float(current[1])],
            z=[float(current[2])],
            mode='markers',
            marker={'size': 7, 'color': '#16a34a', 'symbol': 'cross'},
            name='current ee',
        ),
        row=1,
        col=1,
    )


def _write_html(
    *,
    frame: Dict[str, np.ndarray],
    geom: Dict[str, np.ndarray],
    pred_abs: np.ndarray,
    target_abs: np.ndarray,
    current_xyz: np.ndarray,
    title: str,
    out_path: Path,
    marker_size: float,
    edge_line_width: float,
    edge_opacity: float,
) -> None:
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'}]])
    _add_frame_traces(
        fig,
        frame=frame,
        title='cache',
        row=1,
        col=1,
        geom=geom,
        marker_size=marker_size,
        edge_line_width=edge_line_width,
        edge_opacity=edge_opacity,
        handle_to_name=None,
    )
    _add_action_traces(fig, pred_xyz=pred_abs[:, :3], target_xyz=target_abs[:, :3], current_xyz=current_xyz[:3])
    ranges = _action_scene_ranges(frame, pred_abs[:, :3], target_abs[:, :3])
    fig.update_layout(
        title=title,
        margin={'l': 0, 'r': 0, 't': 54, 'b': 0},
        scene={
            'xaxis': {'range': ranges['x'], 'title': 'x'},
            'yaxis': {'range': ranges['y'], 'title': 'y'},
            'zaxis': {'range': ranges['z'], 'title': 'z'},
            'aspectmode': 'cube',
        },
        legend={'orientation': 'h'},
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs='cdn')


def _array_stats(value: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(value, dtype=np.float32)
    return {
        'shape': [int(x) for x in arr.shape],
        'mean': float(np.mean(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Visualize cached point clouds with predicted and ground-truth action chunks.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--variation', type=int, default=0)
    parser.add_argument('--episodes', default='0', help='Comma-separated query episode ids.')
    parser.add_argument('--query-starts', default='0', help='Comma-separated query t0 indices.')
    parser.add_argument('--support-episodes', default='', help='Optional comma-separated support episode ids for support-conditioned checkpoints.')
    parser.add_argument('--query-window-mode', default='online_history', choices=('online_history', 'forward'))
    parser.add_argument('--query-stride-mode', default='dataset', choices=('dataset', 'consecutive'))
    parser.add_argument('--K', type=int, default=0)
    parser.add_argument('--L', type=int, default=0)
    parser.add_argument('--T_obs', type=int, default=0)
    parser.add_argument('--H', type=int, default=0)
    parser.add_argument('--stride', type=int, default=0)
    parser.add_argument('--traj_len', type=int, default=0)
    parser.add_argument('--support-spacetime-points', dest='support_spacetime_points', type=int, default=0)
    parser.add_argument('--support-spacetime-sampling', dest='support_spacetime_sampling', default='')
    parser.add_argument('--supernodes', type=int, default=-1)
    parser.add_argument('--supernode-temperature', type=float, default=-1.0)
    parser.add_argument('--edge-top-k', type=int, default=8)
    parser.add_argument('--max-edge-supernodes', type=int, default=64)
    parser.add_argument('--skip-self-edges', dest='skip_self_edges', action='store_true')
    parser.add_argument('--show-self-edges', dest='skip_self_edges', action='store_false')
    parser.set_defaults(skip_self_edges=True)
    parser.add_argument('--edge-line-width', type=float, default=3.0)
    parser.add_argument('--edge-opacity', type=float, default=1.0)
    parser.add_argument('--edge-min-length', type=float, default=0.0)
    parser.add_argument('--edge-candidate-multiplier', type=int, default=1)
    parser.add_argument('--marker-size', type=float, default=1.5)
    parser.add_argument('--out-dir', default='eval_outputs/action_chunk_diagnostics')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    cache_root = Path(args.cache_root).expanduser().resolve()
    ckpt = load_checkpoint(ckpt_path)
    ckpt_cfg = _checkpoint_config(ckpt)
    data_cfg = _data_config_from_checkpoint(ckpt_cfg, args)
    rng = np.random.default_rng(0)
    keys = build_variation_keys(cache_root, str(args.task))
    matches = [key for key in keys if int(key.variation) == int(args.variation)]
    if not matches:
        available = sorted({int(key.variation) for key in keys})
        raise RuntimeError(f'No variation {args.variation} for task {args.task!r}. Available: {available[:20]}')
    key = matches[0]
    store = RLBenchCacheStore([key], keep_open=True, preload_to_memory=False)
    num_points, state_dim, action_dim = store.infer_dims()
    policy_cfg = policy_config_from(ckpt_cfg.model, H=data_cfg.H, data_cfg=data_cfg)
    if int(args.supernodes) > 0:
        policy_cfg = replace(policy_cfg, encoder=replace(policy_cfg.encoder, supernodes=int(args.supernodes)))
    num_supernodes = int(policy_cfg.encoder.supernodes)
    temperature = float(args.supernode_temperature) if float(args.supernode_temperature) > 0.0 else float(policy_cfg.encoder.supernode_temperature)
    fixed_conditioning = _fixed_task_variation_ids(ckpt_cfg, str(args.task), int(key.variation))
    model = DirectRegressionPolicy(policy_cfg, state_dim=state_dim, action_dim=action_dim)

    @jax.jit
    def predict(params, batch):
        return model.apply({'params': params}, batch, train=False, return_attn_stats=True)

    query_episodes = _parse_optional_int_list(args.episodes)
    query_starts = _parse_optional_int_list(args.query_starts)
    support_episode_override = _parse_optional_int_list(args.support_episodes)
    if not query_episodes:
        query_episodes = tuple(int(x) for x in store.list_episode_ids(0)[:1])
    if not query_starts:
        query_starts = (0,)
    available_episodes = tuple(int(x) for x in store.list_episode_ids(0))
    uses_support = str(policy_cfg.conditioning.mode) in ('support', 'support_summary_film') and bool(policy_cfg.encoder.use_support_tokens)
    load_rgb = bool(policy_cfg.encoder.use_rgb)
    query_load_mask_id = True
    needs_center_mask = (
        str(getattr(policy_cfg.encoder, 'encoder_type', 'perceiver')) == 'supernode'
        and str(getattr(policy_cfg.encoder, 'supernode_center_sampling', 'linspace')) == 'mask_balanced'
    )
    support_load_mask_id = bool(policy_cfg.encoder.use_mask_id) or needs_center_mask or (
        str(getattr(policy_cfg.encoder, 'support_tokenizer', 'frame')) == 'spacetime_supernode'
        and int(data_cfg.support_spacetime_points) > 0
        and str(data_cfg.support_spacetime_sampling) == 'mask_balanced'
    )

    out_root = Path(args.out_dir).expanduser().resolve() / f'{args.task}_var{int(key.variation)}'
    out_root.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {
        'checkpoint': str(ckpt_path),
        'checkpoint_step': int(ckpt.get('step', -1)),
        'cache_root': str(cache_root),
        'task': str(args.task),
        'variation': int(key.variation),
        'num_points': int(num_points),
        'state_dim': int(state_dim),
        'action_dim': int(action_dim),
        'data': {
            'K': int(data_cfg.K),
            'L': int(data_cfg.L),
            'T_obs': int(data_cfg.T_obs),
            'H': int(data_cfg.H),
            'stride': int(data_cfg.stride),
            'traj_len': int(data_cfg.traj_len),
            'support_spacetime_points': int(data_cfg.support_spacetime_points),
            'support_spacetime_sampling': str(data_cfg.support_spacetime_sampling),
            'action_representation': str(data_cfg.action_representation),
            'query_stride_mode': str(args.query_stride_mode),
            'query_window_mode': str(args.query_window_mode),
        },
        'model': {
            'encoder_type': str(policy_cfg.encoder.encoder_type),
            'conditioning_mode': str(policy_cfg.conditioning.mode),
            'decoder_context_mode': str(policy_cfg.decoder.context_mode),
            'uses_support': bool(uses_support),
            'use_rgb': bool(policy_cfg.encoder.use_rgb),
            'use_mask_id': bool(policy_cfg.encoder.use_mask_id),
            'supernodes': int(num_supernodes),
            'supernode_temperature': float(temperature),
            'supernode_center_sampling': str(getattr(policy_cfg.encoder, 'supernode_center_sampling', 'linspace')),
        },
        'examples': [],
    }

    try:
        for episode_id in query_episodes:
            if int(episode_id) not in available_episodes:
                raise RuntimeError(f'Episode {episode_id} is not available. Available episodes: {available_episodes}')
            support: Dict[str, np.ndarray] = {}
            support_ids: Tuple[int, ...] = ()
            if uses_support:
                support_ids = _choose_support_ids(available_episodes, int(episode_id), int(data_cfg.K), support_episode_override)
                support = _support_conditioning(
                    store,
                    vidx=0,
                    support_ids=support_ids,
                    cfg=data_cfg,
                    load_rgb=load_rgb,
                    load_mask_id=support_load_mask_id,
                    rng=rng,
                )
            for t0 in query_starts:
                query, meta = _query_sample(
                    store,
                    vidx=0,
                    episode_id=int(episode_id),
                    t0=int(t0),
                    cfg=data_cfg,
                    query_stride_mode=str(args.query_stride_mode),
                    query_window_mode=str(args.query_window_mode),
                    load_rgb=load_rgb,
                    load_mask_id=query_load_mask_id,
                )
                sample = {**support, **query}
                batch = _batchify(sample, fixed_conditioning)
                pred_encoded, attn_stats = predict(ckpt['params'], batch)
                pred_encoded_np = np.asarray(jax.device_get(pred_encoded), dtype=np.float32)
                stats_np = {
                    key: float(np.asarray(jax.device_get(value)))
                    for key, value in jax.device_get(attn_stats).items()
                }
                target_encoded_np = np.asarray(batch['target_action'], dtype=np.float32)
                query_state = np.asarray(batch['query_state'], dtype=np.float32)
                pred_abs = decode_action_chunk(
                    pred_encoded_np,
                    query_state=query_state,
                    representation=str(data_cfg.action_representation),
                )[0]
                target_abs = decode_action_chunk(
                    target_encoded_np,
                    query_state=query_state,
                    representation=str(data_cfg.action_representation),
                )[0]
                encoded_mse = float(np.mean(np.square(pred_encoded_np[0] - target_encoded_np[0])))
                xyz_mse = float(np.mean(np.square(pred_abs[:, :3] - target_abs[:, :3])))
                first_xyz_error = float(np.linalg.norm(pred_abs[0, :3] - target_abs[0, :3]))
                last_query = int(data_cfg.T_obs) - 1
                frame = {
                    'xyz': np.asarray(query['query_xyz'][last_query], dtype=np.float32),
                    'valid': np.asarray(query['query_valid'][last_query], dtype=np.bool_),
                }
                if 'query_rgb' in query:
                    frame['rgb'] = np.asarray(query['query_rgb'][last_query], dtype=np.float32)
                if 'query_mask_id' in query:
                    frame['mask_id'] = np.asarray(query['query_mask_id'][last_query], dtype=np.int32)
                geom = _supernode_geometry(
                    frame['xyz'],
                    frame['valid'],
                    num_supernodes=num_supernodes,
                    temperature=temperature,
                    edge_top_k=int(args.edge_top_k),
                    max_edge_supernodes=int(args.max_edge_supernodes),
                    skip_self_edges=bool(args.skip_self_edges),
                    edge_min_length=float(args.edge_min_length),
                    edge_candidate_multiplier=int(args.edge_candidate_multiplier),
                )
                out_path = out_root / f'ep{int(episode_id):04d}_t{int(t0):04d}.html'
                title = (
                    f'{args.task} var {int(key.variation)} ep {int(episode_id)} t0 {int(t0)} | '
                    f'xyz_mse={xyz_mse:.6f} first_xyz_error={first_xyz_error:.4f}'
                )
                _write_html(
                    frame=frame,
                    geom=geom,
                    pred_abs=pred_abs,
                    target_abs=target_abs,
                    current_xyz=np.asarray(query['query_state'][-1, :3], dtype=np.float32),
                    title=title,
                    out_path=out_path,
                    marker_size=float(args.marker_size),
                    edge_line_width=float(args.edge_line_width),
                    edge_opacity=float(args.edge_opacity),
                )
                summary['examples'].append(
                    {
                        'path': str(out_path),
                        'episode_id': int(episode_id),
                        'support_episode_ids': [int(x) for x in support_ids],
                        't0': int(t0),
                        **meta,
                        'encoded_mse': encoded_mse,
                        'xyz_mse': xyz_mse,
                        'first_xyz_error': first_xyz_error,
                        'current_xyz': [float(x) for x in np.asarray(query['query_state'][-1, :3], dtype=np.float32).tolist()],
                        'pred_first_xyz': [float(x) for x in pred_abs[0, :3].tolist()],
                        'target_first_xyz': [float(x) for x in target_abs[0, :3].tolist()],
                        'pred_action': _array_stats(pred_abs),
                        'target_action': _array_stats(target_abs),
                        'attn_stats': stats_np,
                        'supernode_edges': int(geom['edge_start'].shape[0]),
                    }
                )
    finally:
        store.close()

    summary_path = out_root / 'summary.json'
    with summary_path.open('w', encoding='utf-8') as file:
        json.dump(summary, file, indent=2)
    print(f'Wrote {len(summary["examples"])} action chunk diagnostic HTML files under {out_root}')
    print(f'Wrote summary: {summary_path}')


if __name__ == '__main__':
    main()
