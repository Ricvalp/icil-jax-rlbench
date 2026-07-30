from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from absl import app
import jax
import numpy as np
import optax
import plotly.graph_objects as go
from ml_collections import ConfigDict
from ml_collections.config_flags import config_flags

from icil_jax_rlbench.data.action_representation import decode_action_chunk
from icil_jax_rlbench.data.h5_cache import RLBenchCacheStore
from icil_jax_rlbench.data.sampler import ICILSampler
from icil_jax_rlbench.eval.action_chunk_diagnostics import _batchify, _query_sample
from icil_jax_rlbench.eval.action_chunk_plots import supernode_geometry
from icil_jax_rlbench.eval.online_common import (
    _build_param_inner_batch,
    _choose_variation_key,
    _checkpoint_config,
    _data_config_from_eval_and_checkpoint,
    _load_mask_id_for_cached_batches,
    _support_cache_root,
    _task_variation_ids_from_checkpoint,
)
from icil_jax_rlbench.models.config import policy_config_from
from icil_jax_rlbench.models.direct_regression_policy import DirectRegressionPolicy
from icil_jax_rlbench.train.checkpoints import load_checkpoint
from icil_jax_rlbench.train.step import (
    action_loss,
    apply_mask,
    clip_tree_by_global_norm,
    make_maml_inner_mask,
)


_CONFIG = config_flags.DEFINE_config_file(
    'config',
    default='icil_jax_rlbench/configs/eval_param_maml_adaptation_action_chunks.py',
    help_string='Path to cached param-MAML adaptation action-chunk visualization config.',
)


def _as_int_tuple(value: Any) -> Tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return ()
        return tuple(int(x.strip()) for x in value.split(',') if x.strip())
    return tuple(int(x) for x in value)


def _query_window_mode(cfg: ConfigDict, checkpoint_mode: str) -> str:
    mode = str(getattr(getattr(cfg, 'query', ConfigDict()), 'window_mode', 'checkpoint'))
    if mode == 'checkpoint':
        return str(checkpoint_mode)
    if mode not in ('online_history', 'forward'):
        raise ValueError("query.window_mode must be 'checkpoint', 'online_history', or 'forward'.")
    return mode


def _select_query_episodes(
    available: Sequence[int],
    requested: Sequence[int],
    support_ids: Sequence[int],
) -> Tuple[int, ...]:
    if requested:
        out = tuple(int(x) for x in requested)
    else:
        support_set = {int(x) for x in support_ids}
        out = tuple(int(x) for x in available if int(x) not in support_set)[:1]
    if not out:
        raise RuntimeError('Could not choose a query episode.')
    available_set = {int(x) for x in available}
    missing = [int(x) for x in out if int(x) not in available_set]
    if missing:
        raise RuntimeError(f'Query episodes are not available in this variation: {missing}')
    return out


def _select_support_ids(
    available: Sequence[int],
    query_episodes: Sequence[int],
    K: int,
    explicit: Sequence[int],
    rng: np.random.Generator,
) -> Tuple[int, ...]:
    if explicit:
        support = tuple(int(x) for x in explicit)
        if len(support) != int(K):
            raise ValueError(f'Expected exactly K={K} support episodes, got {len(support)}.')
    else:
        query_set = {int(x) for x in query_episodes}
        candidates = np.asarray([int(x) for x in available if int(x) not in query_set], dtype=np.int64)
        if candidates.shape[0] < int(K):
            raise RuntimeError(f'Need K={K} support episodes besides query episodes, found {candidates.shape[0]}.')
        support = tuple(int(x) for x in rng.choice(candidates, size=int(K), replace=False).tolist())
    available_set = {int(x) for x in available}
    missing = [int(x) for x in support if int(x) not in available_set]
    if missing:
        raise RuntimeError(f'Support episodes are not available in this variation: {missing}')
    if len(set(support)) != len(support):
        raise ValueError(f'Support episodes must be unique, got {support}.')
    return support


def _make_support_batch(
    sampler: ICILSampler,
    support_ids: Sequence[int],
    *,
    load_rgb: bool,
    load_mask_id: bool,
) -> Dict[str, np.ndarray]:
    support = sampler.build_support_conditioning(
        vidx=0,
        support_ids=np.asarray(support_ids, dtype=np.int64),
        load_rgb=load_rgb,
        load_mask_id=load_mask_id,
    )
    return {k: np.expand_dims(v, axis=0) for k, v in support.items()}


def _merge_support_query(
    support_batch: Dict[str, np.ndarray],
    query_sample: Dict[str, np.ndarray],
    fixed_conditioning: Optional[Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    query_batch = _batchify(query_sample, fixed_conditioning)
    out = dict(support_batch)
    out.update(query_batch)
    return out


def _make_predict_fn(model: DirectRegressionPolicy):
    @jax.jit
    def predict(params: Any, batch: Dict[str, np.ndarray]):
        return model.apply({'params': params}, batch, train=False)

    return predict


def _make_adapt_step_fn(
    model: DirectRegressionPolicy,
    *,
    inner_lr: float,
    grad_clip_norm: float,
    loss_type: str,
    inner_mask: Any,
):
    def loss_fn(params: Any, batch: Dict[str, np.ndarray]):
        pred = model.apply({'params': params}, batch, train=False)
        return action_loss(pred, batch['target_action'], loss_type)

    def step(params: Any, batch: Dict[str, np.ndarray]):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        grads = apply_mask(grads, inner_mask)
        grads, grad_norm = clip_tree_by_global_norm(grads, float(grad_clip_norm))
        updates = jax.tree_util.tree_map(lambda g: -float(inner_lr) * g, grads)
        updated = optax.apply_updates(params, updates)
        return updated, {'inner_loss': loss, 'inner_grad_norm': grad_norm}

    return jax.jit(step)


def _predict_abs_chunk(
    predict_fn: Any,
    params: Any,
    batch: Dict[str, np.ndarray],
    *,
    action_representation: str,
) -> Tuple[np.ndarray, np.ndarray]:
    pred_encoded = predict_fn(params, batch)
    pred_encoded_np = np.asarray(jax.device_get(pred_encoded), dtype=np.float32)
    pred_abs = decode_action_chunk(
        pred_encoded_np,
        query_state=np.asarray(batch['query_state'], dtype=np.float32),
        representation=str(action_representation),
    )[0]
    return pred_encoded_np[0], pred_abs


def _decode_target_abs(batch: Dict[str, np.ndarray], *, action_representation: str) -> Tuple[np.ndarray, np.ndarray]:
    target_encoded = np.asarray(batch['target_action'], dtype=np.float32)
    target_abs = decode_action_chunk(
        target_encoded,
        query_state=np.asarray(batch['query_state'], dtype=np.float32),
        representation=str(action_representation),
    )[0]
    return target_encoded[0], target_abs


def _rgb_strings(rgb: Optional[np.ndarray], count: int) -> List[str]:
    if rgb is None:
        return ['rgb(70,120,220)'] * int(count)
    arr = np.asarray(rgb)
    if arr.dtype != np.uint8:
        arr = np.clip(arr.astype(np.float32), 0.0, 1.0)
        arr = np.rint(255.0 * arr).astype(np.uint8)
    arr = arr.reshape((-1, 3))
    return [f'rgb({int(r)},{int(g)},{int(b)})' for r, g, b in arr[:count]]


def _axis_ranges(frame: Dict[str, np.ndarray], chunks: Sequence[np.ndarray], current_xyz: np.ndarray) -> Dict[str, Tuple[float, float]]:
    xyz = np.asarray(frame['xyz'], dtype=np.float32).reshape((-1, 3))
    valid = np.asarray(frame.get('valid', np.ones((xyz.shape[0],), dtype=np.bool_))).reshape(-1).astype(np.bool_)
    pieces = [xyz[valid], np.asarray(current_xyz, dtype=np.float32).reshape(1, 3)]
    for chunk in chunks:
        arr = np.asarray(chunk, dtype=np.float32)
        if arr.size:
            pieces.append(arr.reshape((-1, arr.shape[-1]))[:, :3])
    all_pts = np.concatenate([p for p in pieces if p.size > 0], axis=0)
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


def _make_query_frame(query_sample: Dict[str, np.ndarray], data_cfg: Any) -> Dict[str, np.ndarray]:
    last = int(data_cfg.T_obs) - 1
    frame: Dict[str, np.ndarray] = {
        'xyz': np.asarray(query_sample['query_xyz'][last], dtype=np.float32),
        'valid': np.asarray(query_sample['query_valid'][last], dtype=np.bool_),
    }
    if 'query_rgb' in query_sample:
        frame['rgb'] = np.asarray(query_sample['query_rgb'][last], dtype=np.float32)
    if 'query_mask_id' in query_sample:
        frame['mask_id'] = np.asarray(query_sample['query_mask_id'][last], dtype=np.int32)
    return frame


def _write_adaptation_html(
    *,
    frame: Dict[str, np.ndarray],
    current_xyz: np.ndarray,
    target_abs: np.ndarray,
    predictions: Sequence[Dict[str, Any]],
    policy_cfg: Any,
    plot_cfg: ConfigDict,
    title: str,
    out_path: Path,
) -> None:
    xyz = np.asarray(frame['xyz'], dtype=np.float32).reshape((-1, 3))
    valid = np.asarray(frame.get('valid', np.ones((xyz.shape[0],), dtype=np.bool_))).reshape(-1).astype(np.bool_)
    rgb = frame.get('rgb')
    if rgb is not None:
        rgb = np.asarray(rgb).reshape((-1, 3))[valid]
    mask_id = frame.get('mask_id')
    if mask_id is not None:
        mask_id = np.asarray(mask_id).reshape((-1,))[valid]
    xyz = xyz[valid]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            mode='markers',
            marker={
                'size': float(plot_cfg.marker_size),
                'color': _rgb_strings(rgb, xyz.shape[0]),
                'opacity': float(plot_cfg.point_opacity),
            },
            text=[f'mask_id={int(x)}' for x in mask_id] if mask_id is not None else None,
            hovertemplate='%{text}<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>' if mask_id is not None else 'x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>',
            name='query point cloud',
        )
    )

    if bool(getattr(plot_cfg, 'show_supernodes', True)) and str(policy_cfg.encoder.encoder_type) == 'supernode':
        geom = supernode_geometry(
            frame['xyz'],
            frame.get('valid'),
            num_supernodes=int(policy_cfg.encoder.supernodes),
            temperature=float(policy_cfg.encoder.supernode_temperature),
            edge_top_k=int(plot_cfg.edge_top_k),
            max_edge_supernodes=int(plot_cfg.max_edge_supernodes),
            skip_self_edges=bool(plot_cfg.skip_self_edges),
            edge_min_length=float(plot_cfg.edge_min_length),
            edge_candidate_multiplier=int(plot_cfg.edge_candidate_multiplier),
        )
        centers = np.asarray(geom['centers'], dtype=np.float32)
        if centers.size:
            fig.add_trace(
                go.Scatter3d(
                    x=centers[:, 0],
                    y=centers[:, 1],
                    z=centers[:, 2],
                    mode='markers',
                    marker={'size': float(plot_cfg.marker_size) * 3.0, 'color': 'black', 'symbol': 'diamond', 'opacity': 0.9},
                    name='query supernodes',
                )
            )
        edge_start = np.asarray(geom['edge_start'], dtype=np.float32)
        edge_end = np.asarray(geom['edge_end'], dtype=np.float32)
        if edge_start.shape[0] > 0 and edge_end.shape[0] == edge_start.shape[0]:
            edge_x: List[Optional[float]] = []
            edge_y: List[Optional[float]] = []
            edge_z: List[Optional[float]] = []
            for start, end in zip(edge_start, edge_end):
                edge_x.extend([float(start[0]), float(end[0]), None])
                edge_y.extend([float(start[1]), float(end[1]), None])
                edge_z.extend([float(start[2]), float(end[2]), None])
            fig.add_trace(
                go.Scatter3d(
                    x=edge_x,
                    y=edge_y,
                    z=edge_z,
                    mode='lines',
                    line={'color': 'rgb(255,80,0)', 'width': float(plot_cfg.edge_line_width)},
                    opacity=float(plot_cfg.edge_opacity),
                    name='query supernode edges',
                    hoverinfo='skip',
                    connectgaps=False,
                )
            )

    current_xyz = np.asarray(current_xyz, dtype=np.float32).reshape(-1)[:3]
    fig.add_trace(
        go.Scatter3d(
            x=[float(current_xyz[0])],
            y=[float(current_xyz[1])],
            z=[float(current_xyz[2])],
            mode='markers',
            marker={'size': 8, 'color': '#16a34a', 'symbol': 'cross'},
            name='current ee',
        )
    )

    target = np.asarray(target_abs, dtype=np.float32)
    fig.add_trace(
        go.Scatter3d(
            x=target[:, 0],
            y=target[:, 1],
            z=target[:, 2],
            mode='lines+markers',
            line={'color': '#111827', 'width': 10},
            marker={'size': 5, 'color': '#111827'},
            name='ground truth',
        )
    )

    palette = [
        '#94a3b8',
        '#ef4444',
        '#f97316',
        '#eab308',
        '#22c55e',
        '#06b6d4',
        '#3b82f6',
        '#8b5cf6',
        '#ec4899',
    ]
    for pred in predictions:
        step = int(pred['step'])
        chunk = np.asarray(pred['action_abs'], dtype=np.float32)
        color = palette[min(step, len(palette) - 1)]
        width = 5 if step == 0 else 7
        fig.add_trace(
            go.Scatter3d(
                x=chunk[:, 0],
                y=chunk[:, 1],
                z=chunk[:, 2],
                mode='lines+markers',
                line={'color': color, 'width': width},
                marker={'size': 3 if step == 0 else 4, 'color': color},
                name=f'pred after {step} updates',
                hovertemplate=(
                    f'inner_step={step}<br>'
                    f'xyz_mse={float(pred["xyz_mse"]):.6f}<br>'
                    f'first_xyz_error={float(pred["first_xyz_error"]):.4f}<br>'
                    'x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>'
                ),
            )
        )

    ranges = _axis_ranges(frame, [target_abs] + [p['action_abs'] for p in predictions], current_xyz)
    fig.update_layout(
        title=title,
        margin={'l': 0, 'r': 0, 't': 58, 'b': 0},
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


def _prediction_metrics(pred_abs: np.ndarray, pred_encoded: np.ndarray, target_abs: np.ndarray, target_encoded: np.ndarray) -> Dict[str, float]:
    return {
        'encoded_mse': float(np.mean(np.square(pred_encoded - target_encoded))),
        'encoded_l1': float(np.mean(np.abs(pred_encoded - target_encoded))),
        'xyz_mse': float(np.mean(np.square(pred_abs[:, :3] - target_abs[:, :3]))),
        'xyz_l2_mean': float(np.mean(np.linalg.norm(pred_abs[:, :3] - target_abs[:, :3], axis=-1))),
        'first_xyz_error': float(np.linalg.norm(pred_abs[0, :3] - target_abs[0, :3])),
        'last_xyz_error': float(np.linalg.norm(pred_abs[-1, :3] - target_abs[-1, :3])),
    }


def run(cfg: ConfigDict) -> None:
    seed = int(cfg.seed)
    rng = np.random.default_rng(seed)
    checkpoint_path = Path(str(cfg.checkpoint_path)).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    ckpt = load_checkpoint(checkpoint_path)
    ckpt_cfg = _checkpoint_config(ckpt)
    if str(getattr(ckpt_cfg, 'mode', '')) != 'param_maml':
        raise ValueError(f'Expected a param_maml checkpoint, got mode={getattr(ckpt_cfg, "mode", None)!r}.')

    data_cfg = _data_config_from_eval_and_checkpoint(cfg, ckpt)
    query_window = _query_window_mode(cfg, str(data_cfg.query_window_mode))
    query_stride = str(getattr(getattr(cfg, 'query', ConfigDict()), 'stride_mode', 'dataset'))
    if query_stride not in ('dataset', 'consecutive'):
        raise ValueError("query.stride_mode must be 'dataset' or 'consecutive'.")

    cache_root = _support_cache_root(cfg, ckpt)
    task_name = str(cfg.task.name)
    variation = int(cfg.task.variation)
    key = _choose_variation_key(cache_root, task_name, variation, rng)
    store = RLBenchCacheStore(
        [key],
        keep_open=bool(getattr(getattr(cfg, 'conditioning', ConfigDict()), 'keep_open', True)),
        preload_to_memory=False,
    )
    sampler = ICILSampler(store, data_cfg, seed=seed + 17)

    try:
        num_points, state_dim, action_dim = store.infer_dims()
        policy_cfg = policy_config_from(ckpt_cfg.model, H=data_cfg.H, data_cfg=data_cfg)
        fixed_conditioning = _task_variation_ids_from_checkpoint(ckpt_cfg, task_name, int(key.variation))
        uses_support = str(policy_cfg.conditioning.mode) in ('support', 'support_summary_film') and bool(policy_cfg.encoder.use_support_tokens)
        if not uses_support:
            raise ValueError('This diagnostic expects a support-conditioned param-MAML checkpoint.')
        model = DirectRegressionPolicy(policy_cfg, state_dim=state_dim, action_dim=action_dim)
        params0 = jax.device_put(ckpt['params'])

        use_rgb = bool(policy_cfg.encoder.use_rgb)
        load_mask_id = _load_mask_id_for_cached_batches(policy_cfg, data_cfg)
        available_episodes = tuple(int(x) for x in store.list_episode_ids(0))
        explicit_support = _as_int_tuple(getattr(getattr(cfg, 'support', ConfigDict()), 'episodes', ()))
        requested_queries = _as_int_tuple(getattr(getattr(cfg, 'query', ConfigDict()), 'episodes', ()))
        query_episodes = _select_query_episodes(available_episodes, requested_queries, explicit_support)
        support_ids = _select_support_ids(available_episodes, query_episodes, int(data_cfg.K), explicit_support, rng)
        t0s = _as_int_tuple(getattr(getattr(cfg, 'query', ConfigDict()), 't0s', (0,)))
        if not t0s:
            t0s = (0,)

        support_batch = _make_support_batch(
            sampler,
            support_ids,
            load_rgb=use_rgb,
            load_mask_id=load_mask_id,
        )
        maml_cfg = getattr(ckpt_cfg, 'maml', ConfigDict())
        eval_adapt = getattr(cfg, 'adaptation', ConfigDict())
        inner_steps = int(getattr(eval_adapt, 'inner_steps_override', -1))
        if inner_steps < 0:
            inner_steps = int(getattr(maml_cfg, 'inner_steps', 1))
        num_inner_queries = int(getattr(eval_adapt, 'num_inner_queries', 0))
        if num_inner_queries <= 0:
            num_inner_queries = int(getattr(maml_cfg, 'num_inner_queries', data_cfg.K))
        inner_lr = float(getattr(eval_adapt, 'inner_lr', 0.0))
        if inner_lr <= 0.0:
            inner_lr = float(getattr(maml_cfg, 'inner_lr', 1e-2))
        grad_clip = float(getattr(eval_adapt, 'grad_clip_norm', 0.0))
        if grad_clip <= 0.0:
            grad_clip = float(getattr(maml_cfg, 'inner_grad_clip_norm', 1.0))

        inner_batch = _build_param_inner_batch(
            sampler=sampler,
            support_ids=support_ids,
            inner_steps=inner_steps,
            num_inner_queries=num_inner_queries,
            rng=rng,
            load_rgb=use_rgb,
            load_mask_id=load_mask_id,
        )
        inner_mask = make_maml_inner_mask(
            params0,
            preset=str(getattr(maml_cfg, 'fast_param_preset', 'name')),
            include=tuple(getattr(maml_cfg, 'inner_param_include', ())),
            exclude=tuple(getattr(maml_cfg, 'inner_param_exclude', ())),
            decoder_layers=int(policy_cfg.decoder.n_layers),
            top_layers=int(getattr(maml_cfg, 'fast_param_top_layers', 2)),
        )
        predict_fn = _make_predict_fn(model)
        adapt_step_fn = _make_adapt_step_fn(
            model,
            inner_lr=inner_lr,
            grad_clip_norm=grad_clip,
            loss_type=str(getattr(ckpt_cfg.train, 'loss_type', 'mse')),
            inner_mask=inner_mask,
        )

        run_dir = Path(str(cfg.output.root_dir)).expanduser().resolve() / f'{task_name}_var{int(key.variation)}_{time.strftime("%Y%m%d-%H%M%S")}'
        run_dir.mkdir(parents=True, exist_ok=True)
        summary: Dict[str, Any] = {
            'checkpoint_path': str(checkpoint_path),
            'checkpoint_step': int(ckpt.get('step', -1)),
            'cache_root': str(cache_root),
            'task': task_name,
            'variation': int(key.variation),
            'query_episodes': [int(x) for x in query_episodes],
            'query_t0s': [int(x) for x in t0s],
            'support_episodes': [int(x) for x in support_ids],
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
                'action_representation': str(data_cfg.action_representation),
                'query_window_mode': query_window,
                'query_stride_mode': query_stride,
                'support_spacetime_points': int(data_cfg.support_spacetime_points),
                'support_spacetime_sampling': str(data_cfg.support_spacetime_sampling),
            },
            'adaptation': {
                'inner_steps': int(inner_steps),
                'num_inner_queries': int(num_inner_queries),
                'inner_lr': float(inner_lr),
                'grad_clip_norm': float(grad_clip),
                'fast_param_preset': str(getattr(maml_cfg, 'fast_param_preset', 'name')),
                'fast_param_top_layers': int(getattr(maml_cfg, 'fast_param_top_layers', 2)),
            },
            'model': {
                'encoder_type': str(policy_cfg.encoder.encoder_type),
                'conditioning_mode': str(policy_cfg.conditioning.mode),
                'decoder_context_mode': str(policy_cfg.decoder.context_mode),
                'use_rgb': bool(policy_cfg.encoder.use_rgb),
                'use_mask_id': bool(policy_cfg.encoder.use_mask_id),
                'supernodes': int(getattr(policy_cfg.encoder, 'supernodes', 0)),
                'supernode_temperature': float(getattr(policy_cfg.encoder, 'supernode_temperature', 0.005)),
            },
            'examples': [],
        }
        with (run_dir / 'resolved_config.json').open('w', encoding='utf-8') as file:
            json.dump(cfg.to_dict(), file, indent=2)

        for episode_id in query_episodes:
            for t0 in t0s:
                query_sample, query_meta = _query_sample(
                    store,
                    vidx=0,
                    episode_id=int(episode_id),
                    t0=int(t0),
                    cfg=data_cfg,
                    query_stride_mode=query_stride,
                    query_window_mode=query_window,
                    load_rgb=use_rgb,
                    load_mask_id=load_mask_id,
                )
                batch = _merge_support_query(support_batch, query_sample, fixed_conditioning)
                target_encoded, target_abs = _decode_target_abs(
                    batch,
                    action_representation=str(data_cfg.action_representation),
                )
                frame = _make_query_frame(query_sample, data_cfg)
                current_xyz = np.asarray(query_sample['query_state'][-1, :3], dtype=np.float32)

                params = params0
                predictions: List[Dict[str, Any]] = []
                adapt_metrics: List[Dict[str, float]] = []
                for step_idx in range(int(inner_steps) + 1):
                    pred_encoded, pred_abs = _predict_abs_chunk(
                        predict_fn,
                        params,
                        batch,
                        action_representation=str(data_cfg.action_representation),
                    )
                    metrics = _prediction_metrics(pred_abs, pred_encoded, target_abs, target_encoded)
                    predictions.append({'step': int(step_idx), 'action_abs': pred_abs, 'action_encoded': pred_encoded, **metrics})
                    if step_idx < int(inner_steps):
                        step_batch = {k: v[int(step_idx)] for k, v in inner_batch.items()}
                        params, step_metrics = adapt_step_fn(params, step_batch)
                        adapt_metrics.append({k: float(jax.device_get(v)) for k, v in step_metrics.items()})

                stem = f'ep{int(episode_id):04d}_t{int(t0):04d}'
                html_path = run_dir / f'{stem}.html'
                title = (
                    f'{task_name} var {int(key.variation)} ep {int(episode_id)} t0 {int(t0)} | '
                    f'support={list(map(int, support_ids))} | inner_steps={int(inner_steps)}'
                )
                _write_adaptation_html(
                    frame=frame,
                    current_xyz=current_xyz,
                    target_abs=target_abs,
                    predictions=predictions,
                    policy_cfg=policy_cfg,
                    plot_cfg=cfg.plot,
                    title=title,
                    out_path=html_path,
                )
                npz_path = run_dir / f'{stem}_chunks.npz'
                np.savez_compressed(
                    npz_path,
                    target_abs=target_abs.astype(np.float32),
                    target_encoded=target_encoded.astype(np.float32),
                    pred_abs=np.stack([p['action_abs'] for p in predictions], axis=0).astype(np.float32),
                    pred_encoded=np.stack([p['action_encoded'] for p in predictions], axis=0).astype(np.float32),
                    current_xyz=current_xyz.astype(np.float32),
                    obs_idx=np.asarray(query_meta['obs_idx'], dtype=np.int32),
                    act_idx=np.asarray(query_meta['act_idx'], dtype=np.int32),
                    support_episodes=np.asarray(support_ids, dtype=np.int32),
                )
                summary['examples'].append(
                    {
                        'html_path': str(html_path),
                        'npz_path': str(npz_path),
                        'episode_id': int(episode_id),
                        't0': int(t0),
                        **query_meta,
                        'current_xyz': [float(x) for x in current_xyz.tolist()],
                        'adapt_metrics': adapt_metrics,
                        'prediction_metrics': [
                            {
                                'step': int(p['step']),
                                'encoded_mse': float(p['encoded_mse']),
                                'encoded_l1': float(p['encoded_l1']),
                                'xyz_mse': float(p['xyz_mse']),
                                'xyz_l2_mean': float(p['xyz_l2_mean']),
                                'first_xyz_error': float(p['first_xyz_error']),
                                'last_xyz_error': float(p['last_xyz_error']),
                            }
                            for p in predictions
                        ],
                    }
                )
                print(f'Wrote {html_path}', flush=True)

        summary_path = run_dir / 'summary.json'
        with summary_path.open('w', encoding='utf-8') as file:
            json.dump(summary, file, indent=2)
        print(f'Wrote summary: {summary_path}', flush=True)
    finally:
        store.close()


def main(argv: Sequence[str]) -> None:
    del argv
    run(_CONFIG.value)


if __name__ == '__main__':
    app.run(main)
