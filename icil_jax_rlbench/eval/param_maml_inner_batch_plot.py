from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from absl import app
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ml_collections import ConfigDict
from ml_collections.config_flags import config_flags

from icil_jax_rlbench.data.action_representation import decode_action_chunk
from icil_jax_rlbench.data.h5_cache import RLBenchCacheStore
from icil_jax_rlbench.data.sampler import ICILSampler
from icil_jax_rlbench.eval.online_common import (
    _choose_variation_key,
    _checkpoint_config,
    _data_config_from_eval_and_checkpoint,
    _load_mask_id_for_cached_batches,
    _support_cache_root,
)
from icil_jax_rlbench.models.config import policy_config_from
from icil_jax_rlbench.train.checkpoints import load_checkpoint


_CONFIG = config_flags.DEFINE_config_file(
    'config',
    default='icil_jax_rlbench/configs/eval_param_maml_inner_batch_plot.py',
    help_string='Path to cached param-MAML inner-batch visualization config.',
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


def _select_support_ids(
    available: Sequence[int],
    K: int,
    explicit: Sequence[int],
    rng: np.random.Generator,
) -> Tuple[int, ...]:
    if explicit:
        support = tuple(int(x) for x in explicit)
        if len(support) != int(K):
            raise ValueError(f'Expected exactly K={K} support episodes, got {len(support)}.')
    else:
        candidates = np.asarray([int(x) for x in available], dtype=np.int64)
        if candidates.shape[0] < int(K):
            raise RuntimeError(f'Need K={K} support episodes, found {candidates.shape[0]}.')
        support = tuple(int(x) for x in rng.choice(candidates, size=int(K), replace=False).tolist())
    available_set = {int(x) for x in available}
    missing = [int(x) for x in support if int(x) not in available_set]
    if missing:
        raise RuntimeError(f'Support episodes are not available in this variation: {missing}')
    if len(set(support)) != len(support):
        raise ValueError(f'Support episodes must be unique, got {support}.')
    return support


def _rgb_strings(rgb: Optional[np.ndarray], count: int, fallback: str) -> List[str]:
    if rgb is None:
        return [fallback] * int(count)
    arr = np.asarray(rgb)
    if arr.size == 0:
        return [fallback] * int(count)
    if arr.dtype != np.uint8:
        arr = np.clip(arr.astype(np.float32), 0.0, 1.0)
        arr = np.rint(255.0 * arr).astype(np.uint8)
    arr = arr.reshape((-1, 3))
    return [f'rgb({int(r)},{int(g)},{int(b)})' for r, g, b in arr[:count]]


def _heldout_order(
    support_ids: Sequence[int],
    *,
    num_inner_queries: int,
    step_index: int,
    rng: np.random.Generator,
) -> List[int]:
    support_ids = [int(x) for x in support_ids]
    if len(support_ids) < 2:
        raise ValueError('Param-MAML inner batches need at least K=2 support episodes.')
    order: List[int] = []
    for _ in range(int(step_index) + 1):
        order = list(rng.permutation(len(support_ids)))
        while len(order) < int(num_inner_queries):
            order.extend(list(rng.permutation(len(support_ids))))
    return [int(x) for x in order[: int(num_inner_queries)]]


def _build_inner_examples(
    *,
    sampler: ICILSampler,
    support_ids: Sequence[int],
    num_inner_queries: int,
    step_index: int,
    rng: np.random.Generator,
    load_rgb: bool,
    load_mask_id: bool,
    action_representation: str,
) -> List[Dict[str, Any]]:
    heldout_order = _heldout_order(
        support_ids,
        num_inner_queries=int(num_inner_queries),
        step_index=int(step_index),
        rng=rng,
    )
    examples: List[Dict[str, Any]] = []
    for qidx, holdout_idx in enumerate(heldout_order):
        heldout = int(support_ids[int(holdout_idx)])
        context = [int(eid) for i, eid in enumerate(support_ids) if i != int(holdout_idx)]
        sample = sampler._build_context_query_sample(
            vidx=0,
            context_ids=context,
            query_episode_id=heldout,
            load_rgb=load_rgb,
            load_mask_id=load_mask_id,
        )
        target_abs = decode_action_chunk(
            np.asarray(sample['target_action'], dtype=np.float32)[None],
            query_state=np.asarray(sample['query_state'], dtype=np.float32)[None],
            representation=str(action_representation),
        )[0]
        examples.append(
            {
                'query_index': int(qidx),
                'heldout_episode': heldout,
                'context_episodes': context,
                't0': int(np.asarray(sample['chunk_start']).item()),
                'query_xyz': np.asarray(sample['query_xyz'][-1], dtype=np.float32),
                'query_valid': np.asarray(sample['query_valid'][-1], dtype=np.bool_),
                'query_rgb': np.asarray(sample['query_rgb'][-1], dtype=np.float32) if 'query_rgb' in sample else None,
                'query_mask_id': np.asarray(sample['query_mask_id'][-1], dtype=np.int32) if 'query_mask_id' in sample else None,
                'current_xyz': np.asarray(sample['query_state'][-1, :3], dtype=np.float32),
                'target_abs': np.asarray(target_abs, dtype=np.float32),
            }
        )
    return examples


def _support_trajectories(
    store: RLBenchCacheStore,
    *,
    support_ids: Sequence[int],
) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for eid in support_ids:
        T = store.episode_length(0, int(eid))
        idx = np.arange(int(T), dtype=np.int64)
        item = store.load_episode_slices(0, int(eid), idx, load_rgb=False, load_mask_id=False)
        out[int(eid)] = np.asarray(item['action'], dtype=np.float32)[:, :3]
    return out


def _subsample_points(
    xyz: np.ndarray,
    valid: np.ndarray,
    rgb: Optional[np.ndarray],
    mask_id: Optional[np.ndarray],
    *,
    max_points: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    xyz = np.asarray(xyz, dtype=np.float32).reshape((-1, 3))
    valid = np.asarray(valid, dtype=np.bool_).reshape((-1,))
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return np.zeros((0, 3), dtype=np.float32), None, None
    if int(max_points) > 0 and idx.size > int(max_points):
        idx = np.sort(rng.choice(idx, size=int(max_points), replace=False))
    rgb_out = None
    if rgb is not None:
        rgb_out = np.asarray(rgb).reshape((-1, 3))[idx]
    mask_out = None
    if mask_id is not None:
        mask_out = np.asarray(mask_id).reshape((-1,))[idx]
    return xyz[idx], rgb_out, mask_out


def _axis_ranges(
    examples_by_size: Dict[int, Sequence[Dict[str, Any]]],
    support_traj: Dict[int, np.ndarray],
) -> Dict[str, Tuple[float, float]]:
    pieces: List[np.ndarray] = []
    for examples in examples_by_size.values():
        for ex in examples:
            xyz = np.asarray(ex['query_xyz'], dtype=np.float32).reshape((-1, 3))
            valid = np.asarray(ex['query_valid'], dtype=np.bool_).reshape((-1,))
            if np.any(valid):
                pieces.append(xyz[valid])
            pieces.append(np.asarray(ex['current_xyz'], dtype=np.float32).reshape(1, 3))
            pieces.append(np.asarray(ex['target_abs'], dtype=np.float32)[:, :3])
    for traj in support_traj.values():
        arr = np.asarray(traj, dtype=np.float32)
        if arr.size:
            pieces.append(arr[:, :3])
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


def _write_html(
    *,
    out_path: Path,
    title: str,
    batch_sizes: Sequence[int],
    examples_by_size: Dict[int, Sequence[Dict[str, Any]]],
    support_traj: Dict[int, np.ndarray],
    plot_cfg: ConfigDict,
    seed: int,
) -> None:
    cols = len(batch_sizes)
    fig = make_subplots(
        rows=1,
        cols=cols,
        specs=[[{'type': 'scene'} for _ in batch_sizes]],
        subplot_titles=[f'inner batch Q={int(q)}' for q in batch_sizes],
        horizontal_spacing=0.02,
    )
    ranges = _axis_ranges(examples_by_size, support_traj)
    support_palette = ['#0f766e', '#7c3aed', '#dc2626', '#2563eb', '#ca8a04', '#0891b2']
    heldout_palette = ['#ef4444', '#3b82f6', '#22c55e', '#f97316', '#8b5cf6', '#ec4899']

    for col, q in enumerate(batch_sizes, start=1):
        examples = list(examples_by_size[int(q)])
        point_xyz: List[np.ndarray] = []
        point_colors: List[str] = []
        point_text: List[str] = []
        current_xyz: List[np.ndarray] = []
        current_colors: List[str] = []
        current_text: List[str] = []
        point_rng = np.random.default_rng(int(seed) + int(q) * 97)
        for ex in examples:
            heldout_color = heldout_palette[int(ex['heldout_episode']) % len(heldout_palette)]
            xyz, rgb, mask_id = _subsample_points(
                ex['query_xyz'],
                ex['query_valid'],
                ex['query_rgb'],
                ex['query_mask_id'],
                max_points=int(getattr(plot_cfg, 'max_points_per_sample', 0)),
                rng=point_rng,
            )
            if xyz.size:
                point_xyz.append(xyz)
                point_colors.extend(_rgb_strings(rgb, xyz.shape[0], fallback=heldout_color))
                mask_flat = np.asarray(mask_id).reshape((-1,)) if mask_id is not None else None
                for pidx in range(xyz.shape[0]):
                    mask_txt = f'<br>mask_id={int(mask_flat[pidx])}' if mask_flat is not None else ''
                    point_text.append(
                        f'q={int(ex["query_index"])}<br>'
                        f'heldout_ep={int(ex["heldout_episode"])}<br>'
                        f't0={int(ex["t0"])}'
                        f'{mask_txt}'
                    )
            current_xyz.append(np.asarray(ex['current_xyz'], dtype=np.float32).reshape(1, 3))
            current_colors.append(heldout_color)
            current_text.append(
                f'q={int(ex["query_index"])}<br>'
                f'heldout_ep={int(ex["heldout_episode"])}<br>'
                f'context={list(map(int, ex["context_episodes"]))}<br>'
                f't0={int(ex["t0"])}'
            )

        if point_xyz:
            pts = np.concatenate(point_xyz, axis=0)
            fig.add_trace(
                go.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    mode='markers',
                    marker={
                        'size': float(plot_cfg.marker_size),
                        'color': point_colors,
                        'opacity': float(plot_cfg.point_opacity),
                    },
                    text=point_text,
                    hovertemplate='%{text}<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>',
                    name=f'Q={int(q)} query point clouds',
                    showlegend=col == 1,
                ),
                row=1,
                col=col,
            )

        if current_xyz:
            cur = np.concatenate(current_xyz, axis=0)
            fig.add_trace(
                go.Scatter3d(
                    x=cur[:, 0],
                    y=cur[:, 1],
                    z=cur[:, 2],
                    mode='markers',
                    marker={'size': float(plot_cfg.current_marker_size), 'color': current_colors, 'symbol': 'cross'},
                    text=current_text,
                    hovertemplate='%{text}<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>',
                    name=f'Q={int(q)} current EE',
                    showlegend=col == 1,
                ),
                row=1,
                col=col,
            )

        for ex in examples:
            chunk = np.asarray(ex['target_abs'], dtype=np.float32)
            color = heldout_palette[int(ex['heldout_episode']) % len(heldout_palette)]
            text = [
                f'q={int(ex["query_index"])}<br>'
                f'heldout_ep={int(ex["heldout_episode"])}<br>'
                f'context={list(map(int, ex["context_episodes"]))}<br>'
                f't0={int(ex["t0"])}<br>'
                f'h={h}'
                for h in range(chunk.shape[0])
            ]
            fig.add_trace(
                go.Scatter3d(
                    x=chunk[:, 0],
                    y=chunk[:, 1],
                    z=chunk[:, 2],
                    mode='lines+markers',
                    line={'color': color, 'width': float(plot_cfg.action_line_width)},
                    marker={'size': float(plot_cfg.action_marker_size), 'color': color},
                    opacity=0.78,
                    text=text,
                    hovertemplate='%{text}<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>',
                    name=f'target chunk heldout ep {int(ex["heldout_episode"])}',
                    showlegend=False,
                ),
                row=1,
                col=col,
            )

        for sidx, (eid, traj) in enumerate(sorted(support_traj.items())):
            arr = np.asarray(traj, dtype=np.float32)
            if arr.size == 0:
                continue
            color = support_palette[sidx % len(support_palette)]
            fig.add_trace(
                go.Scatter3d(
                    x=arr[:, 0],
                    y=arr[:, 1],
                    z=arr[:, 2],
                    mode='lines',
                    line={'color': color, 'width': float(plot_cfg.support_line_width)},
                    opacity=float(plot_cfg.support_opacity),
                    name=f'support ep {int(eid)} full action xyz',
                    hovertemplate=f'support_ep={int(eid)}<br>x=%{{x:.4f}}<br>y=%{{y:.4f}}<br>z=%{{z:.4f}}<extra></extra>',
                    showlegend=col == 1,
                ),
                row=1,
                col=col,
            )

        scene_name = 'scene' if col == 1 else f'scene{col}'
        fig.layout[scene_name].update(
            xaxis={'range': ranges['x'], 'title': 'x'},
            yaxis={'range': ranges['y'], 'title': 'y'},
            zaxis={'range': ranges['z'], 'title': 'z'},
            aspectmode='cube',
        )

    fig.update_layout(
        title=title,
        margin={'l': 0, 'r': 0, 't': 80, 'b': 0},
        legend={'orientation': 'h'},
        height=760,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs='cdn')


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
    cache_root = _support_cache_root(cfg, ckpt)
    task_name = str(cfg.task.name)
    variation = int(cfg.task.variation)
    key = _choose_variation_key(cache_root, task_name, variation, rng)
    store = RLBenchCacheStore(
        [key],
        keep_open=bool(getattr(getattr(cfg, 'conditioning', ConfigDict()), 'keep_open', True)),
        preload_to_memory=False,
    )
    try:
        num_points, state_dim, action_dim = store.infer_dims()
        policy_cfg = policy_config_from(ckpt_cfg.model, H=data_cfg.H, data_cfg=data_cfg)
        use_rgb = bool(policy_cfg.encoder.use_rgb)
        load_mask_id = _load_mask_id_for_cached_batches(policy_cfg, data_cfg)
        sampler = ICILSampler(store, data_cfg, seed=seed + 17)
        available_episodes = tuple(int(x) for x in store.list_episode_ids(0))
        explicit_support = _as_int_tuple(getattr(getattr(cfg, 'support', ConfigDict()), 'episodes', ()))
        support_ids = _select_support_ids(available_episodes, int(data_cfg.K), explicit_support, rng)
        batch_sizes = tuple(sorted(set(_as_int_tuple(getattr(cfg.inner_batch, 'batch_sizes', '4,64')))))
        if not batch_sizes or any(int(q) <= 0 for q in batch_sizes):
            raise ValueError('inner_batch.batch_sizes must contain positive integers.')
        max_q = max(int(q) for q in batch_sizes)
        step_index = int(getattr(cfg.inner_batch, 'step_index', 0))

        examples_max = _build_inner_examples(
            sampler=sampler,
            support_ids=support_ids,
            num_inner_queries=max_q,
            step_index=step_index,
            rng=np.random.default_rng(seed + 101),
            load_rgb=use_rgb,
            load_mask_id=load_mask_id,
            action_representation=str(data_cfg.action_representation),
        )
        examples_by_size = {int(q): examples_max[: int(q)] for q in batch_sizes}
        support_traj = _support_trajectories(store, support_ids=support_ids)

        run_dir = Path(str(cfg.output.root_dir)).expanduser().resolve() / f'{task_name}_var{int(key.variation)}_{time.strftime("%Y%m%d-%H%M%S")}'
        run_dir.mkdir(parents=True, exist_ok=True)
        q_label = '_'.join(f'q{int(q)}' for q in batch_sizes)
        html_path = run_dir / f'inner_batch_{q_label}_step{step_index}.html'
        title = (
            f'{task_name} var {int(key.variation)} | support={list(map(int, support_ids))} | '
            f'inner step batch index={step_index} | Q={list(map(int, batch_sizes))}'
        )
        _write_html(
            out_path=html_path,
            title=title,
            batch_sizes=batch_sizes,
            examples_by_size=examples_by_size,
            support_traj=support_traj,
            plot_cfg=cfg.plot,
            seed=seed,
        )

        summary = {
            'checkpoint_path': str(checkpoint_path),
            'checkpoint_step': int(ckpt.get('step', -1)),
            'cache_root': str(cache_root),
            'task': task_name,
            'variation': int(key.variation),
            'support_episodes': [int(x) for x in support_ids],
            'batch_sizes': [int(q) for q in batch_sizes],
            'step_index': int(step_index),
            'html_path': str(html_path),
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
                'query_window_mode': str(data_cfg.query_window_mode),
                'support_spacetime_points': int(data_cfg.support_spacetime_points),
                'support_spacetime_sampling': str(data_cfg.support_spacetime_sampling),
            },
            'examples': {
                str(int(q)): [
                    {
                        'query_index': int(ex['query_index']),
                        'heldout_episode': int(ex['heldout_episode']),
                        'context_episodes': [int(x) for x in ex['context_episodes']],
                        't0': int(ex['t0']),
                        'current_xyz': [float(x) for x in np.asarray(ex['current_xyz']).tolist()],
                    }
                    for ex in examples_by_size[int(q)]
                ]
                for q in batch_sizes
            },
        }
        summary_path = run_dir / 'summary.json'
        with (run_dir / 'resolved_config.json').open('w', encoding='utf-8') as file:
            json.dump(cfg.to_dict(), file, indent=2)
        with summary_path.open('w', encoding='utf-8') as file:
            json.dump(summary, file, indent=2)
        print(f'Wrote {html_path}', flush=True)
        print(f'Wrote summary: {summary_path}', flush=True)
    finally:
        store.close()


def main(argv: Sequence[str]) -> None:
    del argv
    run(_CONFIG.value)


if __name__ == '__main__':
    app.run(main)
