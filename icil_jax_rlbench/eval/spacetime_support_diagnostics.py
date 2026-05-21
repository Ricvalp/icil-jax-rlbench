from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from icil_jax_rlbench.data.h5_cache import RLBenchCacheStore, build_variation_keys
from icil_jax_rlbench.data.sampler import ICILDataConfig, ICILSampler
from icil_jax_rlbench.eval.pointcloud_diagnostics import _axis_ranges, _mask_stats, _parse_int_list, _rgb_strings


def _parse_optional_int_list(value: str) -> Tuple[int, ...]:
    value = str(value or '').strip()
    if not value:
        return ()
    return _parse_int_list(value)


def _time_strings(time_values: np.ndarray) -> List[str]:
    t = np.asarray(time_values, dtype=np.float32).reshape(-1)
    return [f't={float(x):.4f}' for x in t.tolist()]


def _hover_text(mask_id: Optional[np.ndarray], time_values: np.ndarray) -> List[str]:
    times = np.asarray(time_values, dtype=np.float32).reshape(-1)
    if mask_id is None:
        return [f't={float(t):.4f}' for t in times.tolist()]
    masks = np.asarray(mask_id).reshape(-1)
    return [f't={float(t):.4f}<br>mask_id={int(m)}' for t, m in zip(times.tolist(), masks.tolist())]


def _spacetime_geometry(
    xyz: np.ndarray,
    time_values: np.ndarray,
    valid: np.ndarray,
    mask_id: Optional[np.ndarray],
    *,
    num_supernodes: int,
    temperature_xyz: float,
    temperature_t: float,
    center_sampling: str,
    edge_top_k: int,
    max_edge_supernodes: int,
    skip_self_edges: bool,
    edge_min_length: float,
    edge_candidate_multiplier: int,
) -> Dict[str, np.ndarray]:
    xyz = np.asarray(xyz, dtype=np.float32).reshape((-1, 3))
    time_values = np.asarray(time_values, dtype=np.float32).reshape(-1)
    valid = np.asarray(valid, dtype=np.bool_).reshape(-1)
    P = int(xyz.shape[0])
    M = min(int(num_supernodes), P)
    if P == 0 or M <= 0:
        return {
            'center_idx': np.zeros((0,), dtype=np.int64),
            'centers_xyz': np.zeros((0, 3), dtype=np.float32),
            'centers_time': np.zeros((0,), dtype=np.float32),
            'edge_start': np.zeros((0, 3), dtype=np.float32),
            'edge_end': np.zeros((0, 3), dtype=np.float32),
            'edge_start_time': np.zeros((0,), dtype=np.float32),
            'edge_end_time': np.zeros((0,), dtype=np.float32),
            'edge_cross_time': np.zeros((0,), dtype=np.bool_),
            'edge_weights': np.zeros((0,), dtype=np.float32),
            'edge_metric_distances': np.zeros((0,), dtype=np.float32),
            'edge_xyz_distances': np.zeros((0,), dtype=np.float32),
            'weights_shape': np.asarray([0, 0], dtype=np.int32),
        }

    center_idx = _center_indices(
        valid=valid,
        mask_id=mask_id,
        num_centers=M,
        center_sampling=str(center_sampling),
    )
    centers_xyz = xyz[center_idx].astype(np.float32)
    centers_time = time_values[center_idx].astype(np.float32)
    xyz_dist2 = np.sum((centers_xyz[:, None, :] - xyz[None, :, :]) ** 2, axis=-1)
    time_dist2 = (centers_time[:, None] - time_values[None, :]) ** 2
    metric = (
        xyz_dist2 / max(float(temperature_xyz), 1e-6)
        + time_dist2 / max(float(temperature_t), 1e-6)
    )
    logits = -metric
    logits = np.where(valid[None, :], logits, -1e9)
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    weights = np.exp(logits)
    weights = weights / np.maximum(np.sum(weights, axis=-1, keepdims=True), 1e-12)

    draw_m = min(M, int(max_edge_supernodes))
    candidates = max(
        int(edge_top_k) + (1 if bool(skip_self_edges) else 0),
        int(edge_top_k) * max(1, int(edge_candidate_multiplier)),
    )
    k = min(max(1, candidates), P)
    top_idx = np.argpartition(metric[:draw_m], kth=np.arange(k), axis=1)[:, :k]

    edge_start: List[np.ndarray] = []
    edge_end: List[np.ndarray] = []
    edge_start_time: List[float] = []
    edge_end_time: List[float] = []
    edge_cross_time: List[bool] = []
    edge_weights: List[float] = []
    edge_metric_distances: List[float] = []
    edge_xyz_distances: List[float] = []
    for sidx in range(draw_m):
        order = top_idx[sidx][np.argsort(metric[sidx, top_idx[sidx]])]
        drawn = 0
        for pidx in order.tolist():
            if not bool(valid[int(pidx)]):
                continue
            if bool(skip_self_edges) and int(pidx) == int(center_idx[sidx]):
                continue
            xyz_distance = float(np.linalg.norm(xyz[int(pidx)] - centers_xyz[sidx]))
            if xyz_distance < float(edge_min_length):
                continue
            edge_start.append(centers_xyz[sidx])
            edge_end.append(xyz[int(pidx)])
            start_t = float(centers_time[sidx])
            end_t = float(time_values[int(pidx)])
            edge_start_time.append(start_t)
            edge_end_time.append(end_t)
            edge_cross_time.append(abs(start_t - end_t) > 1e-7)
            edge_weights.append(float(weights[sidx, int(pidx)]))
            edge_metric_distances.append(float(np.sqrt(max(0.0, metric[sidx, int(pidx)]))))
            edge_xyz_distances.append(xyz_distance)
            drawn += 1
            if drawn >= int(edge_top_k):
                break

    return {
        'center_idx': center_idx,
        'centers_xyz': centers_xyz,
        'centers_time': centers_time,
        'edge_start': np.stack(edge_start, axis=0).astype(np.float32) if edge_start else np.zeros((0, 3), dtype=np.float32),
        'edge_end': np.stack(edge_end, axis=0).astype(np.float32) if edge_end else np.zeros((0, 3), dtype=np.float32),
        'edge_start_time': np.asarray(edge_start_time, dtype=np.float32),
        'edge_end_time': np.asarray(edge_end_time, dtype=np.float32),
        'edge_cross_time': np.asarray(edge_cross_time, dtype=np.bool_),
        'edge_weights': np.asarray(edge_weights, dtype=np.float32),
        'edge_metric_distances': np.asarray(edge_metric_distances, dtype=np.float32),
        'edge_xyz_distances': np.asarray(edge_xyz_distances, dtype=np.float32),
        'weights_shape': np.asarray(weights.shape, dtype=np.int32),
    }


def _center_indices(
    *,
    valid: np.ndarray,
    mask_id: Optional[np.ndarray],
    num_centers: int,
    center_sampling: str,
) -> np.ndarray:
    valid = np.asarray(valid, dtype=np.bool_).reshape(-1)
    N = int(valid.shape[0])
    M = int(num_centers)
    if str(center_sampling) != 'mask_balanced' or mask_id is None:
        return np.linspace(0, max(N - 1, 0), M).round().astype(np.int64)
    masks = np.asarray(mask_id).reshape(-1).astype(np.int64)
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size == 0:
        return np.linspace(0, max(N - 1, 0), M).round().astype(np.int64)
    order = valid_idx[np.argsort(masks[valid_idx], kind='stable')]
    sorted_masks = masks[order]
    start = np.concatenate([[True], sorted_masks[1:] != sorted_masks[:-1]])
    group_id = np.cumsum(start.astype(np.int64)) - 1
    group_start_pos = np.maximum.accumulate(np.where(start, np.arange(order.size), 0))
    rank = np.arange(order.size) - group_start_pos
    counts = np.bincount(group_id, minlength=max(1, int(group_id.max()) + 1))
    rank_fraction = rank.astype(np.float32) / np.maximum(counts[group_id].astype(np.float32), 1.0)
    key = rank_fraction + 1e-3 * group_id.astype(np.float32) / max(1.0, float(order.size))
    balanced = order[np.argsort(key, kind='stable')]
    return balanced[np.arange(M) % max(1, balanced.size)].astype(np.int64)


def _add_edges(fig: Any, geom: Dict[str, np.ndarray], *, row: int, col: int, width: float, opacity: float) -> None:
    import plotly.graph_objects as go

    edge_start = np.asarray(geom['edge_start'], dtype=np.float32)
    edge_end = np.asarray(geom['edge_end'], dtype=np.float32)
    if edge_start.shape[0] == 0:
        return
    cross_time = np.asarray(geom.get('edge_cross_time', np.zeros((edge_start.shape[0],), dtype=np.bool_))).reshape(-1)

    def add_subset(selector: np.ndarray, *, color: str, name: str) -> None:
        edge_x: List[Optional[float]] = []
        edge_y: List[Optional[float]] = []
        edge_z: List[Optional[float]] = []
        for start, end in zip(edge_start[selector], edge_end[selector]):
            edge_x.extend([float(start[0]), float(end[0]), None])
            edge_y.extend([float(start[1]), float(end[1]), None])
            edge_z.extend([float(start[2]), float(end[2]), None])
        if not edge_x:
            return
        fig.add_trace(
            go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode='lines',
                line={'color': color, 'width': float(width)},
                opacity=float(opacity),
                name=name,
                hoverinfo='skip',
                connectgaps=False,
            ),
            row=row,
            col=col,
        )

    add_subset(~cross_time, color='rgb(80,110,255)', name='same-frame pooling edges')
    add_subset(cross_time, color='rgb(255,80,0)', name='cross-frame pooling edges')


def _add_supernodes(fig: Any, geom: Dict[str, np.ndarray], *, row: int, col: int, marker_size: float) -> None:
    import plotly.graph_objects as go

    centers = np.asarray(geom['centers_xyz'], dtype=np.float32)
    center_time = np.asarray(geom['centers_time'], dtype=np.float32)
    if centers.shape[0] == 0:
        return
    fig.add_trace(
        go.Scatter3d(
            x=centers[:, 0],
            y=centers[:, 1],
            z=centers[:, 2],
            mode='markers',
            marker={'size': float(marker_size) * 3.0, 'color': 'black', 'symbol': 'diamond', 'opacity': 0.95},
            text=_time_strings(center_time),
            hovertemplate='%{text}<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>',
            name='spacetime supernode centers',
        ),
        row=row,
        col=col,
    )


def _write_html(
    *,
    frame: Dict[str, np.ndarray],
    geom: Dict[str, np.ndarray],
    out_path: Path,
    title: str,
    marker_size: float,
    edge_line_width: float,
    edge_opacity: float,
) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    xyz = np.asarray(frame['xyz'], dtype=np.float32).reshape((-1, 3))
    valid = np.asarray(frame['valid'], dtype=np.bool_).reshape(-1)
    time_values = np.asarray(frame['time'], dtype=np.float32).reshape(-1)
    rgb = frame.get('rgb')
    mask_id = frame.get('mask_id')
    xyz_v = xyz[valid]
    time_v = time_values[valid]
    rgb_v = None if rgb is None else np.asarray(rgb).reshape((-1, 3))[valid]
    mask_v = None if mask_id is None else np.asarray(mask_id).reshape(-1)[valid]
    hover = _hover_text(mask_v, time_v)

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=('sampled support union colored by time', 'same points colored by RGB'),
    )
    fig.add_trace(
        go.Scatter3d(
            x=xyz_v[:, 0],
            y=xyz_v[:, 1],
            z=xyz_v[:, 2],
            mode='markers',
            marker={
                'size': float(marker_size),
                'color': time_v,
                'colorscale': 'Viridis',
                'colorbar': {'title': 't'},
                'opacity': 0.80,
            },
            text=hover,
            hovertemplate='%{text}<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>',
            name='points by time',
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    _add_supernodes(fig, geom, row=1, col=1, marker_size=marker_size)
    _add_edges(fig, geom, row=1, col=1, width=edge_line_width, opacity=edge_opacity)

    rgb_colors = _rgb_strings(rgb_v, xyz_v.shape[0])
    fig.add_trace(
        go.Scatter3d(
            x=xyz_v[:, 0],
            y=xyz_v[:, 1],
            z=xyz_v[:, 2],
            mode='markers',
            marker={'size': float(marker_size), 'color': rgb_colors, 'opacity': 0.80},
            text=hover,
            hovertemplate='%{text}<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>',
            name='points by RGB',
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    _add_supernodes(fig, geom, row=1, col=2, marker_size=marker_size)

    ranges = _axis_ranges([{'xyz': xyz, 'valid': valid}])
    scene_update = {
        'xaxis': {'range': ranges['x'], 'title': 'x'},
        'yaxis': {'range': ranges['y'], 'title': 'y'},
        'zaxis': {'range': ranges['z'], 'title': 'z'},
        'aspectmode': 'cube',
    }
    fig.update_layout(
        title=title,
        margin={'l': 0, 'r': 0, 't': 58, 'b': 0},
        scene=scene_update,
        scene2=scene_update,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs='cdn')


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Visualize the sampled spacetime support pointcloud consumed by the encoder.')
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--variation', type=int, default=0)
    parser.add_argument('--support-episodes', default='', help='Comma-separated support episode ids. Defaults to the first K episodes.')
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--L', type=int, default=30)
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--T-obs', dest='T_obs', type=int, default=2)
    parser.add_argument('--H', type=int, default=16)
    parser.add_argument('--traj-len', type=int, default=64)
    parser.add_argument('--support-spacetime-points', type=int, default=8192)
    parser.add_argument('--support-spacetime-sampling', default='mask_balanced', choices=('mask_balanced', 'uniform'))
    parser.add_argument('--spacetime-supernodes', type=int, default=256)
    parser.add_argument('--spacetime-temperature-xyz', type=float, default=0.005)
    parser.add_argument('--spacetime-temperature-t', type=float, default=0.04)
    parser.add_argument('--supernode-center-sampling', default='mask_balanced', choices=('linspace', 'mask_balanced'))
    parser.add_argument('--edge-top-k', type=int, default=5)
    parser.add_argument('--max-edge-supernodes', type=int, default=128)
    parser.add_argument('--skip-self-edges', dest='skip_self_edges', action='store_true')
    parser.add_argument('--show-self-edges', dest='skip_self_edges', action='store_false')
    parser.set_defaults(skip_self_edges=True)
    parser.add_argument('--edge-min-length', type=float, default=0.005)
    parser.add_argument('--edge-candidate-multiplier', type=int, default=32)
    parser.add_argument('--edge-line-width', type=float, default=3.0)
    parser.add_argument('--edge-opacity', type=float, default=0.8)
    parser.add_argument('--marker-size', type=float, default=1.8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-dir', default='eval_outputs/spacetime_support_diagnostics')
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    cache_root = Path(args.cache_root).expanduser().resolve()
    keys = [key for key in build_variation_keys(cache_root, str(args.task)) if int(key.variation) == int(args.variation)]
    if not keys:
        raise RuntimeError(f'No cached {args.task}/variation{args.variation}.h5 under {cache_root}.')
    store = RLBenchCacheStore(keys[:1], keep_open=True, preload_to_memory=False)
    try:
        available = tuple(int(x) for x in store.list_episode_ids(0))
        explicit = _parse_optional_int_list(args.support_episodes)
        if explicit:
            support_ids = tuple(int(x) for x in explicit[: int(args.K)])
        else:
            support_ids = available[: int(args.K)]
        missing = [eid for eid in support_ids if int(eid) not in available]
        if missing:
            raise RuntimeError(f'Support episodes {missing} are unavailable. Available: {available}')
        if len(support_ids) < int(args.K):
            raise RuntimeError(f'Need K={args.K} support episodes, found {len(support_ids)} in {available}.')

        data_cfg = ICILDataConfig(
            K=int(args.K),
            L=int(args.L),
            T_obs=int(args.T_obs),
            H=int(args.H),
            stride=int(args.stride),
            traj_len=int(args.traj_len),
            support_spacetime_points=int(args.support_spacetime_points),
            support_spacetime_sampling=str(args.support_spacetime_sampling),
        )
        sampler = ICILSampler(store, data_cfg, seed=int(args.seed))
        support = sampler.build_support_conditioning(
            vidx=0,
            support_ids=support_ids,
            load_rgb=True,
            load_mask_id=True,
        )

        out_root = Path(args.out_dir).expanduser().resolve() / f'{args.task}_var{int(args.variation)}'
        out_root.mkdir(parents=True, exist_ok=True)
        summary: Dict[str, Any] = {
            'cache_root': str(cache_root),
            'task': str(args.task),
            'variation': int(args.variation),
            'support_episode_ids': [int(x) for x in support_ids],
            'K': int(args.K),
            'L': int(args.L),
            'support_spacetime_points': int(args.support_spacetime_points),
            'support_spacetime_sampling': str(args.support_spacetime_sampling),
            'spacetime_supernodes': int(args.spacetime_supernodes),
            'spacetime_temperature_xyz': float(args.spacetime_temperature_xyz),
            'spacetime_temperature_t': float(args.spacetime_temperature_t),
            'supernode_center_sampling': str(args.supernode_center_sampling),
            'outputs': [],
            'note': 'Edges are top-weight points under the same XYZ+time soft-pooling metric used by SpacetimeSupportTokenizer.',
        }

        for kidx, episode_id in enumerate(support_ids):
            frame = {
                'xyz': support['cond_st_xyz'][kidx],
                'time': support['cond_st_time'][kidx],
                'valid': support['cond_st_valid'][kidx],
            }
            if 'cond_st_rgb' in support:
                frame['rgb'] = support['cond_st_rgb'][kidx]
            if 'cond_st_mask_id' in support:
                frame['mask_id'] = support['cond_st_mask_id'][kidx]
            geom = _spacetime_geometry(
                frame['xyz'],
                frame['time'],
                frame['valid'],
                frame.get('mask_id'),
                num_supernodes=int(args.spacetime_supernodes),
                temperature_xyz=float(args.spacetime_temperature_xyz),
                temperature_t=float(args.spacetime_temperature_t),
                center_sampling=str(args.supernode_center_sampling),
                edge_top_k=int(args.edge_top_k),
                max_edge_supernodes=int(args.max_edge_supernodes),
                skip_self_edges=bool(args.skip_self_edges),
                edge_min_length=float(args.edge_min_length),
                edge_candidate_multiplier=int(args.edge_candidate_multiplier),
            )
            out_path = out_root / f'support_ep{int(episode_id):04d}.html'
            title = (
                f'{args.task} var {int(args.variation)} support episode {int(episode_id)} | '
                f'L={int(args.L)} P={int(args.support_spacetime_points)} M={int(args.spacetime_supernodes)}'
            )
            _write_html(
                frame=frame,
                geom=geom,
                out_path=out_path,
                title=title,
                marker_size=float(args.marker_size),
                edge_line_width=float(args.edge_line_width),
                edge_opacity=float(args.edge_opacity),
            )
            summary['outputs'].append(
                {
                    'path': str(out_path),
                    'support_episode_id': int(episode_id),
                    'valid_points': int(np.sum(np.asarray(frame['valid'], dtype=np.bool_))),
                    'time_min': float(np.min(frame['time'])),
                    'time_max': float(np.max(frame['time'])),
                    'mask_stats': _mask_stats(frame.get('mask_id')),
                    'supernodes': int(geom['centers_xyz'].shape[0]),
                    'drawn_edges': int(geom['edge_start'].shape[0]),
                    'same_frame_edges': int(np.sum(~np.asarray(geom['edge_cross_time'], dtype=np.bool_))),
                    'cross_frame_edges': int(np.sum(np.asarray(geom['edge_cross_time'], dtype=np.bool_))),
                    'weights_shape': [int(x) for x in np.asarray(geom['weights_shape']).tolist()],
                }
            )

        summary_path = out_root / 'summary.json'
        with summary_path.open('w', encoding='utf-8') as file:
            json.dump(summary, file, indent=2)
        print(f'Wrote {len(summary["outputs"])} spacetime support HTML files under {out_root}')
        print(f'Wrote summary: {summary_path}')
    finally:
        store.close()


if __name__ == '__main__':
    main()
