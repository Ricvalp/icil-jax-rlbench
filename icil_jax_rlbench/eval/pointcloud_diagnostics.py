from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ml_collections import ConfigDict
import numpy as np

from icil_jax_rlbench.data.h5_cache import RLBenchCacheStore, build_variation_keys
from icil_jax_rlbench.eval.online_common import (
    LiveObservationProcessor,
    build_rlbench_env,
)
from icil_jax_rlbench.models.config import encoder_config_from
from icil_jax_rlbench.models.encoders import EncoderConfig
from icil_jax_rlbench.train.checkpoints import load_checkpoint


def _parse_int_list(value: str) -> Tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(value).split(',') if x.strip())


def _parse_str_list(value: str) -> Tuple[str, ...]:
    return tuple(x.strip().lower() for x in str(value).split(',') if x.strip())


def _checkpoint_encoder_config(path: str) -> EncoderConfig:
    if not path:
        return EncoderConfig()
    ckpt = load_checkpoint(Path(path).expanduser())
    cfg = ConfigDict(ckpt.get('config', {}) or {})
    if not hasattr(cfg, 'model') or not hasattr(cfg.model, 'encoder'):
        return EncoderConfig()
    return encoder_config_from(cfg.model.encoder)


def _rgb_strings(rgb: Optional[np.ndarray], count: int) -> List[str]:
    if rgb is None:
        return ['rgb(60,120,220)'] * int(count)
    arr = np.asarray(rgb)
    if arr.size == 0:
        return ['rgb(60,120,220)'] * int(count)
    if arr.dtype != np.uint8:
        arr = np.clip(arr.astype(np.float32), 0.0, 1.0)
        arr = np.rint(255.0 * arr).astype(np.uint8)
    arr = arr.reshape((-1, 3))
    return [f'rgb({int(r)},{int(g)},{int(b)})' for r, g, b in arr[:count]]


def _mask_stats(mask_id: Optional[np.ndarray], handle_to_name: Optional[Dict[int, str]] = None, limit: int = 12) -> List[Dict[str, Any]]:
    if mask_id is None:
        return []
    values, counts = np.unique(np.asarray(mask_id).reshape(-1), return_counts=True)
    order = np.argsort(counts)[::-1]
    out = []
    names = handle_to_name or {}
    for idx in order[: int(limit)]:
        mid = int(values[idx])
        out.append({'mask_id': mid, 'count': int(counts[idx]), 'name': names.get(mid, '')})
    return out


def _hover_text(mask_id: Optional[np.ndarray], handle_to_name: Optional[Dict[int, str]]) -> Optional[List[str]]:
    if mask_id is None:
        return None
    names = handle_to_name or {}
    out = []
    for value in np.asarray(mask_id).reshape(-1):
        mid = int(value)
        name = names.get(mid, '')
        out.append(f'mask_id={mid}<br>{name}' if name else f'mask_id={mid}')
    return out


def _supernode_geometry(
    xyz: np.ndarray,
    valid: Optional[np.ndarray],
    *,
    num_supernodes: int,
    temperature: float,
    edge_top_k: int,
    max_edge_supernodes: int,
    skip_self_edges: bool,
    edge_min_length: float,
    edge_candidate_multiplier: int,
) -> Dict[str, np.ndarray]:
    xyz = np.asarray(xyz, dtype=np.float32).reshape((-1, 3))
    if valid is not None:
        valid = np.asarray(valid).reshape((-1,)).astype(np.bool_)
        xyz_valid = xyz[valid]
    else:
        xyz_valid = xyz
    if xyz_valid.shape[0] == 0 or int(num_supernodes) <= 0:
        return {
            'centers': np.zeros((0, 3), dtype=np.float32),
            'weights': np.zeros((0, 0), dtype=np.float32),
            'edge_start': np.zeros((0, 3), dtype=np.float32),
            'edge_end': np.zeros((0, 3), dtype=np.float32),
            'edge_xyz': np.zeros((0, 3), dtype=np.float32),
            'edge_weights': np.zeros((0,), dtype=np.float32),
            'edge_distances': np.zeros((0,), dtype=np.float32),
        }

    n = int(xyz_valid.shape[0])
    m = min(int(num_supernodes), n)
    center_idx = np.linspace(0, max(n - 1, 0), m).round().astype(np.int64)
    centers = xyz_valid[center_idx].astype(np.float32)
    if int(edge_top_k) <= 0:
        return {
            'centers': centers,
            'weights': np.zeros((0, 0), dtype=np.float32),
            'edge_start': np.zeros((0, 3), dtype=np.float32),
            'edge_end': np.zeros((0, 3), dtype=np.float32),
            'edge_xyz': np.zeros((0, 3), dtype=np.float32),
            'edge_weights': np.zeros((0,), dtype=np.float32),
            'edge_distances': np.zeros((0,), dtype=np.float32),
        }

    dist2 = np.sum((centers[:, None, :] - xyz_valid[None, :, :]) ** 2, axis=-1)
    logits = -dist2 / max(float(temperature), 1e-6)
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    weights = np.exp(logits)
    weights = weights / np.maximum(np.sum(weights, axis=-1, keepdims=True), 1e-12)
    draw_m = min(m, int(max_edge_supernodes))
    draw_weights = weights[:draw_m]
    draw_dist2 = dist2[:draw_m]
    candidates = max(
        int(edge_top_k) + (1 if bool(skip_self_edges) else 0),
        int(edge_top_k) * max(1, int(edge_candidate_multiplier)),
    )
    k = min(candidates, n)
    top_idx = np.argpartition(draw_dist2, kth=np.arange(k), axis=1)[:, :k]

    segments: List[np.ndarray] = []
    edge_start: List[np.ndarray] = []
    edge_end: List[np.ndarray] = []
    edge_weights: List[float] = []
    edge_distances: List[float] = []
    for sidx in range(draw_m):
        order = top_idx[sidx][np.argsort(draw_dist2[sidx, top_idx[sidx]])]
        drawn = 0
        for pidx in order:
            if bool(skip_self_edges) and int(pidx) == int(center_idx[sidx]):
                continue
            length = float(np.linalg.norm(xyz_valid[int(pidx)] - centers[sidx]))
            if length < float(edge_min_length):
                continue
            edge_start.append(centers[sidx])
            edge_end.append(xyz_valid[int(pidx)])
            segments.append(centers[sidx])
            segments.append(xyz_valid[int(pidx)])
            segments.append(np.asarray([np.nan, np.nan, np.nan], dtype=np.float32))
            edge_weights.append(float(draw_weights[sidx, int(pidx)]))
            edge_distances.append(length)
            drawn += 1
            if drawn >= int(edge_top_k):
                break
    edge_xyz = np.stack(segments, axis=0).astype(np.float32) if segments else np.zeros((0, 3), dtype=np.float32)
    return {
        'centers': centers,
        'weights': weights.astype(np.float32),
        'edge_start': np.stack(edge_start, axis=0).astype(np.float32) if edge_start else np.zeros((0, 3), dtype=np.float32),
        'edge_end': np.stack(edge_end, axis=0).astype(np.float32) if edge_end else np.zeros((0, 3), dtype=np.float32),
        'edge_xyz': edge_xyz,
        'edge_weights': np.asarray(edge_weights, dtype=np.float32),
        'edge_distances': np.asarray(edge_distances, dtype=np.float32),
    }


def _mass_summary(mass: np.ndarray) -> Dict[str, Any]:
    mass = np.asarray(mass, dtype=np.float32).reshape(-1)
    if mass.size == 0:
        return {
            'mean_mass_per_supernode': 0.0,
            'max_mass': 0.0,
            'sum_mass_over_supernodes': 0.0,
            'supernodes_mass_gt_0.01': 0,
            'supernodes_mass_gt_0.05': 0,
            'supernodes_mass_gt_0.10': 0,
            'top_supernodes': [],
        }
    order = np.argsort(-mass)
    return {
        'mean_mass_per_supernode': float(np.mean(mass)),
        'max_mass': float(np.max(mass)),
        'sum_mass_over_supernodes': float(np.sum(mass)),
        'supernodes_mass_gt_0.01': int(np.sum(mass > 0.01)),
        'supernodes_mass_gt_0.05': int(np.sum(mass > 0.05)),
        'supernodes_mass_gt_0.10': int(np.sum(mass > 0.10)),
        'top_supernodes': [
            {'supernode': int(idx), 'mass': float(mass[int(idx)])}
            for idx in order[: min(8, int(order.size))]
        ],
    }


def _pooling_mass_report(
    frame: Dict[str, Any],
    geom: Dict[str, np.ndarray],
    *,
    handle_to_name: Optional[Dict[int, str]],
    focus_name_substrings: Sequence[str],
    focus_mask_ids: Sequence[int],
    top_k_masks: int = 12,
) -> Dict[str, Any]:
    weights = np.asarray(geom.get('weights', np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)
    mask_id = frame.get('mask_id')
    if weights.size == 0 or mask_id is None:
        return {
            'available': False,
            'reason': 'No supernode weights or mask ids available.',
        }
    valid = np.asarray(frame.get('valid', np.ones(np.asarray(mask_id).size, dtype=np.bool_))).reshape(-1).astype(np.bool_)
    ids = np.asarray(mask_id).reshape(-1).astype(np.int64)[valid]
    if ids.shape[0] != weights.shape[1]:
        return {
            'available': False,
            'reason': f'Mask/weight size mismatch: mask_points={ids.shape[0]} weights_points={weights.shape[1]}.',
        }
    names = handle_to_name or {}
    values, counts = np.unique(ids, return_counts=True)
    per_mask = []
    for mid, count in zip(values.tolist(), counts.tolist()):
        selector = ids == int(mid)
        mass = np.sum(weights[:, selector], axis=1)
        item = {
            'mask_id': int(mid),
            'name': names.get(int(mid), ''),
            'point_count': int(count),
            **_mass_summary(mass),
        }
        per_mask.append(item)
    per_mask.sort(key=lambda x: x['sum_mass_over_supernodes'], reverse=True)

    focus_ids = set(int(x) for x in focus_mask_ids)
    lowered = tuple(str(x).lower() for x in focus_name_substrings)
    if lowered:
        for mid, name in names.items():
            lname = str(name).lower()
            if any(token in lname for token in lowered):
                focus_ids.add(int(mid))
    focus_selector = np.zeros((ids.shape[0],), dtype=np.bool_)
    for mid in sorted(focus_ids):
        focus_selector |= ids == int(mid)
    focus_mass = np.sum(weights[:, focus_selector], axis=1) if np.any(focus_selector) else np.zeros((weights.shape[0],), dtype=np.float32)
    return {
        'available': True,
        'focus_name_substrings': tuple(focus_name_substrings),
        'focus_mask_ids': sorted(int(x) for x in focus_ids),
        'focus_point_count': int(np.sum(focus_selector)),
        'focus': _mass_summary(focus_mass),
        'top_masks_by_sum_mass': per_mask[: int(top_k_masks)],
        'top_masks_by_point_count': sorted(per_mask, key=lambda x: x['point_count'], reverse=True)[: int(top_k_masks)],
    }


def _axis_ranges(frames: Sequence[Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
    pts = []
    for frame in frames:
        xyz = np.asarray(frame['xyz'], dtype=np.float32).reshape((-1, 3))
        valid = np.asarray(frame.get('valid', np.ones((xyz.shape[0],), dtype=np.bool_))).reshape((-1,)).astype(np.bool_)
        if np.any(valid):
            pts.append(xyz[valid])
    if not pts:
        return {'x': (-1.0, 1.0), 'y': (-1.0, 1.0), 'z': (0.0, 1.0)}
    all_pts = np.concatenate(pts, axis=0)
    mins = np.nanmin(all_pts, axis=0)
    maxs = np.nanmax(all_pts, axis=0)
    center = 0.5 * (mins + maxs)
    span = float(np.nanmax(maxs - mins))
    span = max(span, 1e-3)
    half = 0.55 * span
    return {
        'x': (float(center[0] - half), float(center[0] + half)),
        'y': (float(center[1] - half), float(center[1] + half)),
        'z': (float(center[2] - half), float(center[2] + half)),
    }


def _add_frame_traces(
    fig: Any,
    *,
    frame: Dict[str, Any],
    title: str,
    row: int,
    col: int,
    geom: Dict[str, np.ndarray],
    marker_size: float,
    edge_line_width: float,
    edge_opacity: float,
    handle_to_name: Optional[Dict[int, str]] = None,
) -> None:
    import plotly.graph_objects as go

    xyz = np.asarray(frame['xyz'], dtype=np.float32).reshape((-1, 3))
    valid = np.asarray(frame.get('valid', np.ones((xyz.shape[0],), dtype=np.bool_))).reshape((-1,)).astype(np.bool_)
    rgb = frame.get('rgb')
    if rgb is not None:
        rgb = np.asarray(rgb).reshape((-1, 3))[valid]
    mask_id = frame.get('mask_id')
    if mask_id is not None:
        mask_id = np.asarray(mask_id).reshape((-1,))[valid]
    xyz = xyz[valid]
    colors = _rgb_strings(rgb, xyz.shape[0])
    hover = _hover_text(mask_id, handle_to_name)

    fig.add_trace(
        go.Scatter3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            mode='markers',
            marker={'size': float(marker_size), 'color': colors, 'opacity': 0.78},
            text=hover,
            hovertemplate='%{text}<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>' if hover else 'x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>',
            name=f'{title} points',
            showlegend=False,
        ),
        row=row,
        col=col,
    )

    centers = np.asarray(geom['centers'], dtype=np.float32)
    if centers.shape[0] > 0:
        fig.add_trace(
            go.Scatter3d(
                x=centers[:, 0],
                y=centers[:, 1],
                z=centers[:, 2],
                mode='markers',
                marker={'size': float(marker_size) * 3.0, 'color': 'black', 'symbol': 'diamond', 'opacity': 0.95},
                name=f'{title} supernodes',
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    edge_start = np.asarray(geom.get('edge_start', np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
    edge_end = np.asarray(geom.get('edge_end', np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
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
                line={'color': 'rgb(255,80,0)', 'width': float(edge_line_width)},
                opacity=float(edge_opacity),
                name=f'{title} top pooling edges',
                showlegend=False,
                hoverinfo='skip',
                connectgaps=False,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter3d(
                x=edge_end[:, 0],
                y=edge_end[:, 1],
                z=edge_end[:, 2],
                mode='markers',
                marker={'size': float(marker_size) * 2.0, 'color': 'rgb(255,80,0)', 'symbol': 'circle-open', 'opacity': 0.95},
                name=f'{title} pooling endpoints',
                showlegend=False,
                hoverinfo='skip',
            ),
            row=row,
            col=col,
        )


def _write_comparison_html(
    *,
    sim_frame: Dict[str, Any],
    cache_frame: Dict[str, Any],
    sim_geom: Dict[str, np.ndarray],
    cache_geom: Dict[str, np.ndarray],
    sim_handle_to_name: Dict[int, str],
    title: str,
    out_path: Path,
    marker_size: float,
    edge_line_width: float,
    edge_opacity: float,
    focus_name_substrings: Sequence[str],
    focus_mask_ids: Sequence[int],
) -> Dict[str, Any]:
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=('simulation/live processor', 'cached H5'),
    )
    _add_frame_traces(
        fig,
        frame=sim_frame,
        title='simulation',
        row=1,
        col=1,
        geom=sim_geom,
        marker_size=marker_size,
        edge_line_width=edge_line_width,
        edge_opacity=edge_opacity,
        handle_to_name=sim_handle_to_name,
    )
    _add_frame_traces(
        fig,
        frame=cache_frame,
        title='cache',
        row=1,
        col=2,
        geom=cache_geom,
        marker_size=marker_size,
        edge_line_width=edge_line_width,
        edge_opacity=edge_opacity,
        handle_to_name=None,
    )
    ranges = _axis_ranges([sim_frame, cache_frame])
    scene_update = {
        'xaxis': {'range': ranges['x'], 'title': 'x'},
        'yaxis': {'range': ranges['y'], 'title': 'y'},
        'zaxis': {'range': ranges['z'], 'title': 'z'},
        'aspectmode': 'cube',
    }
    fig.update_layout(
        title=title,
        margin={'l': 0, 'r': 0, 't': 54, 'b': 0},
        scene=scene_update,
        scene2=scene_update,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs='cdn')
    return {
        'path': str(out_path),
        'sim_points': int(np.sum(np.asarray(sim_frame['valid'], dtype=np.bool_))),
        'cache_points': int(np.sum(np.asarray(cache_frame['valid'], dtype=np.bool_))),
        'sim_mask_stats': _mask_stats(sim_frame.get('mask_id'), sim_handle_to_name),
        'cache_mask_stats': _mask_stats(cache_frame.get('mask_id')),
        'sim_supernodes': int(sim_geom['centers'].shape[0]),
        'cache_supernodes': int(cache_geom['centers'].shape[0]),
        'sim_edges': int(sim_geom['edge_xyz'].shape[0] // 3),
        'cache_edges': int(cache_geom['edge_xyz'].shape[0] // 3),
        'sim_pooling_mass': _pooling_mass_report(
            sim_frame,
            sim_geom,
            handle_to_name=sim_handle_to_name,
            focus_name_substrings=focus_name_substrings,
            focus_mask_ids=focus_mask_ids,
        ),
        'cache_pooling_mass': _pooling_mass_report(
            cache_frame,
            cache_geom,
            handle_to_name=None,
            focus_name_substrings=(),
            focus_mask_ids=focus_mask_ids,
        ),
        'supernode_note': 'Edges are top-weight input points under the supernode soft-pooling kernel, not learned self-attention edges.',
    }


def _load_cache_frame(
    store: RLBenchCacheStore,
    *,
    episode_id: int,
    frame_index: int,
) -> Dict[str, np.ndarray]:
    T = store.episode_length(0, int(episode_id))
    idx = int(np.clip(int(frame_index), 0, max(0, T - 1)))
    item = store.load_episode_slices(0, int(episode_id), np.asarray([idx]), load_rgb=True, load_mask_id=True)
    frame = {
        'xyz': item['xyz'][0],
        'valid': item['valid'][0],
        'state': item['state'][0],
        'mask_id': item['mask_id'][0] if 'mask_id' in item else None,
    }
    if 'rgb' in item:
        frame['rgb'] = item['rgb'][0]
    return frame


def _make_env_cfg(args: argparse.Namespace) -> ConfigDict:
    cfg = ConfigDict()
    cfg.sim = ConfigDict()
    cfg.sim.headless = bool(args.headless)
    cfg.sim.renderer = str(args.renderer)
    cfg.sim.image_size = (int(args.image_size), int(args.image_size))
    cfg.sim.arm_max_velocity = float(args.arm_max_velocity)
    cfg.sim.arm_max_acceleration = float(args.arm_max_acceleration)
    cfg.sim.collision_checking = False
    return cfg


def _str_to_bool(value: str) -> bool:
    value = str(value).strip().lower()
    if value in ('1', 'true', 't', 'yes', 'y', 'on'):
        return True
    if value in ('0', 'false', 'f', 'no', 'n', 'off'):
        return False
    raise argparse.ArgumentTypeError(f'Expected a boolean value, got {value!r}.')


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compare online RLBench pointclouds against cached H5 pointclouds.')
    parser.add_argument('--checkpoint', default='', help='Optional checkpoint used to infer supernode config.')
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--variation', type=int, default=0)
    parser.add_argument('--out-dir', default='eval_outputs/pointcloud_diagnostics')
    parser.add_argument('--num-points', type=int, default=1024)
    parser.add_argument('--cache-episodes', default='0', help='Comma-separated cached episode ids.')
    parser.add_argument('--cache-frames', default='0', help='Comma-separated cached frame indices.')
    parser.add_argument('--sim-resets', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--renderer', default='opengl', choices=('opengl', 'opengl3'))
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--headless', dest='headless', action='store_true')
    parser.add_argument('--no-headless', dest='headless', action='store_false')
    parser.add_argument(
        '--config.sim.headless',
        dest='config_sim_headless',
        type=_str_to_bool,
        default=None,
        help='Compatibility alias matching the online eval config flag.',
    )
    parser.set_defaults(headless=True)
    parser.add_argument('--arm-max-velocity', type=float, default=1.0)
    parser.add_argument('--arm-max-acceleration', type=float, default=4.0)
    parser.add_argument('--workspace-bounds', default='-1,1,-1,1,0,2.5', help='xmin,xmax,ymin,ymax,zmin,zmax or empty to disable.')
    parser.add_argument('--supernodes', type=int, default=-1, help='Override checkpoint/default supernode count.')
    parser.add_argument('--supernode-temperature', type=float, default=-1.0, help='Override checkpoint/default supernode temperature.')
    parser.add_argument('--edge-top-k', type=int, default=2)
    parser.add_argument('--max-edge-supernodes', type=int, default=32)
    parser.add_argument('--skip-self-edges', dest='skip_self_edges', action='store_true')
    parser.add_argument('--show-self-edges', dest='skip_self_edges', action='store_false')
    parser.set_defaults(skip_self_edges=True)
    parser.add_argument('--edge-line-width', type=float, default=6.0)
    parser.add_argument('--edge-opacity', type=float, default=0.85)
    parser.add_argument('--edge-min-length', type=float, default=0.01, help='Skip drawn pooling edges shorter than this many meters.')
    parser.add_argument('--edge-candidate-multiplier', type=int, default=32, help='Search this many times edge_top_k neighbors to find visible nondegenerate edges.')
    parser.add_argument('--marker-size', type=float, default=2.0)
    parser.add_argument('--focus-name-substrings', default='button', help='Comma-separated live object-name substrings to aggregate pooling mass for.')
    parser.add_argument('--focus-mask-ids', default='', help='Comma-separated mask ids to aggregate pooling mass for in both live and cache.')
    args = parser.parse_args(argv)
    if args.config_sim_headless is not None:
        args.headless = bool(args.config_sim_headless)
    return args


def _workspace_bounds(raw: str) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]]:
    if not str(raw).strip():
        return None
    values = [float(x.strip()) for x in str(raw).split(',') if x.strip()]
    if len(values) != 6:
        raise ValueError('--workspace-bounds must be xmin,xmax,ymin,ymax,zmin,zmax.')
    return ((values[0], values[1]), (values[2], values[3]), (values[4], values[5]))


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    rng = np.random.default_rng(int(args.seed))
    enc = _checkpoint_encoder_config(str(args.checkpoint))
    num_supernodes = int(args.supernodes) if int(args.supernodes) > 0 else int(enc.supernodes)
    temperature = float(args.supernode_temperature) if float(args.supernode_temperature) > 0.0 else float(enc.supernode_temperature)
    focus_name_substrings = _parse_str_list(str(args.focus_name_substrings))
    focus_mask_ids = _parse_int_list(str(args.focus_mask_ids))

    cache_root = Path(args.cache_root).expanduser().resolve()
    keys = [k for k in build_variation_keys(cache_root, str(args.task)) if int(k.variation) == int(args.variation)]
    if not keys:
        raise RuntimeError(f'No cached {args.task}/variation{args.variation}.h5 under {cache_root}.')
    store = RLBenchCacheStore(keys[:1], keep_open=True, preload_to_memory=False)
    cache_episodes = _parse_int_list(args.cache_episodes)
    cache_frames = _parse_int_list(args.cache_frames)
    if not cache_episodes:
        cache_episodes = tuple(int(x) for x in store.list_episode_ids(0)[:1])
    if not cache_frames:
        cache_frames = (0,)

    out_dir = Path(args.out_dir).expanduser().resolve() / f'{args.task}_var{int(args.variation)}'
    out_dir.mkdir(parents=True, exist_ok=True)
    env = None
    summary: Dict[str, Any] = {
        'task': str(args.task),
        'variation': int(args.variation),
        'cache_root': str(cache_root),
        'checkpoint': str(args.checkpoint),
        'num_points': int(args.num_points),
        'supernodes': int(num_supernodes),
        'supernode_temperature': float(temperature),
        'edge_top_k': int(args.edge_top_k),
        'max_edge_supernodes': int(args.max_edge_supernodes),
        'skip_self_edges': bool(args.skip_self_edges),
        'focus_name_substrings': focus_name_substrings,
        'focus_mask_ids': focus_mask_ids,
        'outputs': [],
        'note': 'Supernode edges are top-weight soft-pooling edges from input points to deterministic centers.',
    }
    try:
        cfg = _make_env_cfg(args)
        env, task_env = build_rlbench_env(cfg, str(args.task))
        task_env.set_variation(int(args.variation))
        processor = LiveObservationProcessor(
            task_env=task_env,
            num_points=int(args.num_points),
            use_rgb=True,
            use_mask_id=True,
            workspace_bounds=_workspace_bounds(str(args.workspace_bounds)),
            seed=int(args.seed),
        )
        for sim_idx in range(int(args.sim_resets)):
            task_env.set_variation(int(args.variation))
            _descriptions, obs = task_env.reset()
            sim_frame = processor.observation_to_frame(obs)
            sim_geom = _supernode_geometry(
                sim_frame['xyz'],
                sim_frame['valid'],
                num_supernodes=num_supernodes,
                temperature=temperature,
                edge_top_k=int(args.edge_top_k),
                max_edge_supernodes=int(args.max_edge_supernodes),
                skip_self_edges=bool(args.skip_self_edges),
                edge_min_length=float(args.edge_min_length),
                edge_candidate_multiplier=int(args.edge_candidate_multiplier),
            )
            for episode_id in cache_episodes:
                for frame_idx in cache_frames:
                    cache_frame = _load_cache_frame(store, episode_id=int(episode_id), frame_index=int(frame_idx))
                    cache_geom = _supernode_geometry(
                        cache_frame['xyz'],
                        cache_frame['valid'],
                        num_supernodes=num_supernodes,
                        temperature=temperature,
                        edge_top_k=int(args.edge_top_k),
                        max_edge_supernodes=int(args.max_edge_supernodes),
                        skip_self_edges=bool(args.skip_self_edges),
                        edge_min_length=float(args.edge_min_length),
                        edge_candidate_multiplier=int(args.edge_candidate_multiplier),
                    )
                    name = f'sim{sim_idx:02d}_cache_ep{int(episode_id):04d}_frame{int(frame_idx):04d}.html'
                    item = _write_comparison_html(
                        sim_frame=sim_frame,
                        cache_frame=cache_frame,
                        sim_geom=sim_geom,
                        cache_geom=cache_geom,
                        sim_handle_to_name=processor.handle_to_name,
                        title=f'{args.task} variation {int(args.variation)} | sim reset {sim_idx} vs cache ep {int(episode_id)} frame {int(frame_idx)}',
                        out_path=out_dir / name,
                        marker_size=float(args.marker_size),
                        edge_line_width=float(args.edge_line_width),
                        edge_opacity=float(args.edge_opacity),
                        focus_name_substrings=focus_name_substrings,
                        focus_mask_ids=focus_mask_ids,
                    )
                    item.update({'sim_reset': int(sim_idx), 'cache_episode': int(episode_id), 'cache_frame': int(frame_idx)})
                    summary['outputs'].append(item)
                    print(item['path'])
    finally:
        store.close()
        if env is not None:
            env.shutdown()

    summary_path = out_dir / 'summary.json'
    with summary_path.open('w', encoding='utf-8') as file:
        json.dump(summary, file, indent=2)
    print(summary_path)


if __name__ == '__main__':
    main()
