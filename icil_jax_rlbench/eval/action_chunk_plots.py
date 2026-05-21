from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


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


def supernode_geometry(
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
    empty = {
        'centers': np.zeros((0, 3), dtype=np.float32),
        'weights': np.zeros((0, 0), dtype=np.float32),
        'edge_start': np.zeros((0, 3), dtype=np.float32),
        'edge_end': np.zeros((0, 3), dtype=np.float32),
        'edge_weights': np.zeros((0,), dtype=np.float32),
        'edge_distances': np.zeros((0,), dtype=np.float32),
    }
    if xyz_valid.shape[0] == 0 or int(num_supernodes) <= 0:
        return empty

    n = int(xyz_valid.shape[0])
    m = min(int(num_supernodes), n)
    center_idx = np.linspace(0, max(n - 1, 0), m).round().astype(np.int64)
    centers = xyz_valid[center_idx].astype(np.float32)
    if int(edge_top_k) <= 0:
        return {**empty, 'centers': centers}

    dist2 = np.sum((centers[:, None, :] - xyz_valid[None, :, :]) ** 2, axis=-1)
    logits = -dist2 / max(float(temperature), 1e-6)
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    weights = np.exp(logits)
    weights = weights / np.maximum(np.sum(weights, axis=-1, keepdims=True), 1e-12)

    draw_m = min(m, int(max_edge_supernodes))
    draw_dist2 = dist2[:draw_m]
    draw_weights = weights[:draw_m]
    candidates = max(
        int(edge_top_k) + (1 if bool(skip_self_edges) else 0),
        int(edge_top_k) * max(1, int(edge_candidate_multiplier)),
    )
    k = min(candidates, n)
    top_idx = np.argpartition(draw_dist2, kth=np.arange(k), axis=1)[:, :k]

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
            edge_weights.append(float(draw_weights[sidx, int(pidx)]))
            edge_distances.append(length)
            drawn += 1
            if drawn >= int(edge_top_k):
                break

    return {
        'centers': centers,
        'weights': weights.astype(np.float32),
        'edge_start': np.stack(edge_start, axis=0).astype(np.float32) if edge_start else np.zeros((0, 3), dtype=np.float32),
        'edge_end': np.stack(edge_end, axis=0).astype(np.float32) if edge_end else np.zeros((0, 3), dtype=np.float32),
        'edge_weights': np.asarray(edge_weights, dtype=np.float32),
        'edge_distances': np.asarray(edge_distances, dtype=np.float32),
    }


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


def write_online_action_chunk_html(
    *,
    frame: Dict[str, np.ndarray],
    plan: np.ndarray,
    current_state: np.ndarray,
    out_path: str | Path,
    title: str,
    handle_to_name: Optional[Dict[int, str]],
    num_supernodes: int,
    supernode_temperature: float,
    edge_top_k: int,
    max_edge_supernodes: int,
    skip_self_edges: bool,
    edge_min_length: float,
    edge_candidate_multiplier: int,
    edge_line_width: float,
    edge_opacity: float,
    marker_size: float,
    executed_actions: int,
) -> Dict[str, Any]:
    import plotly.graph_objects as go

    plan = np.asarray(plan, dtype=np.float32)
    xyz = np.asarray(frame['xyz'], dtype=np.float32).reshape((-1, 3))
    valid = np.asarray(frame.get('valid', np.ones((xyz.shape[0],), dtype=np.bool_))).reshape(-1).astype(np.bool_)
    rgb = frame.get('rgb')
    if rgb is not None:
        rgb = np.asarray(rgb).reshape((-1, 3))[valid]
    mask_id = frame.get('mask_id')
    if mask_id is not None:
        mask_id = np.asarray(mask_id).reshape((-1,))[valid]
    xyz = xyz[valid]
    colors = _rgb_strings(rgb, xyz.shape[0])
    hover = _hover_text(mask_id, handle_to_name)

    geom = supernode_geometry(
        frame['xyz'],
        frame.get('valid'),
        num_supernodes=num_supernodes,
        temperature=supernode_temperature,
        edge_top_k=edge_top_k,
        max_edge_supernodes=max_edge_supernodes,
        skip_self_edges=skip_self_edges,
        edge_min_length=edge_min_length,
        edge_candidate_multiplier=edge_candidate_multiplier,
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            mode='markers',
            marker={'size': float(marker_size), 'color': colors, 'opacity': 0.72},
            text=hover,
            hovertemplate='%{text}<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>' if hover else 'x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>',
            name='live query point cloud',
        )
    )

    centers = np.asarray(geom['centers'], dtype=np.float32)
    if centers.size:
        fig.add_trace(
            go.Scatter3d(
                x=centers[:, 0],
                y=centers[:, 1],
                z=centers[:, 2],
                mode='markers',
                marker={'size': float(marker_size) * 3.0, 'color': 'black', 'symbol': 'diamond', 'opacity': 0.95},
                name='supernodes',
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
                line={'color': 'rgb(255,80,0)', 'width': float(edge_line_width)},
                opacity=float(edge_opacity),
                name='top pooling edges',
                hoverinfo='skip',
                connectgaps=False,
            )
        )

    current_xyz = np.asarray(current_state, dtype=np.float32).reshape(-1)[:3]
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

    if plan.size:
        fig.add_trace(
            go.Scatter3d(
                x=plan[:, 0],
                y=plan[:, 1],
                z=plan[:, 2],
                mode='lines+markers',
                line={'color': '#dc2626', 'width': 8},
                marker={'size': 4, 'color': '#dc2626'},
                name='predicted action chunk',
            )
        )
        n_exec = min(max(0, int(executed_actions)), int(plan.shape[0]))
        if n_exec > 0:
            prefix = plan[:n_exec]
            fig.add_trace(
                go.Scatter3d(
                    x=prefix[:, 0],
                    y=prefix[:, 1],
                    z=prefix[:, 2],
                    mode='lines+markers',
                    line={'color': '#f59e0b', 'width': 11},
                    marker={'size': 6, 'color': '#f59e0b'},
                    name='executed prefix',
                )
            )

    ranges = _axis_ranges(frame, [plan], current_xyz)
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
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs='cdn')
    return {
        'path': str(out_path),
        'points': int(np.sum(np.asarray(frame.get('valid', np.ones((0,), dtype=np.bool_)), dtype=np.bool_))),
        'supernodes': int(centers.shape[0]),
        'edges': int(edge_start.shape[0]),
        'plan_len': int(plan.shape[0]) if plan.ndim >= 2 else 0,
        'executed_actions': int(executed_actions),
    }
