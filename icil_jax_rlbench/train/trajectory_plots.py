from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np


@dataclass(frozen=True)
class ActionChunkPlotData:
    name: str
    split: str
    pred_xyz: np.ndarray
    target_xyz: np.ndarray
    mse: float


def _axis_span(examples: Sequence[ActionChunkPlotData]) -> float:
    spans = []
    for example in examples:
        points = np.concatenate([example.pred_xyz, example.target_xyz], axis=0)
        spans.append(float(np.max(np.ptp(points, axis=0))))
    return max(max(spans, default=0.0), 1e-4)


def make_action_chunk_figures(examples: Sequence[ActionChunkPlotData]) -> Dict[str, object]:
    if not examples:
        return {}

    import plotly.graph_objects as go

    span = _axis_span(examples)
    half = 0.5 * span
    figures: Dict[str, object] = {}
    for example in examples:
        pred = np.asarray(example.pred_xyz, dtype=np.float32)
        target = np.asarray(example.target_xyz, dtype=np.float32)
        points = np.concatenate([pred, target], axis=0)
        center = 0.5 * (np.min(points, axis=0) + np.max(points, axis=0))
        axis = {
            'range': [float(center[0] - half), float(center[0] + half)],
            'title': 'x',
        }
        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=target[:, 0],
                y=target[:, 1],
                z=target[:, 2],
                mode='lines+markers',
                name='ground truth',
                line={'color': '#2563eb', 'width': 6},
                marker={'size': 3, 'color': '#2563eb'},
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=pred[:, 0],
                y=pred[:, 1],
                z=pred[:, 2],
                mode='lines+markers',
                name='prediction',
                line={'color': '#dc2626', 'width': 6},
                marker={'size': 3, 'color': '#dc2626'},
            )
        )
        fig.update_layout(
            title=f'{example.split} {example.name} | mse={example.mse:.6f}',
            scene={
                'xaxis': axis,
                'yaxis': {
                    'range': [float(center[1] - half), float(center[1] + half)],
                    'title': 'y',
                },
                'zaxis': {
                    'range': [float(center[2] - half), float(center[2] + half)],
                    'title': 'z',
                },
                'aspectmode': 'cube',
            },
            margin={'l': 0, 'r': 0, 't': 40, 'b': 0},
            legend={'orientation': 'h'},
        )
        figures[f'{example.split}_{example.name}'] = fig
    return figures
