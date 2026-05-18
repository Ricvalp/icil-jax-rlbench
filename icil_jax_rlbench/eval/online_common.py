from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from absl import logging
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
import numpy as np
import optax

from icil_jax_rlbench.data.action_representation import decode_action_chunk
from icil_jax_rlbench.data.h5_cache import RLBenchCacheStore, VariationKey, build_variation_keys
from icil_jax_rlbench.data.sampler import ICILDataConfig, ICILSampler
from icil_jax_rlbench.models.config import policy_config_from
from icil_jax_rlbench.models.direct_regression_policy import DirectRegressionPolicy
from icil_jax_rlbench.train.checkpoints import load_checkpoint
from icil_jax_rlbench.train.step import action_loss, apply_mask, clip_tree_by_global_norm, make_name_mask


_CAMERAS: Tuple[str, ...] = (
    'left_shoulder',
    'right_shoulder',
    'overhead',
    'wrist',
    'front',
)

_MASK_NAMES_TO_IGNORE = (
    'Floor',
    'Wall1',
    'Wall2',
    'Wall3',
    'Wall4',
    'Roof',
    'workspace',
    'diningTable_visible',
)

_MASK_NAME_SUBSTRINGS_TO_IGNORE = (
    'floor',
    'wall',
    'roof',
    'workspace',
    'table',
    'panda_link',
)

_DEFAULT_WORKSPACE_BOUNDS = ((-1.0, 1.0), (-1.0, 1.0), (0.0, 2.5))


def _as_bool(value: Any) -> bool:
    return bool(value)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _cfg_get(cfg: Any, name: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _checkpoint_config(ckpt: Dict[str, Any]) -> ConfigDict:
    raw = ckpt.get('config', {}) if isinstance(ckpt, dict) else {}
    return ConfigDict(raw or {})


def _model_conditioning_mode(model_cfg: Any) -> str:
    return str(getattr(getattr(model_cfg, 'conditioning', ConfigDict()), 'mode', 'support'))


def _task_variation_ids_from_checkpoint(ckpt_cfg: ConfigDict, task_name: str, variation: int) -> Optional[Dict[str, np.ndarray]]:
    if _model_conditioning_mode(ckpt_cfg.model) != 'task_variation':
        return None
    data_cfg = getattr(ckpt_cfg, 'data', ConfigDict())
    task_names = list(getattr(data_cfg, 'task_id_names', ()))
    variation_keys = list(getattr(data_cfg, 'task_variation_keys', ()))
    if not task_names or not variation_keys:
        raise ValueError('Task-variation checkpoint is missing data.task_id_names or data.task_variation_keys.')
    variation_key = f'{task_name}:{int(variation)}'
    if task_name not in task_names:
        raise ValueError(f'Task {task_name!r} is not in the checkpoint task-token vocabulary.')
    if variation_key not in variation_keys:
        raise ValueError(f'Variation {variation_key!r} is not in the checkpoint task-variation-token vocabulary.')
    return {
        'task_id': np.asarray([task_names.index(task_name)], dtype=np.int32),
        'task_variation_id': np.asarray([variation_keys.index(variation_key)], dtype=np.int32),
    }


def _data_config_from_eval_and_checkpoint(cfg: ConfigDict, ckpt: Dict[str, Any]) -> ICILDataConfig:
    ckpt_cfg = _checkpoint_config(ckpt)
    ckpt_data = getattr(ckpt_cfg, 'data', ConfigDict())
    eval_data = getattr(cfg, 'dataset', ConfigDict())
    use_ckpt = bool(getattr(eval_data, 'use_checkpoint_dataset_config', True))

    def value(name: str, default: Any) -> Any:
        if use_ckpt and hasattr(ckpt_data, name):
            return getattr(ckpt_data, name)
        return getattr(eval_data, name, default)

    return ICILDataConfig(
        K=int(value('K', 2)),
        L=int(value('L', 8)),
        T_obs=int(value('T_obs', 2)),
        H=int(value('H', 16)),
        stride=int(value('stride', 2)),
        action_representation=str(value('action_representation', 'absolute')),
        task_sampling='variation_uniform',
        task_sampling_alpha=1.0,
        traj_len=int(value('traj_len', 64)),
    )


def _query_stride_mode(cfg: ConfigDict) -> str:
    mode = str(getattr(cfg.dataset, 'query_stride_mode', 'consecutive')).lower()
    if mode not in ('dataset', 'consecutive'):
        raise ValueError("dataset.query_stride_mode must be 'dataset' or 'consecutive'.")
    return mode


def _support_cache_root(cfg: ConfigDict, ckpt: Dict[str, Any]) -> Path:
    root = str(getattr(cfg.conditioning, 'cache_root', '') or '').strip()
    if not root:
        ckpt_cfg = _checkpoint_config(ckpt)
        root = str(getattr(getattr(ckpt_cfg, 'data', ConfigDict()), 'cache_root', '') or '').strip()
    if not root:
        raise ValueError('Set conditioning.cache_root or train with config.data.cache_root saved in the checkpoint.')
    path = Path(root).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f'cache_root not found: {path}')
    return path


def _choose_variation_key(cache_root: Path, task_name: str, variation: int, rng: np.random.Generator) -> VariationKey:
    keys = build_variation_keys(cache_root, task_name)
    if not keys:
        raise RuntimeError(f'No cached variations found for task={task_name!r} under {cache_root}.')
    if int(variation) >= 0:
        matches = [k for k in keys if int(k.variation) == int(variation)]
        if not matches:
            available = sorted({int(k.variation) for k in keys})
            raise RuntimeError(f'No cached variation={variation} for task={task_name!r}. Available: {available}')
        return matches[0]
    return keys[int(rng.integers(0, len(keys)))]


def _subsample_fixed_n(rng: np.random.Generator, count: int, target: int) -> np.ndarray:
    if int(count) >= int(target):
        return rng.choice(int(count), size=int(target), replace=False)
    return rng.choice(int(count), size=int(target), replace=True)


def _filter_by_ignore_ids(masks: np.ndarray, ignore_ids: Sequence[int]) -> np.ndarray:
    if not ignore_ids:
        return np.ones_like(masks, dtype=np.bool_)
    keep = np.ones_like(masks, dtype=np.bool_)
    for mid in ignore_ids:
        keep &= masks != int(mid)
    return keep


def _filter_by_xyz_bounds(points: np.ndarray, bounds: Sequence[Sequence[float]]) -> np.ndarray:
    xb, yb, zb = bounds
    return (
        (points[:, 0] >= float(xb[0])) & (points[:, 0] <= float(xb[1]))
        & (points[:, 1] >= float(yb[0])) & (points[:, 1] <= float(yb[1]))
        & (points[:, 2] >= float(zb[0])) & (points[:, 2] <= float(zb[1]))
    )


def _workspace_bounds_from_cfg(cfg: ConfigDict) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]]:
    if not bool(getattr(cfg.conditioning, 'filter_workspace_bounds', True)):
        return None
    raw = getattr(cfg.conditioning, 'workspace_bounds', _DEFAULT_WORKSPACE_BOUNDS)
    if len(raw) != 3:
        raise ValueError('conditioning.workspace_bounds must contain x/y/z bounds.')
    bounds = []
    for axis_bounds in raw:
        if len(axis_bounds) != 2:
            raise ValueError('Each conditioning.workspace_bounds axis must contain two values.')
        bounds.append((float(axis_bounds[0]), float(axis_bounds[1])))
    return bounds[0], bounds[1], bounds[2]


def _build_vector(obs: Any, keys: Sequence[str]) -> np.ndarray:
    parts = []
    for key in keys:
        value = getattr(obs, key)
        if value is None:
            raise ValueError(f"Observation attribute {key!r} is None.")
        parts.append(np.asarray(value, dtype=np.float32).reshape(-1))
    return np.concatenate(parts, axis=0).astype(np.float32)


def _normalize_quaternion_xyzw(q: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (q / norm).astype(np.float32)


def _sanitize_action(action: np.ndarray, *, normalize_quaternion: bool, discretize_gripper: bool) -> np.ndarray:
    out = np.asarray(action, dtype=np.float32).copy()
    if out.shape[0] >= 7 and normalize_quaternion:
        out[3:7] = _normalize_quaternion_xyzw(out[3:7])
    if out.shape[0] >= 8:
        if discretize_gripper:
            out[7] = 1.0 if float(out[7]) > 0.5 else 0.0
        else:
            out[7] = float(np.clip(out[7], 0.0, 1.0))
    return out


def _position_bounds_from_cfg(cfg: ConfigDict) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    raw = getattr(cfg.control, 'action_position_bounds', _DEFAULT_WORKSPACE_BOUNDS)
    if len(raw) != 3:
        raise ValueError('control.action_position_bounds must contain x/y/z bounds.')
    bounds = []
    for axis_bounds in raw:
        if len(axis_bounds) != 2:
            raise ValueError('Each control.action_position_bounds axis must contain two values.')
        bounds.append((float(axis_bounds[0]), float(axis_bounds[1])))
    return bounds[0], bounds[1], bounds[2]


def _validate_action_for_planning(
    action: np.ndarray,
    current_state: np.ndarray,
    cfg: ConfigDict,
) -> Optional[str]:
    if not np.all(np.isfinite(action)):
        return f'Predicted non-finite action: {action}'
    if action.shape[0] < 3:
        return f'Predicted action has invalid shape: {action.shape}'
    pos = np.asarray(action[:3], dtype=np.float32)
    if bool(getattr(cfg.control, 'reject_out_of_bounds_actions', False)):
        bounds = _position_bounds_from_cfg(cfg)
        for axis, value, axis_bounds in zip('xyz', pos.tolist(), bounds):
            if value < axis_bounds[0] or value > axis_bounds[1]:
                return (
                    f'Predicted end-effector {axis}={value:.4f} is outside '
                    f'control.action_position_bounds={bounds}.'
                )
    max_delta = float(getattr(cfg.control, 'max_position_delta', 0.0))
    if max_delta > 0.0:
        current_pos = np.asarray(current_state[:3], dtype=np.float32)
        delta = float(np.linalg.norm(pos - current_pos))
        if delta > max_delta:
            return (
                f'Predicted end-effector move is too large: delta={delta:.4f} > '
                f'control.max_position_delta={max_delta:.4f}.'
            )
    return None


def _extract_rgb_frame(obs: Any, camera: str) -> np.ndarray:
    frame = getattr(obs, f'{camera}_rgb', None)
    if frame is None:
        frame = getattr(obs, 'front_rgb', None)
    if frame is None:
        return np.zeros((128, 128, 3), dtype=np.uint8)
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    return arr


def _write_video(frames: Sequence[np.ndarray], out_path: Path, fps: int) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == '.mp4':
        try:
            import imageio.v2 as imageio

            imageio.mimsave(str(out_path), list(frames), fps=int(fps))
            return out_path
        except Exception as exc:
            logging.warning('MP4 export failed (%s); falling back to GIF.', exc)
            out_path = out_path.with_suffix('.gif')
    if out_path.suffix.lower() == '.gif':
        try:
            from PIL import Image

            pil_frames = [Image.fromarray(np.asarray(frame, dtype=np.uint8)) for frame in frames]
            pil_frames[0].save(
                out_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=max(1, int(round(1000.0 / max(1, int(fps))))),
                loop=0,
            )
            return out_path
        except Exception as exc:
            logging.warning('GIF export failed (%s); saving raw npz frames.', exc)
    fallback = out_path.with_suffix('.npz')
    np.savez_compressed(str(fallback), frames=np.asarray(frames, dtype=np.uint8))
    return fallback


class LiveObservationProcessor:
    def __init__(
        self,
        *,
        task_env: Any,
        num_points: int,
        use_rgb: bool,
        use_mask_id: bool,
        workspace_bounds: Optional[Sequence[Sequence[float]]],
        seed: int,
    ):
        self.task_env = task_env
        self.num_points = int(num_points)
        self.use_rgb = bool(use_rgb)
        self.use_mask_id = bool(use_mask_id)
        self.workspace_bounds = workspace_bounds
        self.rng = np.random.default_rng(int(seed))
        self.handle_to_name: Dict[int, str] = {}

    def _update_handle_names(self, mask_views: Sequence[np.ndarray]) -> None:
        from rlbench.segmentation_utils import build_handle_label_map

        unresolved: set[int] = set()
        for mask in mask_views:
            arr = np.asarray(mask).reshape(-1).astype(np.int64, copy=False)
            for value in np.unique(arr):
                handle = int(value)
                if handle != 0 and handle not in self.handle_to_name:
                    unresolved.add(handle)
        if not unresolved:
            return
        mapping, _outstanding = build_handle_label_map(self.task_env, unresolved)
        for handle, name in mapping.items():
            self.handle_to_name[int(handle)] = str(name)

    def _ignore_ids(self) -> Tuple[int, ...]:
        ignore = set()
        for handle, name in self.handle_to_name.items():
            if name in _MASK_NAMES_TO_IGNORE:
                ignore.add(int(handle))
                continue
            lower = str(name).lower()
            if any(token in lower for token in _MASK_NAME_SUBSTRINGS_TO_IGNORE):
                ignore.add(int(handle))
        return tuple(sorted(ignore))

    def observation_to_frame(self, obs: Any) -> Dict[str, np.ndarray]:
        points: List[np.ndarray] = []
        colors: List[np.ndarray] = []
        masks: List[np.ndarray] = []
        mask_views: List[np.ndarray] = []
        for camera in _CAMERAS:
            point_cloud = getattr(obs, f'{camera}_point_cloud', None)
            mask = getattr(obs, f'{camera}_mask', None)
            rgb = getattr(obs, f'{camera}_rgb', None)
            if point_cloud is None or mask is None:
                continue
            pts = np.asarray(point_cloud, dtype=np.float32).reshape(-1, 3)
            msk = np.asarray(mask).reshape(-1).astype(np.int32, copy=False)
            col = None
            if self.use_rgb:
                col = np.zeros((pts.shape[0], 3), dtype=np.uint8) if rgb is None else np.asarray(rgb).reshape(-1, 3).astype(np.uint8, copy=False)
            finite = np.isfinite(pts).all(axis=1)
            pts = pts[finite]
            msk = msk[finite]
            if col is not None:
                col = col[finite]
            if pts.shape[0] == 0:
                continue
            points.append(pts)
            masks.append(msk)
            if col is not None:
                colors.append(col)
            mask_views.append(msk)

        if points:
            pts_all = np.concatenate(points, axis=0).astype(np.float32, copy=False)
            msk_all = np.concatenate(masks, axis=0).astype(np.int32, copy=False)
            col_all = np.concatenate(colors, axis=0).astype(np.uint8, copy=False) if self.use_rgb and colors else None
        else:
            pts_all = np.zeros((0, 3), dtype=np.float32)
            msk_all = np.zeros((0,), dtype=np.int32)
            col_all = np.zeros((0, 3), dtype=np.uint8) if self.use_rgb else None

        self._update_handle_names(mask_views)
        keep = _filter_by_ignore_ids(msk_all, self._ignore_ids())
        pts_all = pts_all[keep]
        msk_all = msk_all[keep]
        if col_all is not None:
            col_all = col_all[keep]
        if self.workspace_bounds is not None and pts_all.shape[0] > 0:
            keep = _filter_by_xyz_bounds(pts_all, self.workspace_bounds)
            pts_all = pts_all[keep]
            msk_all = msk_all[keep]
            if col_all is not None:
                col_all = col_all[keep]

        if pts_all.shape[0] == 0:
            xyz = np.zeros((self.num_points, 3), dtype=np.float32)
            valid = np.zeros((self.num_points,), dtype=np.bool_)
            rgb = np.zeros((self.num_points, 3), dtype=np.float32) if self.use_rgb else None
            mask_id = np.zeros((self.num_points,), dtype=np.int32) if self.use_mask_id else None
        else:
            idx = _subsample_fixed_n(self.rng, int(pts_all.shape[0]), self.num_points)
            xyz = pts_all[idx].astype(np.float32, copy=False)
            valid = np.ones((self.num_points,), dtype=np.bool_)
            rgb = col_all[idx].astype(np.float32, copy=False) / 255.0 if self.use_rgb and col_all is not None else None
            mask_id = msk_all[idx].astype(np.int32, copy=False) if self.use_mask_id else None

        out: Dict[str, np.ndarray] = {
            'xyz': xyz,
            'valid': valid,
            'state': _build_vector(obs, ('gripper_pose', 'gripper_open')),
        }
        if rgb is not None:
            out['rgb'] = rgb
        if mask_id is not None:
            out['mask_id'] = mask_id
        return out


def build_rlbench_env(cfg: ConfigDict, task_name: str):
    from pyrep.const import RenderMode
    from rlbench import ObservationConfig
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.backend.utils import task_file_to_task_class
    from rlbench.environment import Environment

    obs_config = ObservationConfig()
    obs_config.set_all(True)
    image_size = tuple(int(x) for x in cfg.sim.image_size)
    renderer_name = str(cfg.sim.renderer).lower()
    if renderer_name == 'opengl':
        render_mode = RenderMode.OPENGL
    elif renderer_name == 'opengl3':
        render_mode = RenderMode.OPENGL3
    else:
        raise ValueError(f'Unsupported renderer={cfg.sim.renderer!r}.')
    for camera in _CAMERAS:
        camera_cfg = getattr(obs_config, f'{camera}_camera')
        camera_cfg.image_size = image_size
        camera_cfg.depth_in_meters = False
        camera_cfg.masks_as_one_channel = True
        camera_cfg.render_mode = render_mode

    action_mode = MoveArmThenGripper(
        EndEffectorPoseViaPlanning(
            absolute_mode=True,
            collision_checking=_as_bool(cfg.sim.collision_checking),
        ),
        Discrete(),
    )
    env = Environment(
        action_mode=action_mode,
        obs_config=obs_config,
        headless=_as_bool(cfg.sim.headless),
        arm_max_velocity=float(cfg.sim.arm_max_velocity),
        arm_max_acceleration=float(cfg.sim.arm_max_acceleration),
    )
    env.launch()
    task_env = env.get_task(task_file_to_task_class(task_name))
    return env, task_env


def _build_query_window(history: Sequence[Dict[str, np.ndarray]], *, data_cfg: ICILDataConfig, query_stride_mode: str) -> Dict[str, np.ndarray]:
    if not history:
        raise RuntimeError('Query history is empty.')
    last = len(history) - 1
    qstep = int(data_cfg.stride) if query_stride_mode == 'dataset' else 1
    idx = [max(0, last - (int(data_cfg.T_obs) - 1 - i) * qstep) for i in range(int(data_cfg.T_obs))]
    frames = [history[i] for i in idx]
    out: Dict[str, np.ndarray] = {
        'query_xyz': np.stack([f['xyz'] for f in frames], axis=0)[None].astype(np.float32),
        'query_state': np.stack([f['state'] for f in frames], axis=0)[None].astype(np.float32),
        'query_valid': np.stack([f['valid'] for f in frames], axis=0)[None].astype(np.bool_),
    }
    if all('rgb' in f for f in frames):
        out['query_rgb'] = np.stack([f['rgb'] for f in frames], axis=0)[None].astype(np.float32)
    if all('mask_id' in f for f in frames):
        out['query_mask_id'] = np.stack([f['mask_id'] for f in frames], axis=0)[None].astype(np.int32)
    return out


def _select_support_ids(store: RLBenchCacheStore, data_cfg: ICILDataConfig, rng: np.random.Generator) -> np.ndarray:
    episode_ids = store.list_episode_ids(0)
    if len(episode_ids) < int(data_cfg.K):
        raise RuntimeError(f'Need at least K={data_cfg.K} cached support episodes, got {len(episode_ids)}.')
    return rng.choice(episode_ids, size=int(data_cfg.K), replace=False).astype(np.int64)


def _build_cached_support(
    *,
    store: RLBenchCacheStore,
    sampler: ICILSampler,
    data_cfg: ICILDataConfig,
    rng: np.random.Generator,
    use_rgb: bool,
    use_mask_id: bool,
) -> Dict[str, Any]:
    support_ids = _select_support_ids(store, data_cfg, rng)
    support = sampler.build_support_conditioning(
        vidx=0,
        support_ids=support_ids,
        load_rgb=use_rgb,
        load_mask_id=use_mask_id,
    )
    support = {k: (v[None] if isinstance(v, np.ndarray) else v) for k, v in support.items()}
    support['meta'] = {
        'task': store.keys[0].task,
        'variation': int(store.keys[0].variation),
        'support_episodes': [int(x) for x in np.asarray(support_ids).tolist()],
    }
    return support


def _build_param_inner_batch(
    *,
    sampler: ICILSampler,
    support_ids: Sequence[int],
    inner_steps: int,
    num_inner_queries: int,
    rng: np.random.Generator,
    load_rgb: bool,
    load_mask_id: bool,
) -> Dict[str, np.ndarray]:
    if int(inner_steps) <= 0:
        return {}
    support_ids = [int(x) for x in support_ids]
    if len(support_ids) < 2:
        raise ValueError('Param-MAML adaptation needs at least K=2 support episodes.')
    steps: List[Dict[str, np.ndarray]] = []
    queries = max(1, int(num_inner_queries))
    for _ in range(int(inner_steps)):
        samples = []
        order = list(rng.permutation(len(support_ids)))
        while len(order) < queries:
            order.extend(list(rng.permutation(len(support_ids))))
        for holdout_idx in order[:queries]:
            heldout = support_ids[int(holdout_idx)]
            context = [eid for i, eid in enumerate(support_ids) if i != int(holdout_idx)]
            samples.append(
                sampler._build_context_query_sample(
                    vidx=0,
                    context_ids=context,
                    query_episode_id=heldout,
                    load_rgb=load_rgb,
                    load_mask_id=load_mask_id,
                )
            )
        steps.append(sampler._stack(samples))
    return {k: np.stack([step[k] for step in steps], axis=0) for k in steps[0]}


def _make_param_adapt_fn(model: DirectRegressionPolicy, *, inner_lr: float, grad_clip_norm: float, loss_type: str, inner_mask: Any):
    def loss_fn(params, batch):
        pred = model.apply({'params': params}, batch, train=False)
        return action_loss(pred, batch['target_action'], loss_type)

    def adapt(params, inner_batch):
        def body(p, step_batch):
            loss, grads = jax.value_and_grad(loss_fn)(p, step_batch)
            grads = apply_mask(grads, inner_mask)
            grads, grad_norm = clip_tree_by_global_norm(grads, float(grad_clip_norm))
            updates = jax.tree_util.tree_map(lambda g: -float(inner_lr) * g, grads)
            return optax.apply_updates(p, updates), {'inner_loss': loss, 'inner_grad_norm': grad_norm}

        adapted, metrics = jax.lax.scan(body, params, inner_batch)
        metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x), metrics)
        return adapted, metrics

    return jax.jit(adapt)


class JaxOnlinePolicy:
    def __init__(
        self,
        model: DirectRegressionPolicy,
        params: Any,
        data_cfg: ICILDataConfig,
        *,
        fixed_conditioning: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.model = model
        self.params = jax.device_put(params)
        self.data_cfg = data_cfg
        self.conditioning_mode = str(model.cfg.conditioning.mode)
        self.uses_support = self.conditioning_mode == 'support' and bool(model.cfg.encoder.use_support_tokens)
        self.fixed_conditioning = fixed_conditioning or {}
        self.support_tokens = None
        self.support_mask = None
        self._encode_support = jax.jit(
            lambda params, batch: model.apply(
                {'params': params},
                batch,
                train=False,
                method=DirectRegressionPolicy.encode_support,
            )
        )
        self._predict_with_memory = jax.jit(
            lambda params, batch, memory, mask: model.apply(
                {'params': params},
                batch,
                memory,
                support_mask=mask,
                train=False,
                method=DirectRegressionPolicy.predict_with_memory,
            )
        )
        self._predict_query_only = jax.jit(
            lambda params, batch: model.apply({'params': params}, batch, train=False)
        )

    def update_params(self, params: Any) -> None:
        self.params = jax.device_put(params)
        self.support_tokens = None
        self.support_mask = None

    def update_support(self, support: Dict[str, np.ndarray]) -> None:
        if not self.uses_support:
            self.support_tokens = None
            self.support_mask = None
            return
        batch = {k: v for k, v in support.items() if k != 'meta'}
        self.support_tokens, self.support_mask = self._encode_support(self.params, batch)
        jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x, self.support_tokens)

    def predict_action_chunk(self, query: Dict[str, np.ndarray]) -> np.ndarray:
        if self.fixed_conditioning:
            query = {**query, **self.fixed_conditioning}
        if not self.uses_support:
            pred = self._predict_query_only(self.params, query)
            pred = np.asarray(jax.device_get(pred), dtype=np.float32)
            return decode_action_chunk(pred, query_state=query['query_state'], representation=self.data_cfg.action_representation)
        if self.support_tokens is None or self.support_mask is None:
            raise RuntimeError('Support conditioning has not been encoded.')
        pred = self._predict_with_memory(self.params, query, self.support_tokens, self.support_mask)
        pred = np.asarray(jax.device_get(pred), dtype=np.float32)
        return decode_action_chunk(pred, query_state=query['query_state'], representation=self.data_cfg.action_representation)


def _maybe_init_wandb(cfg: ConfigDict, run_dir: Path):
    if not hasattr(cfg, 'wandb') or not bool(getattr(cfg.wandb, 'enable', False)):
        return None
    import wandb

    return wandb.init(
        project=str(getattr(cfg.wandb, 'project', 'icil-jax-rlbench-eval')),
        entity=str(getattr(cfg.wandb, 'entity', '') or '') or None,
        name=str(getattr(cfg.wandb, 'name', '') or '') or None,
        mode=str(getattr(cfg.wandb, 'mode', 'online')),
        dir=str(run_dir),
        config=cfg.to_dict(),
    )


def _run_episode(
    *,
    episode_index: int,
    task_env: Any,
    variation: int,
    policy: JaxOnlinePolicy,
    processor: LiveObservationProcessor,
    data_cfg: ICILDataConfig,
    query_stride_mode: str,
    cfg: ConfigDict,
    run_dir: Path,
) -> Dict[str, Any]:
    from rlbench.backend.exceptions import InvalidActionError

    if int(variation) >= 0:
        task_env.set_variation(int(variation))
    _descriptions, obs = task_env.reset()
    history = [processor.observation_to_frame(obs)]
    logging.info(
        'Episode %d reset complete | initial_gripper_xyz=%s',
        episode_index,
        np.array2string(history[-1]['state'][:3], precision=4),
    )
    frames: List[np.ndarray] = []
    if _as_bool(cfg.video.enable):
        frames.append(_extract_rgb_frame(obs, str(cfg.video.camera)))

    success = False
    terminated = False
    error: Optional[str] = None
    env_steps = 0
    max_env_steps = int(cfg.task.max_env_steps)
    execute_actions = max(1, int(cfg.control.execute_actions_per_plan))

    try:
        from tqdm.auto import tqdm

        pbar = tqdm(total=max_env_steps, desc=f'Episode {episode_index}', leave=False, unit='step')
    except Exception:
        pbar = None

    try:
        while env_steps < max_env_steps and not success and not terminated:
            query = _build_query_window(history, data_cfg=data_cfg, query_stride_mode=query_stride_mode)
            current_state = np.asarray(query['query_state'][0, -1], dtype=np.float32)
            predict_start = time.time()
            plan_np = policy.predict_action_chunk(query)[0]
            predict_s = time.time() - predict_start
            if env_steps == 0:
                first_delta = float(np.linalg.norm(plan_np[0, :3] - current_state[:3]))
                logging.info(
                    'Episode %d first action chunk | predict_s=%.3f current_xyz=%s first_action_xyz=%s delta=%.4f',
                    episode_index,
                    predict_s,
                    np.array2string(current_state[:3], precision=4),
                    np.array2string(plan_np[0, :3], precision=4),
                    first_delta,
                )
            n_exec = int(min(execute_actions, plan_np.shape[0], max_env_steps - env_steps))
            for i in range(n_exec):
                action = _sanitize_action(
                    plan_np[i],
                    normalize_quaternion=_as_bool(cfg.control.normalize_quaternion),
                    discretize_gripper=_as_bool(cfg.control.discretize_gripper),
                )
                invalid_reason = _validate_action_for_planning(action, current_state, cfg)
                if invalid_reason is not None:
                    error = invalid_reason
                    terminated = True
                    logging.warning('Episode %d rejecting action before RLBench planning | %s', episode_index, invalid_reason)
                    break
                if env_steps == 0 and i == 0:
                    logging.info(
                        'Episode %d executing first action | action=%s',
                        episode_index,
                        np.array2string(action, precision=4),
                    )
                try:
                    obs, reward, terminated = task_env.step(action.astype(np.float32))
                except InvalidActionError as exc:
                    error = f'InvalidActionError: {exc}'
                    terminated = True
                    break
                except Exception as exc:
                    error = f'{type(exc).__name__}: {exc}'
                    terminated = True
                    break
                env_steps += 1
                if pbar is not None:
                    pbar.update(1)
                success = bool(float(reward) > 0.5)
                history.append(processor.observation_to_frame(obs))
                current_state = np.asarray(history[-1]['state'], dtype=np.float32)
                if _as_bool(cfg.video.enable):
                    frames.append(_extract_rgb_frame(obs, str(cfg.video.camera)))
                if success or terminated or env_steps >= max_env_steps:
                    break
    finally:
        if pbar is not None:
            pbar.close()

    video_path = None
    if _as_bool(cfg.video.enable) and frames:
        video_file = run_dir / 'videos' / f'episode_{episode_index:04d}.{str(cfg.video.format).lower()}'
        video_path = str(_write_video(frames, video_file, fps=int(cfg.video.fps)))

    return {
        'episode_index': int(episode_index),
        'success': bool(success),
        'terminated': bool(terminated),
        'env_steps': int(env_steps),
        'error': error,
        'video_path': video_path,
    }


def evaluate(cfg: ConfigDict, *, adaptation: str) -> None:
    adaptation = str(adaptation)
    if adaptation not in ('none', 'param_maml'):
        raise ValueError("adaptation must be 'none' or 'param_maml'.")
    seed = int(cfg.seed)
    _set_seed(seed)
    rng = np.random.default_rng(seed + 17)

    checkpoint_path = Path(str(cfg.checkpoint_path)).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    ckpt = load_checkpoint(checkpoint_path)
    ckpt_cfg = _checkpoint_config(ckpt)
    data_cfg = _data_config_from_eval_and_checkpoint(cfg, ckpt)
    query_mode = _query_stride_mode(cfg)
    workspace_bounds = _workspace_bounds_from_cfg(cfg)
    task_name = str(cfg.task.name)
    variation = int(cfg.task.variation)
    cache_root = _support_cache_root(cfg, ckpt)
    key = _choose_variation_key(cache_root, task_name, variation, rng)
    store = RLBenchCacheStore([key], keep_open=bool(getattr(cfg.conditioning, 'keep_open', True)), preload_to_memory=False)
    sampler = ICILSampler(store, data_cfg, seed=seed + 23)
    num_points, state_dim, action_dim = store.infer_dims()
    if int(getattr(cfg.conditioning, 'num_points', 0)) <= 0:
        cfg.conditioning.num_points = int(num_points)

    policy_cfg = policy_config_from(ckpt_cfg.model, H=data_cfg.H, data_cfg=data_cfg)
    fixed_conditioning = _task_variation_ids_from_checkpoint(ckpt_cfg, task_name, int(key.variation))
    uses_support = str(policy_cfg.conditioning.mode) == 'support' and bool(policy_cfg.encoder.use_support_tokens)
    if adaptation == 'param_maml' and not uses_support:
        raise ValueError('Param-MAML online eval requires a checkpoint with model.encoder.use_support_tokens=True.')
    model = DirectRegressionPolicy(policy_cfg, state_dim=state_dim, action_dim=action_dim)
    params = ckpt['params']
    policy = JaxOnlinePolicy(model, params, data_cfg, fixed_conditioning=fixed_conditioning)

    use_rgb = bool(policy_cfg.encoder.use_rgb)
    use_mask_id = bool(policy_cfg.encoder.use_mask_id)
    if use_rgb and not bool(getattr(cfg.conditioning, 'use_rgb', True)):
        logging.warning('Checkpoint model uses RGB; forcing conditioning.use_rgb=True for shape compatibility.')
    if use_mask_id and not bool(getattr(cfg.conditioning, 'use_mask_id', False)):
        logging.warning('Checkpoint model uses mask ids; forcing conditioning.use_mask_id=True for shape compatibility.')

    output_root = Path(str(cfg.output.root_dir)).expanduser().resolve()
    run_dir = output_root / f'{task_name}_var{variation}_{time.strftime("%Y%m%d-%H%M%S")}'
    run_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = _maybe_init_wandb(cfg, run_dir)
    if wandb_run is not None:
        run_dir = output_root / f'{task_name}_var{variation}_{wandb_run.id}'
        run_dir.mkdir(parents=True, exist_ok=True)
        wandb_run.name = str(getattr(cfg.wandb, 'name', '') or run_dir.name)

    resolved_cfg = cfg.to_dict()
    resolved_cfg['resolved'] = {
        'checkpoint_path': str(checkpoint_path),
        'checkpoint_step': int(ckpt.get('step', -1)),
        'run_dir': str(run_dir),
        'adaptation': adaptation,
        'data': {
            'K': int(data_cfg.K),
            'L': int(data_cfg.L),
            'T_obs': int(data_cfg.T_obs),
            'H': int(data_cfg.H),
            'stride': int(data_cfg.stride),
            'traj_len': int(data_cfg.traj_len),
            'action_representation': str(data_cfg.action_representation),
        },
        'model_encoder': str(policy_cfg.encoder.encoder_type),
        'model_conditioning_mode': str(policy_cfg.conditioning.mode),
        'model_uses_support_tokens': bool(uses_support),
        'model_encoder_max_positions': int(policy_cfg.encoder.max_positions),
        'cache_variation': int(key.variation),
        'workspace_bounds': workspace_bounds,
    }
    config_path = run_dir / 'resolved_eval_config.json'
    with config_path.open('w', encoding='utf-8') as file:
        json.dump(resolved_cfg, file, indent=2)

    logging.info('Online eval | task=%s variation=%d checkpoint=%s', task_name, variation, checkpoint_path)
    logging.info('Resolved data cfg: K=%d L=%d T_obs=%d H=%d stride=%d traj_len=%d action=%s', data_cfg.K, data_cfg.L, data_cfg.T_obs, data_cfg.H, data_cfg.stride, data_cfg.traj_len, data_cfg.action_representation)
    logging.info(
        'Model: encoder=%s conditioning=%s uses_support=%s max_positions=%d params_step=%s action_dim=%d state_dim=%d points=%d',
        policy_cfg.encoder.encoder_type,
        str(policy_cfg.conditioning.mode),
        uses_support,
        int(policy_cfg.encoder.max_positions),
        ckpt.get('step', None),
        action_dim,
        state_dim,
        num_points,
    )
    logging.info('Live point filtering: num_points=%d workspace_bounds=%s query_stride_mode=%s', int(cfg.conditioning.num_points), workspace_bounds, query_mode)
    if not uses_support and fixed_conditioning is None:
        logging.info('Checkpoint disables support tokens; online eval will run query-only without cached support demos.')
    if fixed_conditioning is not None:
        logging.info(
            'Checkpoint uses task-variation conditioning without cached support demos: task_id=%d task_variation_id=%d',
            int(fixed_conditioning['task_id'][0]),
            int(fixed_conditioning['task_variation_id'][0]),
        )

    env = None
    results: List[Dict[str, Any]] = []
    support: Optional[Dict[str, Any]] = None
    adapted_params = params
    adapt_metrics: Optional[Dict[str, float]] = None
    try:
        env, task_env = build_rlbench_env(cfg, task_name)
        processor = LiveObservationProcessor(
            task_env=task_env,
            num_points=int(cfg.conditioning.num_points),
            use_rgb=use_rgb,
            use_mask_id=use_mask_id,
            workspace_bounds=workspace_bounds,
            seed=seed + 11,
        )
        for episode in range(int(cfg.task.num_eval_episodes)):
            if uses_support:
                regen = support is None or bool(getattr(cfg.conditioning, 'regenerate_demos_each_episode', False))
                regen = regen or (adaptation == 'param_maml' and bool(getattr(getattr(cfg, 'adaptation', ConfigDict()), 'regenerate_each_episode', False)))
                if regen:
                    support = _build_cached_support(
                        store=store,
                        sampler=sampler,
                        data_cfg=data_cfg,
                        rng=rng,
                        use_rgb=use_rgb,
                        use_mask_id=use_mask_id,
                    )
                    adapted_params = params
                    adapt_metrics = None
                    if adaptation == 'param_maml':
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
                            support_ids=support['meta']['support_episodes'],
                            inner_steps=inner_steps,
                            num_inner_queries=num_inner_queries,
                            rng=rng,
                            load_rgb=use_rgb,
                            load_mask_id=use_mask_id,
                        )
                        if inner_steps > 0:
                            mask = make_name_mask(
                                params,
                                include=tuple(getattr(maml_cfg, 'inner_param_include', ())),
                                exclude=tuple(getattr(maml_cfg, 'inner_param_exclude', ())),
                            )
                            adapt_fn = _make_param_adapt_fn(
                                model,
                                inner_lr=inner_lr,
                                grad_clip_norm=grad_clip,
                                loss_type=str(getattr(ckpt_cfg.train, 'loss_type', 'mse')),
                                inner_mask=mask,
                            )
                            adapted_params, metrics = adapt_fn(jax.device_put(params), inner_batch)
                            adapt_metrics = {k: float(jax.device_get(v)) for k, v in metrics.items()}
                        logging.info('Adapted params | support=%s | inner_steps=%d | inner_lr=%.6g | metrics=%s', support['meta']['support_episodes'], inner_steps, inner_lr, adapt_metrics)
                    else:
                        logging.info('Built support conditioning | support=%s', support['meta']['support_episodes'])
                    policy.update_params(adapted_params)
                    policy.update_support(support)

            result = _run_episode(
                episode_index=episode,
                task_env=task_env,
                variation=variation,
                policy=policy,
                processor=processor,
                data_cfg=data_cfg,
                query_stride_mode=query_mode,
                cfg=cfg,
                run_dir=run_dir,
            )
            if support is not None:
                result['support'] = support['meta']
            if adapt_metrics is not None:
                result['adaptation'] = adapt_metrics
            results.append(result)
            logging.info(
                'Episode %d | success=%s | steps=%d%s',
                episode,
                result['success'],
                result['env_steps'],
                f" | error={result['error']}" if result['error'] else '',
            )

        successes = sum(1 for r in results if r['success'])
        success_rate = float(successes) / float(max(1, len(results)))
        summary = {
            'task': task_name,
            'variation': variation,
            'checkpoint_path': str(checkpoint_path),
            'checkpoint_step': int(ckpt.get('step', -1)),
            'adaptation': adaptation,
            'num_episodes': len(results),
            'num_success': int(successes),
            'success_rate': success_rate,
            'results': results,
        }
        summary_path = run_dir / 'summary.json'
        with summary_path.open('w', encoding='utf-8') as file:
            json.dump(summary, file, indent=2)
        if wandb_run is not None:
            wandb_run.log({'eval/success_rate': success_rate, 'eval/num_success': successes, 'eval/num_episodes': len(results)})
            wandb_run.save(str(summary_path), policy='now')
        logging.info('Evaluation complete | success=%d/%d (%.3f) | outputs=%s', successes, len(results), success_rate, run_dir)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        store.close()
        if env is not None:
            env.shutdown()
