from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .action_representation import encode_action_chunk, encode_support_traj, normalize_action_representation
from .h5_cache import RLBenchCacheStore


@dataclass(frozen=True)
class ICILDataConfig:
    K: int = 4
    L: int = 16
    T_obs: int = 2
    H: int = 16
    stride: int = 2
    action_representation: str = 'absolute'
    task_sampling: str = 'variation_uniform'
    task_sampling_alpha: float = 1.0
    traj_len: int = 64

    def __post_init__(self):
        if self.K < 1 or self.L < 1 or self.T_obs < 1 or self.H < 1 or self.stride < 1:
            raise ValueError('K, L, T_obs, H, and stride must be positive.')
        normalize_action_representation(self.action_representation)
        if self.task_sampling not in ('variation_uniform', 'task_uniform', 'variation_power'):
            raise ValueError('task_sampling must be variation_uniform, task_uniform, or variation_power.')


class ICILSampler:
    def __init__(self, store: RLBenchCacheStore, cfg: ICILDataConfig, *, seed: int = 0, num_tries_per_item: int = 100):
        self.store = store
        self.cfg = cfg
        self.rng = np.random.default_rng(int(seed))
        self.num_tries_per_item = int(num_tries_per_item)
        self._task_names: Optional[List[str]] = None
        self._vidx_by_task: Optional[List[np.ndarray]] = None
        self._task_probs: Optional[np.ndarray] = None

    def _build_task_sampling_index(self) -> bool:
        if self._task_probs is not None:
            return True
        by_task = self.store.task_sampling_index()
        if not by_task:
            return False
        names = sorted(by_task)
        vidxs = [np.asarray(by_task[name], dtype=np.int64) for name in names]
        counts = np.asarray([len(x) for x in vidxs], dtype=np.float64)
        alpha = 0.0 if self.cfg.task_sampling == 'task_uniform' else float(self.cfg.task_sampling_alpha)
        weights = np.power(counts, alpha)
        weights = weights / max(float(weights.sum()), 1e-9)
        self._task_names = names
        self._vidx_by_task = vidxs
        self._task_probs = weights
        return True

    def _sample_vidx(self, min_episodes: int) -> int:
        use_task_sampling = self.cfg.task_sampling != 'variation_uniform' and self._build_task_sampling_index()
        for _ in range(self.num_tries_per_item):
            if use_task_sampling:
                assert self._task_probs is not None and self._vidx_by_task is not None
                tidx = int(self.rng.choice(len(self._task_probs), p=self._task_probs))
                choices = self._vidx_by_task[tidx]
                vidx = int(choices[int(self.rng.integers(0, len(choices)))])
            else:
                vidx = int(self.rng.integers(0, len(self.store)))
            if len(self.store.list_episode_ids(vidx)) >= int(min_episodes):
                return vidx
        raise RuntimeError(f'Could not sample variation with at least {min_episodes} episodes.')

    def _sample_episode_ids(self, vidx: int, count: int) -> np.ndarray:
        eids = self.store.list_episode_ids(vidx)
        if len(eids) < count:
            raise RuntimeError(f'variation {vidx} has {len(eids)} episodes, need {count}.')
        return self.rng.choice(eids, size=int(count), replace=False).astype(np.int64)

    def _sample_keyframes(self, T: int, L: int) -> np.ndarray:
        if T >= L:
            return np.sort(self.rng.choice(T, size=L, replace=False)).astype(np.int64)
        return np.sort(self.rng.choice(T, size=L, replace=True)).astype(np.int64)

    def _sample_t0(self, T: int) -> int:
        required = 1 + ((self.cfg.T_obs - 1) * self.cfg.stride)
        max_t0 = int(T) - required
        if max_t0 < 0:
            raise RuntimeError(f'Episode too short: T={T}, required={required}.')
        return int(self.rng.integers(0, max_t0 + 1))

    def _obs_act_indices(self, t0: int, T: int) -> Tuple[np.ndarray, np.ndarray]:
        obs = int(t0) + np.arange(0, self.cfg.T_obs * self.cfg.stride, self.cfg.stride, dtype=np.int64)
        act_start = int(obs[-1] + self.cfg.stride)
        act = act_start + np.arange(0, self.cfg.H * self.cfg.stride, self.cfg.stride, dtype=np.int64)
        act = np.minimum(act, int(T) - 1)
        return obs, act

    def _traj_indices(self, T: int) -> Tuple[np.ndarray, np.ndarray]:
        M = int(self.cfg.traj_len)
        if M <= 0:
            return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.bool_)
        raw = np.arange(0, int(T), max(1, int(self.cfg.stride)), dtype=np.int64)
        if len(raw) >= M:
            idx = raw[:M]
            mask = np.ones((M,), dtype=np.bool_)
        else:
            pad = np.full((M - len(raw),), int(T) - 1, dtype=np.int64)
            idx = np.concatenate([raw, pad], axis=0)
            mask = np.zeros((M,), dtype=np.bool_)
            mask[: len(raw)] = True
        return idx, mask

    def build_support_conditioning(self, *, vidx: int, support_ids: Sequence[int], load_rgb: bool, load_mask_id: bool) -> Dict[str, np.ndarray]:
        cond_xyz: List[np.ndarray] = []
        cond_state: List[np.ndarray] = []
        cond_valid: List[np.ndarray] = []
        cond_rgb: List[np.ndarray] = []
        cond_mask: List[np.ndarray] = []
        cond_traj: List[np.ndarray] = []
        cond_traj_mask: List[np.ndarray] = []
        has_rgb = bool(load_rgb)
        has_mask = bool(load_mask_id)
        has_traj = self.cfg.traj_len > 0
        for eid in support_ids:
            T = self.store.episode_length(vidx, int(eid))
            kf = self._sample_keyframes(T, self.cfg.L)
            item = self.store.load_episode_slices(vidx, int(eid), kf, load_rgb=load_rgb, load_mask_id=load_mask_id)
            cond_xyz.append(item['xyz'])
            cond_state.append(item['state'])
            cond_valid.append(item['valid'])
            if load_rgb and 'rgb' in item:
                cond_rgb.append(item['rgb'])
            else:
                has_rgb = False
            if load_mask_id and 'mask_id' in item:
                cond_mask.append(item['mask_id'])
            else:
                has_mask = False
            if self.cfg.traj_len > 0:
                tidx, tmask = self._traj_indices(T)
                traj = self.store.load_episode_slices(vidx, int(eid), tidx, load_rgb=False, load_mask_id=False)['action']
                cond_traj.append(encode_support_traj(traj, representation=self.cfg.action_representation))
                cond_traj_mask.append(tmask)
        out: Dict[str, np.ndarray] = {
            'cond_xyz': np.stack(cond_xyz, axis=0).astype(np.float32),
            'cond_state': np.stack(cond_state, axis=0).astype(np.float32),
            'cond_valid': np.stack(cond_valid, axis=0).astype(np.bool_),
        }
        if has_rgb:
            out['cond_rgb'] = np.stack(cond_rgb, axis=0).astype(np.float32)
        if has_mask:
            out['cond_mask_id'] = np.stack(cond_mask, axis=0).astype(np.int32)
        if has_traj:
            out['cond_traj'] = np.stack(cond_traj, axis=0).astype(np.float32)
            out['cond_traj_mask'] = np.stack(cond_traj_mask, axis=0).astype(np.bool_)
        return out

    def build_query_sample(self, *, vidx: int, episode_id: int, load_rgb: bool, load_mask_id: bool) -> Dict[str, np.ndarray]:
        T = self.store.episode_length(vidx, int(episode_id))
        t0 = self._sample_t0(T)
        obs_idx, act_idx = self._obs_act_indices(t0, T)
        obs = self.store.load_episode_slices(vidx, int(episode_id), obs_idx, load_rgb=load_rgb, load_mask_id=load_mask_id)
        act = self.store.load_episode_slices(vidx, int(episode_id), act_idx, load_rgb=False, load_mask_id=False)
        out: Dict[str, np.ndarray] = {
            'query_xyz': obs['xyz'].astype(np.float32),
            'query_state': obs['state'].astype(np.float32),
            'query_valid': obs['valid'].astype(np.bool_),
            'target_action': encode_action_chunk(act['action'], query_state=obs['state'], representation=self.cfg.action_representation).astype(np.float32),
            'chunk_start': np.asarray(float(t0), dtype=np.float32),
        }
        if load_rgb and 'rgb' in obs:
            out['query_rgb'] = obs['rgb'].astype(np.float32)
        if load_mask_id and 'mask_id' in obs:
            out['query_mask_id'] = obs['mask_id'].astype(np.int32)
        return out

    @staticmethod
    def _stack(samples: Sequence[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        keys = set.intersection(*(set(s.keys()) for s in samples))
        return {k: np.stack([s[k] for s in samples], axis=0) for k in sorted(keys)}

    @staticmethod
    def _merge(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        out = dict(a)
        out.update(b)
        return out

    def build_pretrain_batch(self, batch_size: int, *, load_rgb: bool = True, load_mask_id: bool = False) -> Dict[str, np.ndarray]:
        samples: List[Dict[str, np.ndarray]] = []
        for _ in range(int(batch_size)):
            for _try in range(self.num_tries_per_item):
                try:
                    vidx = self._sample_vidx(self.cfg.K + 1)
                    eids = self._sample_episode_ids(vidx, self.cfg.K + 1)
                    support = self.build_support_conditioning(vidx=vidx, support_ids=eids[: self.cfg.K], load_rgb=load_rgb, load_mask_id=load_mask_id)
                    query = self.build_query_sample(vidx=vidx, episode_id=int(eids[self.cfg.K]), load_rgb=load_rgb, load_mask_id=load_mask_id)
                    samples.append(self._merge(support, query))
                    break
                except RuntimeError:
                    if _try == self.num_tries_per_item - 1:
                        raise
        return self._stack(samples)

    def _build_context_query_sample(
        self,
        *,
        vidx: int,
        context_ids: Sequence[int],
        query_episode_id: int,
        load_rgb: bool,
        load_mask_id: bool,
    ) -> Dict[str, np.ndarray]:
        support = self.build_support_conditioning(vidx=vidx, support_ids=context_ids, load_rgb=load_rgb, load_mask_id=load_mask_id)
        query = self.build_query_sample(vidx=vidx, episode_id=int(query_episode_id), load_rgb=load_rgb, load_mask_id=load_mask_id)
        return self._merge(support, query)

    def build_param_maml_batch(
        self,
        batch_size: int,
        *,
        inner_steps: int,
        num_inner_queries: int,
        num_query_loss_samples: int,
        outer_context_size: Optional[int] = None,
        load_rgb: bool = True,
        load_mask_id: bool = False,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        B = int(batch_size)
        S = int(inner_steps)
        Q = int(num_inner_queries)
        O = int(num_query_loss_samples)
        inner_tasks: List[Dict[str, np.ndarray]] = []
        query_tasks: List[Dict[str, np.ndarray]] = []
        meta_vidx: List[int] = []
        for _ in range(B):
            vidx = self._sample_vidx(self.cfg.K + 1)
            eids = self._sample_episode_ids(vidx, self.cfg.K + 1)
            support_ids = [int(x) for x in eids[: self.cfg.K]]
            query_id = int(eids[self.cfg.K])
            inner_steps_batches: List[Dict[str, np.ndarray]] = []
            for _s in range(S):
                samples = []
                order = list(self.rng.permutation(len(support_ids)))
                while len(order) < Q:
                    order.extend(list(self.rng.permutation(len(support_ids))))
                for holdout_idx in order[:Q]:
                    heldout = support_ids[int(holdout_idx)]
                    context = [eid for j, eid in enumerate(support_ids) if j != int(holdout_idx)]
                    samples.append(self._build_context_query_sample(vidx=vidx, context_ids=context, query_episode_id=heldout, load_rgb=load_rgb, load_mask_id=load_mask_id))
                inner_steps_batches.append(self._stack(samples))
            if S > 0:
                inner_tasks.append({k: np.stack([step[k] for step in inner_steps_batches], axis=0) for k in inner_steps_batches[0]})
            else:
                inner_tasks.append({})
            ctx_n = self.cfg.K if outer_context_size is None else min(int(outer_context_size), self.cfg.K)
            q_samples = [self._build_context_query_sample(vidx=vidx, context_ids=support_ids[:ctx_n], query_episode_id=query_id, load_rgb=load_rgb, load_mask_id=load_mask_id) for _ in range(O)]
            query_tasks.append(self._stack(q_samples))
            meta_vidx.append(int(vidx))
        inner = {k: np.stack([t[k] for t in inner_tasks], axis=0) for k in inner_tasks[0]} if S > 0 else {}
        query = {k: np.stack([t[k] for t in query_tasks], axis=0) for k in query_tasks[0]}
        return {'inner': inner, 'query': query, 'meta': {'vidx': np.asarray(meta_vidx, dtype=np.int32)}}

    def build_memory_maml_batch(
        self,
        batch_size: int,
        *,
        inner_steps: int,
        num_inner_queries: int,
        num_query_loss_samples: int,
        holdout_index: int = -1,
        load_rgb: bool = True,
        load_mask_id: bool = False,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        B = int(batch_size)
        S = int(inner_steps)
        Q = int(num_inner_queries)
        O = int(num_query_loss_samples)
        mem_inits: List[Dict[str, np.ndarray]] = []
        inner_tasks: List[Dict[str, np.ndarray]] = []
        query_tasks: List[Dict[str, np.ndarray]] = []
        meta_vidx: List[int] = []
        for _ in range(B):
            vidx = self._sample_vidx(self.cfg.K + 1)
            eids = self._sample_episode_ids(vidx, self.cfg.K + 1)
            support_ids = [int(x) for x in eids[: self.cfg.K]]
            query_id = int(eids[self.cfg.K])
            hidx = int(holdout_index) if int(holdout_index) >= 0 else int(self.rng.integers(0, len(support_ids)))
            heldout = support_ids[hidx]
            memory_support_ids = [eid for j, eid in enumerate(support_ids) if j != hidx]
            mem_inits.append(self.build_support_conditioning(vidx=vidx, support_ids=memory_support_ids, load_rgb=load_rgb, load_mask_id=load_mask_id))
            inner_steps_batches: List[Dict[str, np.ndarray]] = []
            for _s in range(S):
                samples = [self.build_query_sample(vidx=vidx, episode_id=heldout, load_rgb=load_rgb, load_mask_id=load_mask_id) for _ in range(Q)]
                inner_steps_batches.append(self._stack(samples))
            if S > 0:
                inner_tasks.append({k: np.stack([step[k] for step in inner_steps_batches], axis=0) for k in inner_steps_batches[0]})
            else:
                inner_tasks.append({})
            q_samples = [self.build_query_sample(vidx=vidx, episode_id=query_id, load_rgb=load_rgb, load_mask_id=load_mask_id) for _ in range(O)]
            query_tasks.append(self._stack(q_samples))
            meta_vidx.append(int(vidx))
        memory_init = {k: np.stack([t[k] for t in mem_inits], axis=0) for k in mem_inits[0]}
        inner = {k: np.stack([t[k] for t in inner_tasks], axis=0) for k in inner_tasks[0]} if S > 0 else {}
        query = {k: np.stack([t[k] for t in query_tasks], axis=0) for k in query_tasks[0]}
        return {'memory_init': memory_init, 'inner': inner, 'query': query, 'meta': {'vidx': np.asarray(meta_vidx, dtype=np.int32)}}


def shard_batch(batch: Any, num_devices: int) -> Any:
    def reshape(x):
        if not hasattr(x, 'shape') or len(x.shape) == 0:
            return x
        if x.shape[0] % int(num_devices) != 0:
            return x
        return x.reshape((int(num_devices), x.shape[0] // int(num_devices)) + x.shape[1:])
    if isinstance(batch, dict):
        return {k: shard_batch(v, num_devices) for k, v in batch.items()}
    return reshape(batch)
