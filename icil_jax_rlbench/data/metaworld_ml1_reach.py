from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from phi_mujoco.integrations.metaworld_ml1_reach import (
    ML1ReachTaskIndex,
    TaskSplit,
    load_ml1_reach_task_index,
)
from phi_mujoco.offline import (
    EpisodeSplit,
    SplitConfig,
    StandardNormalization,
    ValidatedCollection,
    fit_standard_normalization,
    validate_collection_bundle,
)

SPLITS: tuple[TaskSplit, ...] = ('train', 'validation', 'test')


def _episode_indices(
    task_index: ML1ReachTaskIndex,
    split: TaskSplit,
) -> tuple[int, ...]:
    return tuple(
        episode_index
        for task_id in task_index.task_ids(split)
        for episode_index in task_index.episode_indices(task_id)
    )


def build_task_episode_split(
    bundle: ValidatedCollection,
    task_index: ML1ReachTaskIndex,
) -> EpisodeSplit:
    """Translate the declared task split into phi-mujoco episode indices."""

    if bundle.episode_count != len(task_index.episodes):
        raise ValueError('Task metadata and processed bundle episode counts differ.')
    if any(not episode.success for episode in bundle.episodes):
        raise ValueError('ML1 Reach imitation caches must contain successful episodes only.')
    train = _episode_indices(task_index, 'train')
    validation = _episode_indices(task_index, 'validation')
    test = _episode_indices(task_index, 'test')
    partitions = (set(train), set(validation), set(test))
    if any(left & right for index, left in enumerate(partitions) for right in partitions[index + 1 :]):
        raise ValueError('ML1 Reach task splits contain overlapping episodes.')
    eligible = tuple(sorted(train + validation + test))
    expected = tuple(range(bundle.episode_count))
    if eligible != expected:
        raise ValueError('ML1 Reach task metadata does not cover every cache episode exactly once.')
    return EpisodeSplit(
        config=SplitConfig(),
        successful_only=True,
        eligible_episode_indices=eligible,
        train_episode_indices=tuple(sorted(train)),
        val_episode_indices=tuple(sorted(validation)),
        test_episode_indices=tuple(sorted(test)),
    )


def normalization_identifier(
    normalization: StandardNormalization,
    *,
    data_sha256: str,
) -> str:
    payload = {
        'data_sha256': str(data_sha256),
        'normalization': normalization.to_dict(),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
    ).hexdigest()
    return f'ml1_reach_train_{digest[:12]}'


class ML1ReachTaskDataset:
    """Task-aware, simulator-free policy view of one processed ML1 cache."""

    def __init__(
        self,
        cache_root: str | Path,
        *,
        normalization: StandardNormalization | Mapping[str, object] | None = None,
        normalization_eps: float = 1e-4,
        cache_prepared_episodes: bool = True,
    ) -> None:
        self.bundle = validate_collection_bundle(
            Path(cache_root).expanduser().resolve(),
            integration='metaworld_ml1_reach',
        )
        self.task_index = load_ml1_reach_task_index(self.bundle)
        self.episode_split = build_task_episode_split(self.bundle, self.task_index)
        if self.bundle.spec.observations['state'].shape != (39,):
            raise ValueError('ML1 Reach policy observations must be 39-dimensional.')
        if self.bundle.spec.action.shape != (4,):
            raise ValueError('ML1 Reach policy actions must be four-dimensional.')

        if isinstance(normalization, Mapping):
            resolved_normalization = StandardNormalization.from_dict(normalization)
        elif normalization is None:
            resolved_normalization = fit_standard_normalization(
                self.bundle,
                self.episode_split,
                observation_modality='state',
                eps=float(normalization_eps),
            )
        elif isinstance(normalization, StandardNormalization):
            resolved_normalization = normalization
        else:
            raise TypeError(
                'normalization must be StandardNormalization, a mapping, or None.'
            )
        if (
            resolved_normalization.observation_modality != 'state'
            or resolved_normalization.source_episode_indices
            != self.episode_split.train_episode_indices
        ):
            raise ValueError(
                'Normalization must be fitted exclusively from declared training-task episodes.'
            )
        if len(resolved_normalization.obs_mean) != 39:
            raise ValueError('Observation normalization has the wrong dimension.')
        if len(resolved_normalization.action_mean) != 4:
            raise ValueError('Action normalization has the wrong dimension.')

        self.normalization = resolved_normalization
        self.normalization_id = normalization_identifier(
            resolved_normalization,
            data_sha256=self.bundle.data_sha256,
        )
        self.horizon = max(episode.steps for episode in self.bundle.episodes)
        self._cache_prepared_episodes = bool(cache_prepared_episodes)
        self._prepared_episodes: dict[int, dict[str, np.ndarray]] = {}

    @property
    def observation_dim(self) -> int:
        return 39

    @property
    def action_dim(self) -> int:
        return 4

    def task_ids(self, split: TaskSplit) -> tuple[str, ...]:
        if split not in SPLITS:
            raise ValueError(f'split must be one of {SPLITS}, got {split!r}.')
        return self.task_index.task_ids(split)

    def _prepare_episode(self, episode_index: int) -> dict[str, np.ndarray]:
        episode_index = int(episode_index)
        cached = self._prepared_episodes.get(episode_index)
        if cached is not None:
            return cached
        episode = self.bundle.load_episode(episode_index)
        steps = int(episode.actions.shape[0])
        observations = self.normalization.normalize_observations(
            episode.observations['state']
        )
        actions = self.normalization.normalize_actions(episode.actions)
        prepared = {
            'observation': np.zeros(
                (self.horizon, self.observation_dim), dtype=np.float32
            ),
            'action': np.zeros((self.horizon, self.action_dim), dtype=np.float32),
            'next_observation': np.zeros(
                (self.horizon, self.observation_dim), dtype=np.float32
            ),
            'valid': np.zeros((self.horizon,), dtype=np.bool_),
            'episode_id': np.asarray(episode_index, dtype=np.int32),
            'episode_seed': np.asarray(episode.seed, dtype=np.uint32),
        }
        prepared['observation'][:steps] = observations[:-1]
        prepared['action'][:steps] = actions
        prepared['next_observation'][:steps] = observations[1:]
        prepared['valid'][:steps] = True
        if self._cache_prepared_episodes:
            self._prepared_episodes[episode_index] = prepared
        return prepared

    def _stack_episodes(self, episode_indices: Sequence[int]) -> dict[str, np.ndarray]:
        episodes = [self._prepare_episode(index) for index in episode_indices]
        return {
            name: np.stack([episode[name] for episode in episodes], axis=0)
            for name in episodes[0]
        }

    def section(self, episode_indices: Sequence[int], *, support: bool) -> dict[str, np.ndarray]:
        if not episode_indices:
            raise ValueError('An episode section must contain at least one episode.')
        section = self._stack_episodes(episode_indices)
        if support:
            section['write_mask'] = np.array(section['valid'], copy=True)
            section['outer_loss_mask'] = np.zeros_like(section['valid'])
        else:
            section['outer_loss_mask'] = np.array(section['valid'], copy=True)
        return section

    def provenance(self) -> dict[str, object]:
        return {
            'name': 'metaworld_ml1_reach',
            'cache_root': str(self.bundle.root),
            'cache_data_sha256': self.bundle.data_sha256,
            'task_catalog_seed': self.task_index.catalog.catalog_seed,
            'task_split_sizes': {
                split: len(self.task_ids(split)) for split in SPLITS
            },
            'episodes_per_task': self.task_index.episodes_per_task,
            'maximum_episode_steps': self.horizon,
            'normalizer_id': self.normalization_id,
            'normalizer_fit_split': 'train_tasks_only',
            'support_query_separate_episodes': True,
            'goal_provided_to_policy': False,
        }

    def integrity_report(self) -> dict[str, object]:
        train_indices = set(self.episode_split.train_episode_indices)
        normalization_indices = set(self.normalization.source_episode_indices)
        return {
            **self.provenance(),
            'episode_count': self.bundle.episode_count,
            'expert_success_rate': (
                self.bundle.successful_episodes / self.bundle.episode_count
            ),
            'unique_episode_seeds': (
                len({episode.seed for episode in self.bundle.episodes})
                == self.bundle.episode_count
            ),
            'normalizer_uses_exact_training_task_episodes': (
                normalization_indices == train_indices
            ),
            'observation_mean': list(self.normalization.obs_mean),
            'observation_std': list(self.normalization.obs_std),
            'action_mean': list(self.normalization.action_mean),
            'action_std': list(self.normalization.action_std),
        }


class ML1ReachTaskSampler:
    """Sample explicit task, demonstration, and time axes from a fixed cache."""

    def __init__(
        self,
        dataset: ML1ReachTaskDataset,
        *,
        split: TaskSplit,
        seed: int,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f'split must be one of {SPLITS}, got {split!r}.')
        self.dataset = dataset
        self.split = split
        self.task_ids = dataset.task_ids(split)
        self.rng = np.random.default_rng(int(seed))

    def _resolve_tasks(
        self,
        batch_size: int,
        task_ids: Sequence[str] | None,
    ) -> tuple[str, ...]:
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError('batch_size must be positive.')
        if task_ids is None:
            choices = self.rng.integers(0, len(self.task_ids), size=(batch_size,))
            return tuple(self.task_ids[int(index)] for index in choices)
        resolved = tuple(str(task_id) for task_id in task_ids)
        if len(resolved) != batch_size:
            raise ValueError(
                f'task_ids must contain {batch_size} values, got {len(resolved)}.'
            )
        invalid = set(resolved) - set(self.task_ids)
        if invalid:
            raise ValueError(
                f'Tasks do not belong to split {self.split!r}: {sorted(invalid)}.'
            )
        return resolved

    def _choose_episodes(self, task_id: str, count: int) -> tuple[int, ...]:
        available = self.dataset.task_index.episode_indices(task_id)
        if int(count) > len(available):
            raise ValueError(
                f'Task {task_id!r} has {len(available)} episodes, fewer than {count} requested.'
            )
        selected = self.rng.choice(available, size=int(count), replace=False)
        return tuple(int(index) for index in selected)

    @staticmethod
    def _stack_tasks(
        tasks: Sequence[Mapping[str, np.ndarray]],
    ) -> dict[str, np.ndarray]:
        return {
            name: np.stack([task[name] for task in tasks], axis=0)
            for name in tasks[0]
        }

    def build_batch(
        self,
        batch_size: int,
        *,
        support_episodes: int,
        query_episodes: int,
        task_ids: Sequence[str] | None = None,
    ) -> dict[str, dict[str, np.ndarray]]:
        support_episodes = int(support_episodes)
        query_episodes = int(query_episodes)
        if support_episodes < 1 or query_episodes < 1:
            raise ValueError('support_episodes and query_episodes must be positive.')
        selected_tasks = self._resolve_tasks(batch_size, task_ids)
        support_tasks = []
        query_tasks = []
        for task_id in selected_tasks:
            episode_indices = self._choose_episodes(
                task_id, support_episodes + query_episodes
            )
            support_tasks.append(
                self.dataset.section(
                    episode_indices[:support_episodes], support=True
                )
            )
            query_tasks.append(
                self.dataset.section(
                    episode_indices[support_episodes:], support=False
                )
            )
        return {
            'support': self._stack_tasks(support_tasks),
            'query': self._stack_tasks(query_tasks),
        }

    def build_query_batch(
        self,
        batch_size: int,
        *,
        query_episodes: int,
        task_ids: Sequence[str] | None = None,
    ) -> dict[str, dict[str, np.ndarray]]:
        query_episodes = int(query_episodes)
        if query_episodes < 1:
            raise ValueError('query_episodes must be positive.')
        selected_tasks = self._resolve_tasks(batch_size, task_ids)
        query_tasks = [
            self.dataset.section(
                self._choose_episodes(task_id, query_episodes), support=False
            )
            for task_id in selected_tasks
        ]
        return {'query': self._stack_tasks(query_tasks)}


__all__ = [
    'SPLITS',
    'ML1ReachTaskDataset',
    'ML1ReachTaskSampler',
    'build_task_episode_split',
    'normalization_identifier',
]
