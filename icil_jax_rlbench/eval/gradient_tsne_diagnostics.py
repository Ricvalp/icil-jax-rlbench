#!/usr/bin/env python3
"""Collect first-inner-step MAML gradients and visualize them with t-SNE.

This is an offline diagnostic script. It loads a package checkpoint, samples
balanced support sets from the RLBench cache, computes the first parameter-MAML
inner-loop gradient for each support set, projects each gradient to a compact
feature vector, and optionally writes a t-SNE plot.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax import traverse_util
from ml_collections import ConfigDict

from icil_jax_rlbench.data.h5_cache import RLBenchCacheStore, build_keys
from icil_jax_rlbench.data.sampler import ICILDataConfig, ICILSampler
from icil_jax_rlbench.models.direct_regression_policy import DirectRegressionPolicy
from icil_jax_rlbench.models.config import policy_config_from
from icil_jax_rlbench.train.checkpoints import load_checkpoint
from icil_jax_rlbench.train.step import action_loss, make_maml_inner_mask


class _ProgressBar:
    def __init__(self, total: int, *, enabled: bool = True):
        self.total = max(1, int(total))
        self.count = 0
        self.enabled = bool(enabled)
        self._bar = None
        self._last_plain_percent = -1
        if not self.enabled:
            return
        try:
            from tqdm.auto import tqdm

            self._bar = tqdm(total=self.total, desc="Gradients", unit="grad", dynamic_ncols=True)
        except Exception:
            self._bar = None

    def update(self, n: int = 1, *, label: str = "") -> None:
        self.count = min(self.total, self.count + int(n))
        if not self.enabled:
            return
        if self._bar is not None:
            if label:
                self._bar.set_postfix_str(label[:80])
            self._bar.update(int(n))
            return
        percent = int(round(100.0 * self.count / self.total))
        filled = int(round(30.0 * self.count / self.total))
        bar = "#" * filled + "-" * (30 - filled)
        if sys.stderr.isatty():
            sys.stderr.write(f"\rGradients [{bar}] {self.count}/{self.total} ({percent:3d}%) {label[:60]}")
            sys.stderr.flush()
            return
        if percent != self._last_plain_percent and (percent % 5 == 0 or self.count == self.total):
            self._last_plain_percent = percent
            print(
                f"Gradients [{bar}] {self.count}/{self.total} ({percent:3d}%) {label[:60]}",
                file=sys.stderr,
                flush=True,
            )

    def close(self) -> None:
        if not self.enabled:
            return
        if self._bar is not None:
            self._bar.close()
            return
        if sys.stderr.isatty():
            sys.stderr.write("\n")
            sys.stderr.flush()


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    return tuple(str(v) for v in value)


def _parse_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _load_mask_id_for_batch(policy_cfg: Any, data_cfg: ICILDataConfig) -> bool:
    encoder = policy_cfg.encoder
    return bool(
        encoder.use_mask_id
        or (
            encoder.encoder_type == "supernode"
            and getattr(encoder, "supernode_center_sampling", "fps") == "mask_balanced"
        )
        or (
            getattr(encoder, "support_tokenizer", "") == "spacetime_supernode"
            and int(data_cfg.support_spacetime_points) > 0
            and str(data_cfg.support_spacetime_sampling) == "mask_balanced"
        )
    )


def _data_config_from_checkpoint(ckpt_cfg: ConfigDict) -> ICILDataConfig:
    data = ckpt_cfg.data
    return ICILDataConfig(
        K=int(data.K),
        L=int(data.L),
        T_obs=int(data.T_obs),
        H=int(data.H),
        stride=int(data.get("stride", 1)),
        action_representation=str(data.get("action_representation", "absolute")),
        traj_len=int(data.get("traj_len", data.H)),
        task_sampling=str(data.get("task_sampling", "variation_uniform")),
        task_sampling_alpha=float(data.get("task_sampling_alpha", 1.0)),
        query_window_mode=str(data.get("query_window_mode", "online_history")),
        support_spacetime_points=int(data.get("support_spacetime_points", 0)),
        support_spacetime_sampling=str(data.get("support_spacetime_sampling", "mask_balanced")),
    )


def _resolve_cache_root(args: argparse.Namespace, ckpt_cfg: ConfigDict) -> str:
    root = args.cache_root or os.environ.get("ICIL_CACHE_ROOT") or ckpt_cfg.data.get("cache_root", "")
    if not root:
        raise ValueError(
            "No cache root provided. Use --cache-root or set ICIL_CACHE_ROOT."
        )
    return str(root)


def _make_output_dir(args: argparse.Namespace, checkpoint: Path, step: int) -> Path:
    run_name = checkpoint.parent.name
    step_name = checkpoint.stem
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir) / f"{run_name}_{step_name}_grad_tsne_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def _group_variations(
    keys: Sequence[Any],
    class_level: str,
) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for vidx, key in enumerate(keys):
        if class_level == "task_variation":
            label = f"{key.task}:variation{key.variation}"
        else:
            label = key.task
        groups[label].append(vidx)
    return dict(groups)


def _sample_support_ids(
    store: RLBenchCacheStore,
    vidx: int,
    K: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    episode_ids = np.asarray(store.list_episode_ids(vidx), dtype=np.int32)
    if episode_ids.shape[0] < K:
        return None
    return rng.choice(episode_ids, size=K, replace=False).astype(np.int32)


def _build_first_inner_batch(
    sampler: ICILSampler,
    vidx: int,
    support_ids: np.ndarray,
    num_inner_queries: int,
    load_rgb: bool,
    load_mask_id: bool,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    K = int(support_ids.shape[0])
    Q = max(1, int(num_inner_queries))
    order = list(rng.permutation(K))
    while len(order) < Q:
        order.extend(list(rng.permutation(K)))

    samples = []
    heldout_ids = []
    for holdout_idx in order[:Q]:
        heldout_episode = int(support_ids[holdout_idx])
        context_ids = [int(eid) for i, eid in enumerate(support_ids) if i != holdout_idx]
        samples.append(
            sampler._build_context_query_sample(
                vidx=vidx,
                context_ids=context_ids,
                query_episode_id=heldout_episode,
                load_rgb=load_rgb,
                load_mask_id=load_mask_id,
            )
        )
        heldout_ids.append(heldout_episode)
    return sampler._stack(samples), np.asarray(heldout_ids, dtype=np.int32)


def _make_grad_fn(
    model: DirectRegressionPolicy,
    loss_type: str,
):
    def loss_fn(params: Mapping[str, Any], batch: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
        pred = model.apply({"params": params}, batch, train=False)
        return action_loss(pred, batch["target_action"], loss_type)

    return jax.jit(jax.value_and_grad(loss_fn))


def _selected_param_paths(
    params: Mapping[str, Any],
    mask: Mapping[str, Any],
) -> tuple[list[str], int, dict[str, int]]:
    flat_params = traverse_util.flatten_dict(params, sep="/")
    flat_mask = traverse_util.flatten_dict(mask, sep="/")
    paths: list[str] = []
    sizes: dict[str, int] = {}
    total = 0
    for path in sorted(flat_params):
        selected = bool(np.asarray(jax.device_get(flat_mask.get(path, False))))
        if not selected:
            continue
        size = int(np.asarray(flat_params[path]).size)
        paths.append(path)
        sizes[path] = size
        total += size
    return paths, total, sizes


def _make_projection_specs(
    selected_paths: Sequence[str],
    param_sizes: Mapping[str, int],
    projection_dim: int,
    seed: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    specs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for path in selected_paths:
        size = int(param_sizes[path])
        bins = rng.integers(0, projection_dim, size=size, dtype=np.int32)
        signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=size)
        specs[path] = (bins, signs.astype(np.float32, copy=False))
    return specs


def _project_gradients(
    grads: Mapping[str, Any],
    projection_specs: Mapping[str, tuple[np.ndarray, np.ndarray]],
    projection_dim: int,
) -> np.ndarray:
    flat_grads = traverse_util.flatten_dict(grads, sep="/")
    projected = np.zeros((projection_dim,), dtype=np.float32)
    for path, (bins, signs) in projection_specs.items():
        grad = np.asarray(jax.device_get(flat_grads[path]), dtype=np.float32).reshape(-1)
        projected += np.bincount(
            bins,
            weights=grad * signs,
            minlength=projection_dim,
        ).astype(np.float32, copy=False)
    return projected


def _compute_tsne(
    vectors: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, str | None]:
    if vectors.shape[0] < 3:
        return None, "Need at least three points for t-SNE."
    try:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
    except Exception as exc:  # pragma: no cover - optional dependency
        return None, f"scikit-learn is not available: {exc}"

    x = vectors.astype(np.float32, copy=False)
    pca_dim = min(int(args.pca_dims), x.shape[1], x.shape[0] - 1)
    if pca_dim > 0 and x.shape[1] > pca_dim:
        x = PCA(n_components=pca_dim, random_state=int(args.seed)).fit_transform(x)

    perplexity = min(float(args.perplexity), max(1.0, float(x.shape[0] - 1) / 3.0))
    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca" if x.shape[1] >= 2 else "random",
        random_state=int(args.seed),
        metric="euclidean",
    ).fit_transform(x)
    return embedding.astype(np.float32), None


def _write_plot(
    out_path: Path,
    embedding: np.ndarray,
    labels: np.ndarray,
    tasks: np.ndarray,
    variations: np.ndarray,
    losses: np.ndarray,
    grad_norms: np.ndarray,
) -> str | None:
    try:
        import plotly.express as px
    except Exception as exc:  # pragma: no cover - optional dependency
        return f"plotly is not available: {exc}"

    fig = px.scatter(
        {
            "x": embedding[:, 0],
            "y": embedding[:, 1],
            "label": labels,
            "task": tasks,
            "variation": variations,
            "loss": losses,
            "grad_norm": grad_norms,
        },
        x="x",
        y="y",
        color="label",
        hover_data=["task", "variation", "loss", "grad_norm"],
        render_mode="webgl",
        title="First inner-step gradient t-SNE",
    )
    fig.update_traces(marker={"size": 5, "opacity": 0.8})
    fig.update_layout(legend_title_text="Class")
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    return None


def _json_safe_counts(values: Sequence[str]) -> dict[str, int]:
    return {str(k): int(v) for k, v in Counter(values).items()}


def _write_summary(path: Path, summary: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect first parameter-MAML inner-loop gradients from RLBench "
            "support sets and write projected-gradient/t-SNE diagnostics."
        )
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a package .pkl checkpoint.")
    parser.add_argument(
        "--cache-root",
        default="",
        help="RLBench dense cache root. Defaults to ICIL_CACHE_ROOT, then checkpoint config.",
    )
    parser.add_argument(
        "--out-dir",
        default="eval_outputs/gradient_tsne_diagnostics",
        help="Directory where a timestamped diagnostic folder will be created.",
    )
    parser.add_argument(
        "--points-per-class",
        type=int,
        default=50,
        help="Number of sampled support sets per class.",
    )
    parser.add_argument(
        "--max-classes",
        type=int,
        default=100,
        help="Maximum number of classes to sample. Use 0 for all available classes.",
    )
    parser.add_argument(
        "--class-level",
        choices=("task", "task_variation"),
        default="task",
        help="Label gradients by task or by task+variation.",
    )
    parser.add_argument(
        "--tasks",
        default="",
        help="Comma-separated explicit task list. If set, checkpoint exclude_tasks are ignored.",
    )
    parser.add_argument(
        "--include-excluded",
        action="store_true",
        help="When --tasks is not set, include tasks listed in checkpoint data.exclude_tasks.",
    )
    parser.add_argument(
        "--preload-to-memory",
        action="store_true",
        help="Preload selected cache variations into host RAM. Default streams H5 files.",
    )
    parser.add_argument(
        "--keep-open",
        dest="keep_open",
        action="store_true",
        default=True,
        help="Keep H5 files open while streaming from disk.",
    )
    parser.add_argument(
        "--no-keep-open",
        dest="keep_open",
        action="store_false",
        help="Do not keep H5 files open between reads.",
    )
    parser.add_argument(
        "--num-inner-queries",
        type=int,
        default=0,
        help="Held-out support chunks per gradient. 0 means checkpoint maml.num_inner_queries.",
    )
    parser.add_argument(
        "--projection-dim",
        type=int,
        default=512,
        help="Random sign-hash projection dimension for each gradient.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling and t-SNE seed.")
    parser.add_argument(
        "--projection-seed",
        type=int,
        default=1009,
        help="Seed for the random gradient projection.",
    )
    parser.add_argument(
        "--max-tries-per-point",
        type=int,
        default=100,
        help="Sampling retries before skipping a point.",
    )
    parser.add_argument(
        "--normalize",
        dest="normalize",
        action="store_true",
        default=True,
        help="L2-normalize projected gradients before t-SNE.",
    )
    parser.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="Do not L2-normalize projected gradients.",
    )
    parser.add_argument(
        "--no-tsne",
        action="store_true",
        help="Only collect and save projected gradients; skip PCA/t-SNE and HTML plot.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the gradient collection progress bar.",
    )
    parser.add_argument("--pca-dims", type=int, default=50, help="PCA dimensions before t-SNE.")
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.points_per_class <= 0:
        raise ValueError("--points-per-class must be positive.")
    if args.projection_dim <= 0:
        raise ValueError("--projection-dim must be positive.")

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    ckpt = load_checkpoint(checkpoint_path)
    ckpt_cfg = ConfigDict(ckpt.get("config", {}) or {})
    step = int(ckpt.get("step", 0))
    mode = str(ckpt_cfg.get("mode", ""))
    if mode and mode != "param_maml":
        print(
            f"Warning: checkpoint mode is {mode!r}; this script computes param-MAML gradients.",
            flush=True,
        )

    cache_root = _resolve_cache_root(args, ckpt_cfg)
    data_cfg = _data_config_from_checkpoint(ckpt_cfg)
    requested_tasks = _parse_csv(args.tasks)
    tasks = requested_tasks or _as_tuple(ckpt_cfg.data.get("tasks", ()))
    exclude_from_ckpt = _as_tuple(ckpt_cfg.data.get("exclude_tasks", ()))
    exclude_tasks = () if (requested_tasks or args.include_excluded) else exclude_from_ckpt

    keys, _task_names = build_keys(
        cache_root,
        tasks=tasks,
        exclude_tasks=exclude_tasks,
    )
    if not keys:
        raise RuntimeError(f"No RLBench cache variations found under {cache_root}.")

    store = RLBenchCacheStore(
        keys,
        preload_to_memory=bool(args.preload_to_memory),
        keep_open=bool(args.keep_open),
    )
    _num_points, state_dim, action_dim = store.infer_dims()
    policy_cfg = policy_config_from(ckpt_cfg.model, H=data_cfg.H, data_cfg=data_cfg)

    model = DirectRegressionPolicy(policy_cfg, state_dim=int(state_dim), action_dim=int(action_dim))
    params = ckpt["params"]
    inner_mask = make_maml_inner_mask(
        params,
        preset=str(ckpt_cfg.maml.get("fast_param_preset", "")),
        include=_as_tuple(ckpt_cfg.maml.get("inner_param_include", ())),
        exclude=_as_tuple(ckpt_cfg.maml.get("inner_param_exclude", ())),
        decoder_layers=int(policy_cfg.decoder.n_layers),
        top_layers=int(ckpt_cfg.maml.get("fast_param_top_layers", 1)),
    )
    selected_paths, selected_param_count, param_sizes = _selected_param_paths(params, inner_mask)
    if not selected_paths:
        raise RuntimeError("The MAML fast-parameter mask selected no parameter tensors.")

    projection_specs = _make_projection_specs(
        selected_paths,
        param_sizes,
        projection_dim=int(args.projection_dim),
        seed=int(args.projection_seed),
    )
    sampler = ICILSampler(store, data_cfg, seed=int(args.seed))
    load_rgb = bool(policy_cfg.encoder.use_rgb)
    load_mask_id = _load_mask_id_for_batch(policy_cfg, data_cfg)
    num_inner_queries = int(args.num_inner_queries) or int(ckpt_cfg.maml.num_inner_queries)
    grad_fn = _make_grad_fn(model, str(ckpt_cfg.train.get("loss_type", "mse")))

    groups = _group_variations(keys, args.class_level)
    class_labels = sorted(groups)
    if args.max_classes > 0:
        class_labels = class_labels[: int(args.max_classes)]
    if not class_labels:
        raise RuntimeError("No classes available for sampling.")

    out_dir = _make_output_dir(args, checkpoint_path, step)
    print(f"Writing diagnostics to {out_dir}", flush=True)
    print(
        "Selected "
        f"{len(selected_paths)} parameter tensors "
        f"({selected_param_count:,} scalar parameters); "
        f"projecting to {args.projection_dim} dimensions.",
        flush=True,
    )
    print(
        f"Sampling {args.points_per_class} points per {args.class_level} "
        f"for {len(class_labels)} classes.",
        flush=True,
    )

    rng = np.random.default_rng(int(args.seed))
    vectors = []
    labels = []
    tasks_out = []
    variations_out = []
    vidxs_out = []
    support_ids_out = []
    heldout_ids_out = []
    losses = []
    grad_norms = []
    skipped: dict[str, int] = {}
    progress = _ProgressBar(
        len(class_labels) * int(args.points_per_class),
        enabled=not bool(args.no_progress),
    )

    try:
        for class_i, label in enumerate(class_labels, start=1):
            vidxs = list(groups[label])
            class_points = 0
            attempts = 0
            while class_points < int(args.points_per_class):
                attempts += 1
                if attempts > int(args.points_per_class) * int(args.max_tries_per_point):
                    skipped[label] = int(args.points_per_class) - class_points
                    print(
                        f"Skipping {skipped[label]} remaining points for {label}: sampling failed.",
                        flush=True,
                    )
                    progress.update(skipped[label], label=f"{label} skipped")
                    break
                vidx = int(rng.choice(vidxs))
                support_ids = _sample_support_ids(store, vidx, data_cfg.K, rng)
                if support_ids is None:
                    continue
                try:
                    batch, heldout_ids = _build_first_inner_batch(
                        sampler=sampler,
                        vidx=vidx,
                        support_ids=support_ids,
                        num_inner_queries=num_inner_queries,
                        load_rgb=load_rgb,
                        load_mask_id=load_mask_id,
                        rng=rng,
                    )
                    loss, grads = grad_fn(params, batch)
                    projected = _project_gradients(grads, projection_specs, int(args.projection_dim))
                except Exception as exc:
                    print(f"Sample failed for {label} variation index {vidx}: {exc}", flush=True)
                    continue

                grad_norm = float(np.linalg.norm(projected))
                if args.normalize:
                    projected = projected / max(grad_norm, 1e-8)
                key = keys[vidx]

                vectors.append(projected.astype(np.float32, copy=False))
                labels.append(label)
                tasks_out.append(str(key.task))
                variations_out.append(int(key.variation))
                vidxs_out.append(vidx)
                support_ids_out.append(support_ids.astype(np.int32, copy=False))
                heldout_ids_out.append(heldout_ids.astype(np.int32, copy=False))
                losses.append(float(jax.device_get(loss)))
                grad_norms.append(grad_norm)
                class_points += 1
                progress.update(1, label=label)

            print(
                f"[{class_i}/{len(class_labels)}] {label}: "
                f"{class_points}/{args.points_per_class} points",
                flush=True,
            )
    finally:
        progress.close()

    if not vectors:
        raise RuntimeError("No gradients were collected.")

    vectors_np = np.stack(vectors, axis=0).astype(np.float32, copy=False)
    labels_np = np.asarray(labels)
    tasks_np = np.asarray(tasks_out)
    variations_np = np.asarray(variations_out, dtype=np.int32)
    vidxs_np = np.asarray(vidxs_out, dtype=np.int32)
    support_ids_np = np.stack(support_ids_out, axis=0).astype(np.int32, copy=False)
    heldout_ids_np = np.stack(heldout_ids_out, axis=0).astype(np.int32, copy=False)
    losses_np = np.asarray(losses, dtype=np.float32)
    grad_norms_np = np.asarray(grad_norms, dtype=np.float32)

    embedding = None
    tsne_error = None
    plot_error = None
    if not args.no_tsne:
        embedding, tsne_error = _compute_tsne(vectors_np, args)
        if embedding is not None:
            plot_error = _write_plot(
                out_dir / "tsne.html",
                embedding,
                labels_np,
                tasks_np,
                variations_np,
                losses_np,
                grad_norms_np,
            )

    np.savez_compressed(
        out_dir / "projected_gradients.npz",
        projected_gradients=vectors_np,
        embedding=np.asarray([] if embedding is None else embedding, dtype=np.float32),
        labels=labels_np,
        tasks=tasks_np,
        variations=variations_np,
        variation_indices=vidxs_np,
        support_episode_ids=support_ids_np,
        heldout_episode_ids=heldout_ids_np,
        losses=losses_np,
        grad_norms=grad_norms_np,
    )

    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": step,
        "checkpoint_mode": mode,
        "cache_root": cache_root,
        "out_dir": str(out_dir),
        "num_points": int(vectors_np.shape[0]),
        "num_classes": int(len(set(labels))),
        "points_per_class_requested": int(args.points_per_class),
        "class_level": args.class_level,
        "class_counts": _json_safe_counts(labels),
        "skipped": {str(k): int(v) for k, v in skipped.items()},
        "loss_mean": float(np.mean(losses_np)),
        "loss_std": float(np.std(losses_np)),
        "grad_norm_mean": float(np.mean(grad_norms_np)),
        "grad_norm_std": float(np.std(grad_norms_np)),
        "projection_dim": int(args.projection_dim),
        "projected_gradients_normalized": bool(args.normalize),
        "selected_param_tensors": selected_paths,
        "selected_param_count": int(selected_param_count),
        "num_inner_queries": int(num_inner_queries),
        "load_rgb": bool(load_rgb),
        "load_mask_id": bool(load_mask_id),
        "preload_to_memory": bool(args.preload_to_memory),
        "keep_open": bool(args.keep_open),
        "tasks": list(tasks),
        "exclude_tasks": list(exclude_tasks),
        "tsne_error": tsne_error,
        "plot_error": plot_error,
    }
    _write_summary(out_dir / "summary.json", summary)
    print(f"Saved projected gradients: {out_dir / 'projected_gradients.npz'}", flush=True)
    if embedding is not None and plot_error is None:
        print(f"Saved t-SNE plot: {out_dir / 'tsne.html'}", flush=True)
    elif tsne_error:
        print(f"Skipped t-SNE: {tsne_error}", flush=True)
    elif plot_error:
        print(f"Skipped HTML plot: {plot_error}", flush=True)
    print(f"Saved summary: {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
