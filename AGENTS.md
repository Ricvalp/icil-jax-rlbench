# AGENTS.md

Instructions for Codex or other coding agents working on this standalone JAX RLBench ICIL repository.

## Hard Constraints

- This directory is intended to be moved and developed as an independent repository.
- Do not import from the old parent repo packages, including `icil`, `icil_jax_query_memory`, `diagnostics`, or MetaWorld code.
- Keep all implementation self-contained under `icil_jax_rlbench/` unless explicitly asked to add external tooling.
- Do not add checkpoint conversion from old PyTorch checkpoints unless explicitly requested. The current checkpoint format is standalone pickle.
- Preserve backward compatibility for checkpoints produced by this package whenever feasible.
- Avoid destructive git commands. Do not reset or revert user changes unless explicitly asked.

## Project Purpose

This package is a JAX/Flax reimplementation of the RLBench in-context imitation learning direct-regression experiments. The main goal is to make pretraining and MAML-style finetuning fast enough to run many more steps than the older Python-loop PyTorch implementation.

Supported training modes:

- `pretrain`: standard in-context direct regression from support demonstrations plus query observation to an action chunk.
- `param_maml`: parameter MAML/FOMAML. Inner loop updates model parameters on support-derived held-out queries, outer loop evaluates adapted parameters on a query episode.
- `memory_maml`: memory MAML/FOMAML. Inner loop updates encoded support-memory tokens only; model parameters are optimized by the outer loop.

The first implementation intentionally covers direct regression only. It does not implement all old ICIL variants.

## Directory Structure

- `icil_jax_rlbench/data/`: standalone RLBench dense H5 cache reader and samplers.
- `icil_jax_rlbench/models/`: Flax attention blocks, Perceiver/Supernode encoders, and direct-regression policy.
- `icil_jax_rlbench/train/`: pmap training runner, MAML/pretrain steps, checkpoint utilities.
- `icil_jax_rlbench/configs/`: ml-collections configs for the six main combinations.
- `icil_jax_rlbench/pretrain_direct_regression.py`: pretraining entrypoint.
- `icil_jax_rlbench/param_maml_direct_regression.py`: parameter MAML/FOMAML entrypoint.
- `icil_jax_rlbench/memory_maml_direct_regression.py`: memory MAML/FOMAML entrypoint.
- `README.md`: user-facing quickstart.

## Data Contract

The only external data dependency is the dense RLBench H5 cache under `ICIL_CACHE_ROOT` or `--config.data.cache_root`.

Expected cache layout:

```text
CACHE_ROOT/
  task_name/
    variation0.h5
    variation1.h5
    ...
```

Each variation H5 file is expected to contain:

```text
episode_ids
episodes/<episode_id>/xyz       [T, N, 3]
episodes/<episode_id>/valid     [T, N]
episodes/<episode_id>/state     [T, S]
episodes/<episode_id>/action    [T, A]
episodes/<episode_id>/rgb       [T, N, 3] optional
episodes/<episode_id>/mask_id   [T, N] optional
```

Known current RLBench dense cache dimensions are usually:

- `N=1024` points
- `S=8` state dims
- `A=8` action dims

Do not assume those dimensions in code; infer them through `RLBenchCacheStore.infer_dims()`.

## Sampling Semantics

Core config fields:

- `data.K`: number of support demos.
- `data.L`: keyframes sampled from each support demo.
- `data.T_obs`: query observation history length.
- `data.H`: predicted action chunk length.
- `data.stride`: temporal stride for observation/action indices.
- `data.traj_len`: fixed support action-trajectory token length.
- `data.task_sampling`: `variation_uniform`, `task_uniform`, or `variation_power`.
- `data.preload_to_memory`: if `True`, load the full H5 cache into host RAM at startup.

Batch builders in `data/sampler.py` define the training semantics. If changing them, update this file and README.

Pretrain batch:

- sample one variation
- sample `K` support episodes and one query episode
- support gives `cond_*`
- query gives `query_*` and `target_action`

Parameter-MAML batch:

- sample one variation
- sample `K` support episodes and one query episode
- inner loop uses leave-one-out support episodes: `K-1` context demos predict the held-out support episode chunk
- outer loop predicts chunks from the query episode using support context

Memory-MAML batch:

- sample one variation
- sample `K` support episodes and one query episode
- initialize memory from `K-1` support demos
- inner loop updates memory tokens using chunks from the held-out support demo
- outer loop predicts chunks from the query episode using adapted memory

## Model Semantics

Main model: `DirectRegressionPolicy` in `models/direct_regression_policy.py`.

High-level flow:

1. Encode support demonstrations into support/memory tokens.
2. Encode query observations into query tokens.
3. Decode action chunk with learned action queries attending to query + support/memory tokens.

Encoder choices:

- `perceiver`: each frame point cloud is compressed by latent cross-attention.
- `supernode`: each frame point cloud is softly pooled around deterministic supernode centers, then refined with self-attention.

Support tokens include:

- point/state tokens from sampled support keyframes
- optional support trajectory action tokens when `model.encoder.use_traj_tokens=True`
- optional RGB when `model.encoder.use_rgb=True`
- optional mask IDs when `model.encoder.use_mask_id=True`

Memory in this repo means encoded support tokens, not an external replay buffer.

## MAML Semantics

Config:

```bash
--config.maml.first_order=False  # full second-order MAML
--config.maml.first_order=True   # FOMAML
```

Parameter MAML:

- inner loop differentiates `inner_loss` w.r.t. parameters
- if `first_order=False`, outer gradients flow through the inner update
- if `first_order=True`, inner gradients are stop-gradiented before applying updates
- `maml.inner_param_include` and `maml.inner_param_exclude` optionally select fast weights by parameter-name substring

Memory MAML:

- support memory tokens are encoded from support demonstrations
- inner loop differentiates support-heldout loss w.r.t. those memory tokens
- adapted memory tokens are used for the outer query loss
- model parameters receive outer gradients; full second-order mode allows gradients through memory adaptation

## Attention Logging

Optional config:

```bash
--config.train.log_attention_stats=True
```

Default is `False` to avoid requesting decoder attention weights on the normal path.

Metrics:

- `train/attn_memory_entropy`: normalized entropy over support/memory tokens. Uniform is near 1; selective is closer to 0.
- `train/attn_memory_max`: maximum probability after renormalizing over memory tokens only.
- `train/attn_memory_raw_max`: maximum raw decoder cross-attention probability assigned to a memory token.
- `train/attn_memory_mass`: total raw decoder cross-attention mass assigned to memory tokens.
- `train/attn_query_entropy`: normalized entropy over query tokens.

## Checkpoints

Checkpoint files are pickle payloads saved by `train/checkpoints.py`.

Payload fields:

- `step`
- `params`
- `opt_state`
- `rng`
- `config`
- `extra`

Resume from a package checkpoint:

```bash
--config.train.resume_path=/path/to/step_XXXXXXX.pkl
```

Finetune from pretrained params while resetting optimizer and RNG:

```bash
--config.train.resume_path=/path/to/pretrain.pkl \
--config.train.resume_optimizer=False \
--config.train.resume_rng=False
```

## Common Commands

Pretrain Perceiver:

```bash
PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false python icil_jax_rlbench/pretrain_direct_regression.py \
  --config=icil_jax_rlbench/configs/pretrain_perceiver.py
```

Pretrain Supernode:

```bash
PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false python icil_jax_rlbench/pretrain_direct_regression.py \
  --config=icil_jax_rlbench/configs/pretrain_supernode.py
```

Full second-order parameter MAML:

```bash
PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false python icil_jax_rlbench/param_maml_direct_regression.py \
  --config=icil_jax_rlbench/configs/param_maml_perceiver.py \
  --config.maml.first_order=False
```

FOMAML parameter fine-tuning:

```bash
PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false python icil_jax_rlbench/param_maml_direct_regression.py \
  --config=icil_jax_rlbench/configs/param_maml_perceiver.py \
  --config.maml.first_order=True
```

Full second-order memory MAML:

```bash
PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false python icil_jax_rlbench/memory_maml_direct_regression.py \
  --config=icil_jax_rlbench/configs/memory_maml_perceiver.py \
  --config.maml.first_order=False
```

Preload the full cache into RAM:

```bash
--config.data.preload_to_memory=True --config.data.keep_open=False
```

## Verification Before Finalizing Changes

Always run at least:

```bash
python -m compileall -q icil_jax_rlbench
```

Check standalone constraint:

```bash
rg "from icil(\.|\s)|import icil(\.|\s)|icil_jax_query_memory|diagnostics|metaworld" icil_jax_rlbench
```

If touching model or training logic, run a tiny smoke step on real cache if available:

```bash
PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false python icil_jax_rlbench/pretrain_direct_regression.py \
  --config=icil_jax_rlbench/configs/pretrain_perceiver.py \
  --config.train.num_steps=1 \
  --config.train.batch_size=1 \
  --config.train.log_every=1 \
  --config.train.ckpt_every=1 \
  --config.train.checkpoint_dir=/tmp/icil_jax_rlbench_smoke \
  --config.data.K=2 \
  --config.data.L=2 \
  --config.data.H=4 \
  --config.data.traj_len=8 \
  --config.model.encoder.use_rgb=False \
  --config.model.encoder.d_model=32 \
  --config.model.encoder.n_heads=4 \
  --config.model.encoder.frame_num_latents=2 \
  --config.model.encoder.query_num_latents=2 \
  --config.model.encoder.support_num_latents=4 \
  --config.model.decoder.n_layers=1
```

For MAML changes, also smoke-test one tiny full second-order step:

```bash
PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false python icil_jax_rlbench/param_maml_direct_regression.py \
  --config=icil_jax_rlbench/configs/param_maml_perceiver.py \
  --config.train.num_steps=1 \
  --config.train.batch_size=1 \
  --config.train.log_every=1 \
  --config.train.ckpt_every=1 \
  --config.train.checkpoint_dir=/tmp/icil_jax_rlbench_param_smoke \
  --config.data.K=2 \
  --config.data.L=2 \
  --config.data.H=4 \
  --config.data.traj_len=8 \
  --config.model.encoder.use_rgb=False \
  --config.model.encoder.d_model=32 \
  --config.model.encoder.n_heads=4 \
  --config.model.encoder.frame_num_latents=2 \
  --config.model.encoder.query_num_latents=2 \
  --config.model.encoder.support_num_latents=4 \
  --config.model.decoder.n_layers=1 \
  --config.maml.inner_steps=1 \
  --config.maml.num_inner_queries=1 \
  --config.maml.num_query_loss_samples=1 \
  --config.maml.first_order=False
```

## Coding Guidelines

- Prefer simple, explicit JAX/Flax code over clever abstractions.
- Keep tensor shape comments near nontrivial reshapes/scans/vmaps.
- Avoid dynamic shapes inside jitted/pmap code.
- Any new optional metric should be behind a config flag if it requires extra forward computation or attention weight materialization.
- Keep configs in `configs/base.py` as the source of defaults; wrapper configs should stay thin.
- Preserve ASCII-only source unless there is a concrete reason not to.
- Do not commit generated `__pycache__`, temporary checkpoints, W&B outputs, or smoke-test artifacts.

## Known Limitations / Open Work

- No evaluation/rollout scripts are implemented yet in this standalone package.
- No old checkpoint conversion.
- Direct-regression only; no full old ICIL feature parity.
- Supernode tokenizer is a deterministic soft-pooling implementation, not a byte-for-byte clone of any old tokenizer.
- Data loading is synchronous in the training loop. Optional preload helps small caches; larger datasets may need async/prefetch workers later.
