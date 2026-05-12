# Standalone JAX RLBench ICIL

This directory is independent from the existing `icil/`, `icil_jax_query_memory/`, and diagnostics code. It only assumes the dense RLBench H5 cache format already present under `ICIL_CACHE_ROOT`.

## Implemented

- Direct-regression in-context imitation policy in JAX/Flax.
- RLBench dense H5 reader with optional full RAM preload.
- Perceiver point-cloud encoder.
- Supernode point-cloud tokenizer encoder.
- Pretraining on cached support/query episodes.
- Parameter MAML/FOMAML fine-tuning.
- Memory MAML/FOMAML fine-tuning where the inner loop updates encoded support-memory tokens.
- Multi-device `pmap` training with global batch sharding.
- Pickle checkpoints containing `params`, `opt_state`, `rng`, `step`, and config.

## Main Commands

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

Enable full cache preload:

```bash
--config.data.preload_to_memory=True --config.data.keep_open=False
```

## Environment Setup

Source the environment script for the machine you are using after activating the
Python/JAX environment and before launching training:

```bash
# Local workstation
source ./env_jax_rlbench_local.sh

# DAS
source ./env_jax_rlbench_das.sh

# Snellius
source ./env_jax_rlbench_snellius.sh
```

These scripts set the dense cache root, output/checkpoint/profile roots, W&B
defaults, and JAX runtime flags:

- `ICIL_CACHE_ROOT`: dense RLBench H5 cache.
- `ICIL_JAX_RLBENCH_RUN_ROOT`: base directory for run artifacts.
- `ICIL_JAX_RLBENCH_OUTPUT_DIR`: general output directory.
- `ICIL_JAX_RLBENCH_CHECKPOINT_DIR`: checkpoint root; configs append the training mode.
- `ICIL_JAX_RLBENCH_PROFILE_DIR`: profiling output directory.
- `ICIL_JAX_RLBENCH_WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_MODE`: W&B defaults.
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `PYTHONUNBUFFERED=1`.

The local script also sets CoppeliaSim/X11 variables for local RLBench
evaluation. Training commands still need this repo on `PYTHONPATH`, as shown in
the main commands above.

W&B is disabled by default in the configs; enable it with
`--config.wandb.enable=True`. Resume or fine-tune checkpoint paths are passed
through config flags:

```bash
PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false python icil_jax_rlbench/pretrain_direct_regression.py \
  --config=icil_jax_rlbench/configs/pretrain_perceiver.py \
  --config.train.resume_path=/path/to/step_0100000.pkl
```

Switch encoder:

```bash
--config.model.encoder.encoder_type=supernode
```

## Important Configs

- `data.K`: number of support demonstrations.
- `data.L`: support keyframes sampled per demonstration.
- `data.T_obs`: query observation history length.
- `data.H`: predicted action chunk length.
- `data.stride`: temporal stride for observations/actions.
- `data.traj_len`: fixed support action trajectory tokens per demo.
- `data.preload_to_memory`: load all H5 arrays into host RAM at startup.
- `model.encoder.encoder_type`: `perceiver` or `supernode`.
- `model.encoder.use_rgb`: include dense RGB point features.
- `model.encoder.support_num_latents`: compressed support-memory token count.
- `maml.first_order`: `False` gives full second-order MAML; `True` gives FOMAML.
- `maml.inner_param_include`: optional substring filters for parameter inner-loop updates.
- `maml.inner_lr`: inner-loop learning rate.

## Notes

The first version intentionally implements the direct-regression path only. It does not convert old checkpoints and does not import any existing repo modules. Parameter MAML can update all parameters by default; memory MAML updates only the encoded support-memory tokens in the inner loop.

Resume / fine-tune from a pretraining checkpoint while resetting the outer optimizer:

```bash
--config.train.resume_path=/path/to/pretrain.pkl \
--config.train.resume_optimizer=False \
--config.train.resume_rng=False
```

Optional decoder memory-attention logging:

```bash
--config.train.log_attention_stats=True
```

This adds the following metrics. The default is `False`, so the normal training path does not request attention weights.

- `train/attn_memory_entropy`: normalized entropy of decoder cross-attention over support/memory tokens; uniform attention is near 1, selective attention is closer to 0.
- `train/attn_memory_max`: max probability after renormalizing attention over support/memory tokens only.
- `train/attn_memory_raw_max`: max raw cross-attention probability assigned to a support/memory token.
- `train/attn_memory_mass`: total raw cross-attention probability mass assigned to support/memory tokens.
