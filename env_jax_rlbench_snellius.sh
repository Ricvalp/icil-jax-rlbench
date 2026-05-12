#!/bin/bash
# Snellius environment for the standalone JAX RLBench ICIL repo.
# Source inside sbatch scripts after activating the JAX environment: source env_jax_rlbench_snellius.sh

# =============================================================================
# RLBench dense cache
# =============================================================================
export ICIL_CACHE_ROOT="/projects/prjs1905/robotics/rlbench/icil_rlbench/.rlbench_cache_dense_v4"

# =============================================================================
# Outputs/checkpoints
# =============================================================================
export ICIL_JAX_RLBENCH_RUN_ROOT="/projects/prjs1905/robotics/rlbench/icil_jax_rlbench_runs"
export ICIL_JAX_RLBENCH_OUTPUT_DIR="${ICIL_JAX_RLBENCH_RUN_ROOT}/outputs"
export ICIL_JAX_RLBENCH_CHECKPOINT_DIR="${ICIL_JAX_RLBENCH_RUN_ROOT}/checkpoints"
export ICIL_JAX_RLBENCH_PROFILE_DIR="${ICIL_JAX_RLBENCH_RUN_ROOT}/profiles"

mkdir -p "${ICIL_JAX_RLBENCH_OUTPUT_DIR}" "${ICIL_JAX_RLBENCH_CHECKPOINT_DIR}" "${ICIL_JAX_RLBENCH_PROFILE_DIR}"

# =============================================================================
# W&B
# =============================================================================
export ICIL_JAX_RLBENCH_WANDB_PROJECT="icil-jax-rlbench"
export WANDB_ENTITY="ricvalp"
export WANDB_MODE="online"

# =============================================================================
# JAX/runtime
# =============================================================================
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTHONUNBUFFERED="1"

echo "[env_jax_rlbench_snellius.sh] ICIL_CACHE_ROOT=${ICIL_CACHE_ROOT}"
echo "[env_jax_rlbench_snellius.sh] ICIL_JAX_RLBENCH_OUTPUT_DIR=${ICIL_JAX_RLBENCH_OUTPUT_DIR}"
echo "[env_jax_rlbench_snellius.sh] ICIL_JAX_RLBENCH_CHECKPOINT_DIR=${ICIL_JAX_RLBENCH_CHECKPOINT_DIR}"
echo "[env_jax_rlbench_snellius.sh] ICIL_JAX_RLBENCH_WANDB_PROJECT=${ICIL_JAX_RLBENCH_WANDB_PROJECT}"
echo "[env_jax_rlbench_snellius.sh] WANDB_ENTITY=${WANDB_ENTITY}"
echo "[env_jax_rlbench_snellius.sh] WANDB_MODE=${WANDB_MODE}"
