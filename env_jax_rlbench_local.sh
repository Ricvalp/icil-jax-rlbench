#!/bin/bash
# Local environment for the standalone JAX RLBench ICIL repo.
# Source before local training/evaluation: source env_jax_rlbench_local.sh

# =============================================================================
# RLBench dense cache
# =============================================================================
export ICIL_CACHE_ROOT="/mnt/external_storage/robotics/rlbench/icil_rlbench/.rlbench_cache_dense_v4"

# =============================================================================
# Outputs/checkpoints
# =============================================================================
export ICIL_JAX_RLBENCH_RUN_ROOT="/mnt/external_storage/robotics/rlbench/icil_jax_rlbench_runs"
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

# =============================================================================
# Local RLBench/CoppeliaSim evaluation
# =============================================================================
# Adjust COPPELIASIM_ROOT if your local install uses a different path.
export COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-${HOME}/CoppeliaSim}"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-xcb_glx}"
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"
export DISPLAY="${DISPLAY:-:99}"

echo "[env_jax_rlbench_local.sh] ICIL_CACHE_ROOT=${ICIL_CACHE_ROOT}"
echo "[env_jax_rlbench_local.sh] ICIL_JAX_RLBENCH_OUTPUT_DIR=${ICIL_JAX_RLBENCH_OUTPUT_DIR}"
echo "[env_jax_rlbench_local.sh] ICIL_JAX_RLBENCH_CHECKPOINT_DIR=${ICIL_JAX_RLBENCH_CHECKPOINT_DIR}"
echo "[env_jax_rlbench_local.sh] ICIL_JAX_RLBENCH_WANDB_PROJECT=${ICIL_JAX_RLBENCH_WANDB_PROJECT}"
echo "[env_jax_rlbench_local.sh] WANDB_ENTITY=${WANDB_ENTITY}"
echo "[env_jax_rlbench_local.sh] WANDB_MODE=${WANDB_MODE}"
echo "[env_jax_rlbench_local.sh] COPPELIASIM_ROOT=${COPPELIASIM_ROOT}"
echo "[env_jax_rlbench_local.sh] DISPLAY=${DISPLAY}"
