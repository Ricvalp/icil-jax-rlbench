#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=rlbench_supernode_preload
#SBATCH --partition=defq
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --cpus-per-task=32
#SBATCH --time=72:00:00
#SBATCH --output=sbatch/das/logs/rlbench_supernode_preload-%j.out
#SBATCH --error=sbatch/das/logs/rlbench_supernode_preload-%j.err

set -euo pipefail

# Submit from the repo root, or override REPO_DIR when submitting:
#   sbatch --export=ALL,REPO_DIR=/path/to/icil_jax_rlbench sbatch/das/pretrain_supernode_preload_h100.sbatch
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CONDA_ENV="${CONDA_ENV:-icil-rlbench}"

# Data/config knobs.
SEED="${SEED:-0}"
NUM_STEPS="${NUM_STEPS:-100000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"  # Must be divisible by 4 local GPUs.
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
LOSS_TYPE="${LOSS_TYPE:-mse}"
LOG_EVERY="${LOG_EVERY:-100}"
CKPT_EVERY="${CKPT_EVERY:-10000}"
TASK_SAMPLING="${TASK_SAMPLING:-task_uniform}"
TASK_SAMPLING_ALPHA="${TASK_SAMPLING_ALPHA:-1.0}"
TASKS="${TASKS:-}"                  # Optional comma-separated task names.
EXCLUDE_TASKS="${EXCLUDE_TASKS:-}"  # Optional comma-separated task names.

K="${K:-4}"
L="${L:-10}"
T_OBS="${T_OBS:-2}"
H="${H:-16}"
STRIDE="${STRIDE:-2}"
TRAJ_LEN="${TRAJ_LEN:-64}"
ACTION_REPRESENTATION="${ACTION_REPRESENTATION:-absolute}"

# Supernode/model knobs.
D_MODEL="${D_MODEL:-256}"
N_HEADS="${N_HEADS:-4}"
DROPOUT="${DROPOUT:-0.0}"
MLP_MULT="${MLP_MULT:-4}"
FRAME_LAYERS="${FRAME_LAYERS:-2}"
QUERY_LAYERS="${QUERY_LAYERS:-1}"
SUPPORT_LAYERS="${SUPPORT_LAYERS:-2}"
TRAJ_LAYERS="${TRAJ_LAYERS:-1}"
SUPERNODES="${SUPERNODES:-64}"
SUPERNODE_TEMPERATURE="${SUPERNODE_TEMPERATURE:-0.02}"
SUPERNODE_LAYERS="${SUPERNODE_LAYERS:-2}"
QUERY_NUM_LATENTS="${QUERY_NUM_LATENTS:-16}"
SUPPORT_NUM_LATENTS="${SUPPORT_NUM_LATENTS:-64}"
DECODER_LAYERS="${DECODER_LAYERS:-4}"
USE_RGB="${USE_RGB:-True}"
USE_MASK_ID="${USE_MASK_ID:-False}"
USE_TRAJ_TOKENS="${USE_TRAJ_TOKENS:-True}"

# Logging/checkpoint knobs.
WANDB_ENABLE="${WANDB_ENABLE:-True}"
RUN_NAME="${RUN_NAME:-pretrain_supernode_preload_das_${SLURM_JOB_ID:-manual}}"
RESUME_PATH="${RESUME_PATH:-}"
RESUME_OPTIMIZER="${RESUME_OPTIMIZER:-True}"
RESUME_RNG="${RESUME_RNG:-True}"

if (( GLOBAL_BATCH_SIZE % 4 != 0 )); then
  echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must be divisible by 4 GPUs." >&2
  exit 1
fi

if [[ -n "${RESUME_PATH}" && ! -f "${RESUME_PATH}" ]]; then
  echo "RESUME_PATH does not exist: ${RESUME_PATH}" >&2
  exit 1
fi

module load cuda12.6/toolkit

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniforge3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/miniconda3/bin/activate" ]]; then
  source "${HOME}/miniconda3/bin/activate"
elif [[ -f "${HOME}/miniforge3/bin/activate" ]]; then
  source "${HOME}/miniforge3/bin/activate"
else
  echo "Could not find a conda activation script under ${HOME}." >&2
  exit 1
fi
conda activate "${CONDA_ENV}"

cd "${REPO_DIR}"
source ./env_jax_rlbench_das.sh

mkdir -p sbatch/das/logs

export PYTHONPATH="$(dirname "${REPO_DIR}")${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ICIL_JAX_RLBENCH_CHECKPOINT_DIR}/pretrain_supernode_preload/${SLURM_JOB_ID:-manual}}"
mkdir -p "${CHECKPOINT_DIR}"

args=(
  "${REPO_DIR}/pretrain_direct_regression.py"
  "--config=${REPO_DIR}/configs/pretrain_supernode.py"
  "--config.data.cache_root=${ICIL_CACHE_ROOT}"
  "--config.data.preload_to_memory=True"
  "--config.data.keep_open=False"
  "--config.data.K=${K}"
  "--config.data.L=${L}"
  "--config.data.T_obs=${T_OBS}"
  "--config.data.H=${H}"
  "--config.data.stride=${STRIDE}"
  "--config.data.traj_len=${TRAJ_LEN}"
  "--config.data.action_representation=${ACTION_REPRESENTATION}"
  "--config.data.task_sampling=${TASK_SAMPLING}"
  "--config.data.task_sampling_alpha=${TASK_SAMPLING_ALPHA}"
  "--config.model.encoder.d_model=${D_MODEL}"
  "--config.model.encoder.n_heads=${N_HEADS}"
  "--config.model.encoder.dropout=${DROPOUT}"
  "--config.model.encoder.mlp_mult=${MLP_MULT}"
  "--config.model.encoder.frame_layers=${FRAME_LAYERS}"
  "--config.model.encoder.query_layers=${QUERY_LAYERS}"
  "--config.model.encoder.support_layers=${SUPPORT_LAYERS}"
  "--config.model.encoder.traj_layers=${TRAJ_LAYERS}"
  "--config.model.encoder.supernodes=${SUPERNODES}"
  "--config.model.encoder.supernode_temperature=${SUPERNODE_TEMPERATURE}"
  "--config.model.encoder.supernode_layers=${SUPERNODE_LAYERS}"
  "--config.model.encoder.query_num_latents=${QUERY_NUM_LATENTS}"
  "--config.model.encoder.support_num_latents=${SUPPORT_NUM_LATENTS}"
  "--config.model.encoder.use_rgb=${USE_RGB}"
  "--config.model.encoder.use_mask_id=${USE_MASK_ID}"
  "--config.model.encoder.use_traj_tokens=${USE_TRAJ_TOKENS}"
  "--config.model.decoder.n_layers=${DECODER_LAYERS}"
  "--config.train.seed=${SEED}"
  "--config.train.num_steps=${NUM_STEPS}"
  "--config.train.batch_size=${GLOBAL_BATCH_SIZE}"
  "--config.train.lr=${LR}"
  "--config.train.weight_decay=${WEIGHT_DECAY}"
  "--config.train.grad_clip_norm=${GRAD_CLIP_NORM}"
  "--config.train.loss_type=${LOSS_TYPE}"
  "--config.train.log_every=${LOG_EVERY}"
  "--config.train.ckpt_every=${CKPT_EVERY}"
  "--config.train.checkpoint_dir=${CHECKPOINT_DIR}"
  "--config.train.resume_path=${RESUME_PATH}"
  "--config.train.resume_optimizer=${RESUME_OPTIMIZER}"
  "--config.train.resume_rng=${RESUME_RNG}"
  "--config.wandb.enable=${WANDB_ENABLE}"
  "--config.wandb.name=${RUN_NAME}"
)

if [[ -n "${TASKS}" ]]; then
  args+=("--config.data.tasks=${TASKS}")
fi
if [[ -n "${EXCLUDE_TASKS}" ]]; then
  args+=("--config.data.exclude_tasks=${EXCLUDE_TASKS}")
fi

echo "Running ${RUN_NAME}"
echo "Repo: ${REPO_DIR}"
echo "Checkpoint dir: ${CHECKPOINT_DIR}"
echo "Global batch size: ${GLOBAL_BATCH_SIZE}"

srun python "${args[@]}"
