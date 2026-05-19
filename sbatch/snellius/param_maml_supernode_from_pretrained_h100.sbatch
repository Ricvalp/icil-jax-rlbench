#!/bin/bash
#SBATCH --job-name=rlbench_param_maml_pretrained
#SBATCH --output=sbatch/snellius/logs/rlbench_param_maml_pretrained-%j.out
#SBATCH --error=sbatch/snellius/logs/rlbench_param_maml_pretrained-%j.err
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=0
#SBATCH --time=24:00:00

set -euo pipefail

# Edit this before submitting. Model architecture and shape-related data knobs
# are loaded from this checkpoint when resume_config_from_checkpoint=True below.
PRETRAINED_CHECKPOINT="/projects/prjs1905/robotics/rlbench/icil_jax_rlbench_runs/checkpoints/pretrain/1snk0gkb/step_0400000.pkl"

if [[ "$PRETRAINED_CHECKPOINT" == "/path/to/pretrained/checkpoint.pkl" || ! -f "$PRETRAINED_CHECKPOINT" ]]; then
  echo "Set PRETRAINED_CHECKPOINT in this sbatch file to an existing checkpoint." >&2
  exit 1
fi

if [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniforge3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/miniforge3/bin/activate" ]]; then
  source "${HOME}/miniforge3/bin/activate"
elif [[ -f "${HOME}/miniconda3/bin/activate" ]]; then
  source "${HOME}/miniconda3/bin/activate"
else
  echo "Could not find a conda activation script under ${HOME}." >&2
  exit 1
fi
conda activate jax-icil

cd "${SLURM_SUBMIT_DIR}"
mkdir -p sbatch/snellius/logs
source ./env_jax_rlbench_snellius.sh
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "CUDA_VISIBLE_DEVICES before srun: ${CUDA_VISIBLE_DEVICES:-unset}"
srun --gres=gpu:2 nvidia-smi
srun --gres=gpu:2 python -c 'import jax; backend = jax.default_backend(); print(f"JAX backend: {backend}; devices: {jax.devices()}"); assert backend == "gpu", f"Expected JAX gpu backend, got {backend}"'

srun --gres=gpu:2 python -m icil_jax_rlbench.param_maml_direct_regression \
  --config=icil_jax_rlbench/configs/param_maml_supernode.py \
  --config.data.preload_to_memory=True \
  --config.data.keep_open=False \
  --config.data.task_sampling=variation_power \
  --config.data.task_sampling_alpha=0.8 \
  --config.maml.inner_steps=4 \
  --config.maml.num_inner_queries=4 \
  --config.maml.num_query_loss_samples=4 \
  --config.maml.inner_lr=5e-2 \
  --config.maml.first_order=False \
  --config.maml.inner_grad_clip_norm=1.0 \
  --config.train.seed=0 \
  --config.train.num_steps=200000 \
  --config.train.batch_size=32 \
  --config.train.lr=1e-4 \
  --config.train.weight_decay=1e-4 \
  --config.train.grad_clip_norm=1.0 \
  --config.train.loss_type=l1 \
  --config.train.log_every=100 \
  --config.train.ckpt_every=5000 \
  --config.train.prefetch_workers=4 \
  --config.train.prefetch_batches=4 \
  --config.train.resume_path="${PRETRAINED_CHECKPOINT}" \
  --config.train.resume_config_from_checkpoint=True \
  --config.train.resume_optimizer=False \
  --config.train.resume_rng=False \
  --config.wandb.enable=True \
  --config.wandb.name=param_maml_supernode_from_pretrained_snellius \
  --config.wandb.prediction_log_every=500 \
  --config.wandb.prediction_num_samples=64 \
  --config.wandb.prediction_num_plots=16 \
  --config.train.log_attention_stats=True
