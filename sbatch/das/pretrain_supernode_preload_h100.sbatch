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

# Submit from the package/repo directory:
#   sbatch sbatch/das/pretrain_supernode_preload_h100.sbatch

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
conda activate icil-rlbench

cd "${SLURM_SUBMIT_DIR}"
mkdir -p sbatch/das/logs
source ./env_jax_rlbench_das.sh
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "CUDA_VISIBLE_DEVICES before srun: ${CUDA_VISIBLE_DEVICES:-unset}"
srun --gres=gpu:4 nvidia-smi
srun --gres=gpu:4 python -c 'import jax; backend = jax.default_backend(); print(f"JAX backend: {backend}; devices: {jax.devices()}"); assert backend == "gpu", f"Expected JAX gpu backend, got {backend}"'

srun --gres=gpu:4 python -m icil_jax_rlbench.pretrain_direct_regression \
  --config=icil_jax_rlbench/configs/pretrain_supernode.py \
  --config.data.preload_to_memory=True \
  --config.data.keep_open=False \
  --config.data.K=4 \
  --config.data.L=10 \
  --config.data.T_obs=2 \
  --config.data.H=16 \
  --config.data.stride=2 \
  --config.data.traj_len=64 \
  --config.data.action_representation=absolute \
  --config.data.task_sampling=task_uniform \
  --config.data.task_sampling_alpha=1.0 \
  --config.model.encoder.d_model=256 \
  --config.model.encoder.n_heads=4 \
  --config.model.encoder.dropout=0.0 \
  --config.model.encoder.mlp_mult=4 \
  --config.model.encoder.frame_layers=2 \
  --config.model.encoder.query_layers=1 \
  --config.model.encoder.support_layers=2 \
  --config.model.encoder.traj_layers=1 \
  --config.model.encoder.supernodes=64 \
  --config.model.encoder.supernode_temperature=0.02 \
  --config.model.encoder.supernode_layers=2 \
  --config.model.encoder.query_num_latents=16 \
  --config.model.encoder.support_num_latents=64 \
  --config.model.encoder.use_support_tokens=True \
  --config.model.encoder.use_rgb=True \
  --config.model.encoder.use_mask_id=False \
  --config.model.encoder.use_traj_tokens=True \
  --config.model.decoder.n_layers=4 \
  --config.model.decoder.context_mode=two_ctx \
  --config.train.seed=0 \
  --config.train.num_steps=100000 \
  --config.train.batch_size=8 \
  --config.train.lr=1e-4 \
  --config.train.weight_decay=1e-4 \
  --config.train.grad_clip_norm=1.0 \
  --config.train.loss_type=mse \
  --config.train.log_every=100 \
  --config.train.ckpt_every=10000 \
  --config.train.prefetch_workers=4 \
  --config.train.prefetch_batches=4 \
  --config.train.resume_path='' \
  --config.train.resume_optimizer=True \
  --config.train.resume_rng=True \
  --config.wandb.enable=True \
  --config.wandb.name=pretrain_supernode_preload_das \
  --config.wandb.prediction_log_every=200 \
  --config.wandb.prediction_num_samples=64 \
  --config.wandb.prediction_num_plots=4
