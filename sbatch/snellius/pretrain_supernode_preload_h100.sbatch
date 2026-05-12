#!/bin/bash
#SBATCH --job-name=rlbench_supernode_preload
#SBATCH --output=sbatch/snellius/logs/rlbench_supernode_preload-%j.out
#SBATCH --error=sbatch/snellius/logs/rlbench_supernode_preload-%j.err
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=72:00:00

set -euo pipefail

# Submit from the package/repo directory:
#   sbatch sbatch/snellius/pretrain_supernode_preload_h100.sbatch

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
conda activate icil-rlbench

cd "${SLURM_SUBMIT_DIR}"
mkdir -p sbatch/snellius/logs
source ./env_jax_rlbench_snellius.sh

# Run from the parent directory so imports resolve as icil_jax_rlbench.*.
cd ..

srun python icil_jax_rlbench/pretrain_direct_regression.py \
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
  --config.model.encoder.use_rgb=True \
  --config.model.encoder.use_mask_id=False \
  --config.model.encoder.use_traj_tokens=True \
  --config.model.decoder.n_layers=4 \
  --config.train.seed=0 \
  --config.train.num_steps=100000 \
  --config.train.batch_size=8 \
  --config.train.lr=1e-4 \
  --config.train.weight_decay=1e-4 \
  --config.train.grad_clip_norm=1.0 \
  --config.train.loss_type=mse \
  --config.train.log_every=100 \
  --config.train.ckpt_every=10000 \
  --config.train.resume_path='' \
  --config.train.resume_optimizer=True \
  --config.train.resume_rng=True \
  --config.wandb.enable=True \
  --config.wandb.name=pretrain_supernode_preload_snellius
