#!/bin/bash
#SBATCH --job-name=direct_regression_traj_perceiver_v2_resume
#SBATCH --output=/gpfs/home1/rvalperga/icil-rlbench/.smanager/scripts/icil.pretrain_perceiver_encoder_decoder_ddp/logs/direct_regression_traj_perceiver_v2_resume.%j.out
#SBATCH --error=/gpfs/home1/rvalperga/icil-rlbench/.smanager/scripts/icil.pretrain_perceiver_encoder_decoder_ddp/logs/direct_regression_traj_perceiver_v2_resume.%j.err
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:4
#SBATCH --time=72:00:00

source ~/miniforge3/bin/activate
conda activate icil-rlbench

cd /gpfs/home1/rvalperga/icil_jax_rlbench

source env_snellius.sh
source env_snellius_coppeliasim.sh

export WANDB_PROJECT="icil-jax-direct-regression-pretrain"
export WANDB_ENTITY="ricvalp"
export WANDB_MODE="online"

export PYTHONPATH=/gpfs/home1/rvalperga/icil-rlbench

export DISPLAY=:$((100 + SLURM_JOB_ID % 5000))

RESUME_CHECKPOINT="/gpfs/home1/rvalperga/icil-rlbench/REPLACE_WITH_RESUME_CHECKPOINT.pt"
if [ ! -f "${RESUME_CHECKPOINT}" ]; then
  echo "Set RESUME_CHECKPOINT to a valid checkpoint path before submitting." >&2
  exit 1
fi

cd /gpfs/home1/rvalperga/icil-rlbench

torchrun --standalone --nproc_per_node=4 \
  /gpfs/home1/rvalperga/icil_jax_rlbench/icil/pretrain_direct_regression_perceiver_encoder_decoder_ddp.py \

