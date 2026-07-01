#!/bin/bash
#SBATCH --job-name=rlbench_maml_kz71poab_q64_ml
#SBATCH --output=sbatch/snellius/logs/rlbench_maml_kz71poab_q64_ml-%j.out
#SBATCH --error=sbatch/snellius/logs/rlbench_maml_kz71poab_q64_ml-%j.err
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=48:00:00

set -euo pipefail

# Memory-lite Q64 WRITE experiment.
# Compared with the direct Q64 file, this keeps the large WRITE batch but:
# - disables attention-stat materialization,
# - uses one outer query-loss sample,
# - enables cuda_malloc_async to reduce allocator fragmentation.
PRETRAINED_CHECKPOINT="/projects/prjs1905/robotics/rlbench/icil_jax_rlbench_runs/checkpoints/pretrain/kz71poab/step_0300000.pkl"

if [[ ! -f "${PRETRAINED_CHECKPOINT}" ]]; then
  echo "Missing PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT}" >&2
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

export ICIL_CACHE_ROOT="/projects/prjs1905/robotics/rlbench/icil_rlbench/.rlbench_cache_dense_v4_contextual_v1"
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR=cuda_malloc_async

if [[ ! -d "${ICIL_CACHE_ROOT}/contextual_slide_pick_sequence" ]]; then
  echo "Missing contextual training task in ICIL_CACHE_ROOT=${ICIL_CACHE_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${ICIL_CACHE_ROOT}/eval_contextual_slide_pick_sequence" ]]; then
  echo "Missing contextual eval task in ICIL_CACHE_ROOT=${ICIL_CACHE_ROOT}" >&2
  exit 1
fi

echo "CUDA_VISIBLE_DEVICES before srun: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "ICIL_CACHE_ROOT=${ICIL_CACHE_ROOT}"
echo "PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT}"
echo "TF_GPU_ALLOCATOR=${TF_GPU_ALLOCATOR}"
srun --gres=gpu:4 nvidia-smi
srun --gres=gpu:4 python -c 'import jax; backend = jax.default_backend(); print(f"JAX backend: {backend}; devices: {jax.devices()}"); assert backend == "gpu", f"Expected JAX gpu backend, got {backend}"'

srun --gres=gpu:4 python -m icil_jax_rlbench.param_maml_direct_regression \
  --config=icil_jax_rlbench/configs/param_maml_supernode_spacetime_support_summary_film.py \
  --config.data.preload_to_memory=True \
  --config.data.keep_open=False \
  --config.data.task_sampling=variation_power \
  --config.data.task_sampling_alpha=0.8 \
  --config.maml.inner_steps=4 \
  --config.maml.num_inner_queries=64 \
  --config.maml.num_query_loss_samples=1 \
  --config.maml.inner_lr=1e-2 \
  --config.maml.first_order=False \
  --config.maml.inner_grad_clip_norm=1.0 \
  --config.maml.fast_param_preset=film_top \
  --config.maml.fast_param_top_layers=2 \
  --config.train.seed=0 \
  --config.train.num_steps=200000 \
  --config.train.batch_size=4 \
  --config.train.lr=1e-4 \
  --config.train.weight_decay=1e-4 \
  --config.train.grad_clip_norm=1.0 \
  --config.train.loss_type=l1 \
  --config.train.log_every=100 \
  --config.train.ckpt_every=5000 \
  --config.train.prefetch_workers=2 \
  --config.train.prefetch_batches=1 \
  --config.train.resume_path="${PRETRAINED_CHECKPOINT}" \
  --config.train.resume_config_from_checkpoint=True \
  --config.train.resume_optimizer=False \
  --config.train.resume_rng=False \
  --config.wandb.enable=True \
  --config.wandb.name=param_maml_spacetime_film_contextual_kz71poab_q64_memlite_second_order_snellius \
  --config.wandb.prediction_log_every=500 \
  --config.wandb.prediction_num_samples=64 \
  --config.wandb.prediction_num_plots=16 \
  --config.train.log_attention_stats=False
