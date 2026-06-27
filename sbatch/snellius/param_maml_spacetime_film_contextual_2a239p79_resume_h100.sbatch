#!/bin/bash
#SBATCH --job-name=rlbench_maml_2a239p79_resume
#SBATCH --output=sbatch/snellius/logs/rlbench_maml_2a239p79_resume-%j.out
#SBATCH --error=sbatch/snellius/logs/rlbench_maml_2a239p79_resume-%j.err
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=24:00:00

set -euo pipefail

# Continue the param-MAML support-summary FiLM spacetime run.
# Same-mode resume_config_from_checkpoint=True restores saved model/data/train/MAML
# hyperparameters. Runtime controls below intentionally choose the checkpoint,
# continuation length, output run, optimizer resume, and RNG resume.
RESUME_CHECKPOINT="/projects/prjs1905/robotics/rlbench/icil_jax_rlbench_runs/checkpoints/param_maml/2a239p79/step_0040000.pkl"

if [[ ! -f "${RESUME_CHECKPOINT}" ]]; then
  echo "Missing RESUME_CHECKPOINT=${RESUME_CHECKPOINT}" >&2
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
echo "RESUME_CHECKPOINT=${RESUME_CHECKPOINT}"
srun --gres=gpu:4 nvidia-smi
srun --gres=gpu:4 python -c 'import jax; backend = jax.default_backend(); print(f"JAX backend: {backend}; devices: {jax.devices()}"); assert backend == "gpu", f"Expected JAX gpu backend, got {backend}"'

srun --gres=gpu:4 python -m icil_jax_rlbench.param_maml_direct_regression \
  --config=icil_jax_rlbench/configs/param_maml_supernode_spacetime_support_summary_film.py \
  --config.data.cache_root="${ICIL_CACHE_ROOT}" \
  --config.train.resume_path="${RESUME_CHECKPOINT}" \
  --config.train.resume_config_from_checkpoint=True \
  --config.train.resume_optimizer=True \
  --config.train.resume_rng=True \
  --config.train.num_steps=200000 \
  --config.wandb.enable=True \
  --config.wandb.name=param_maml_spacetime_film_contextual_2a239p79_resume_snellius
