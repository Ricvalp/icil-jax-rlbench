# Fast-Weight Test-Time Training for Robot Imitation

This branch implements the adaptation-only fast-weight TTT program in
`ICIL_TTT_IMPLEMENTATION_PLAN.md`. The former direct-regression pretraining,
parameter-MAML, and memory-MAML paths have been removed deliberately.

The first policy experiment uses hidden-goal MetaWorld ML1 Reach through the
independent `phi-mujoco` interoperability layer. A self-contained synthetic
hidden-goal benchmark remains available for fast mechanism and autodiff tests.
End-to-end RLBench TTT remains gated on successful held-out-latent state
experiments.

## Setup

The project uses Python 3.11 and [`uv`](https://docs.astral.sh/uv/). The
synthetic-only environment is:

```bash
uv sync
```

The MetaWorld path uses the independent `phi-mujoco` checkout as an editable
sibling rather than copying or nesting that repository:

```text
Robotics/
  icil-jax-rlbench/
  phi-mujoco/
```

Install that local integration and optional W&B support with:

```bash
uv sync --group metaworld --extra wandb
uv run --group metaworld pytest -q
```

This installs the `metaworld_ml1_reach` data/evaluation adapter. ICIL owns the
task-aware support/query sampler, train-task normalization, JAX policy,
adaptation, and experimental controls; simulator and collection logic remains
in `phi-mujoco`.

On a CUDA 12 machine, add the JAX CUDA wheels:

```bash
uv sync --group metaworld --extra cuda12 --extra wandb
```

`uv sync` is exact: repeat `--extra cuda12` on every later sync of this
environment, or `uv` will remove the optional CUDA plugin and JAX will fall
back to CPU. Verify the backend with:

```bash
uv run --frozen python -c "import jax; print(jax.default_backend(), jax.devices())"
```

All commands below can then run through `uv run`; the package is installed
editable in its own environment.

## Core Invariant

Support demonstrations may affect a query prediction only by updating an
explicit transient fast state. The prediction API receives:

```text
(slow parameters, adapted fast state, query observation) -> query action
```

It has no support argument, task label, language input, or query-action input.
Fast state is reset to meta-learned `W0` at each task boundary, adapted on
support, and then frozen while evaluating independent query episodes.

## Current Components

- `phi-mujoco` ML1 Reach cache loading with declared 40/10/50 task splits.
- Train-task-only observation and action normalization stored in checkpoints.
- Query-only MetaWorld behavior cloning with 39D hidden-goal state and 4D
  continuous actions.
- Ordinary support-BC Gate 1 adaptation and matched fresh closed-loop rollouts.
- Full-second-order MetaWorld KVB meta-training initialized from the query-only
  policy, with matched FOMAML and support-BC WRITE configs.
- Held-out MetaWorld support-count and support-information controls with offline
  and fresh closed-loop evaluation.
- Synthetic 2D hidden-goal reach-and-grasp benchmark with disjoint train,
  validation, and test goals.
- Query-only base policy for the ordinary-adaptation upper bound.
- Full-second-order key-value binding (KVB) WRITE objective.
- Matched FOMAML and support action-BC WRITE ablations.
- Closed-loop no/correct/wrong/shuffled-support controls.
- Fixed-meta-batch gradient and update-direction diagnostics.
- RLBench delta-translation, 6D-rotation, geodesic, and gripper losses.
- RLBench space-time supernodes and small event registers for the later visual
  phase.

## Experiment Sequence

Create a balanced ML1 Reach cache with `phi-mujoco`. This example collects 24
independent demonstrations for each of the 100 declared tasks:

```bash
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
raw="$PWD/datasets/raw/metaworld_ml1_reach-$stamp"
processed="$PWD/datasets/processed/metaworld_ml1_reach-$stamp"

uv run --frozen --group metaworld phi-mujoco collect metaworld_ml1_reach \
  --output-directory "$raw" \
  --episodes 2400 \
  --seed 0 \
  --quiet --progress

uv run --frozen --group metaworld phi-mujoco convert \
  metaworld_ml1_reach "$raw" \
  --output-directory "$processed"

uv run --frozen --group metaworld phi-mujoco validate-cache "$processed"
export PHI_MUJOCO_ML1_REACH_CACHE="$processed"
```

Train the support-free query-only policy over the 40 training goals:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --frozen --group metaworld python \
  -m icil_jax_rlbench.train_metaworld_query_only \
  --config=icil_jax_rlbench/configs/metaworld_ml1_reach_query_only.py
```

Run Gate 1 on the 10 validation goals. Each condition uses identical fresh
query starts, while offline support and query demonstrations are disjoint:

```bash
MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --frozen --group metaworld python \
  -m icil_jax_rlbench.eval_metaworld_ml1_reach_gate1 \
  --config=icil_jax_rlbench/configs/eval_metaworld_ml1_reach_gate1.py \
  --config.checkpoint_path=/path/to/query_only/last.pkl
```

The default Gate 1 run adapts all parameters for 100 SGD steps from four
support demonstrations and compares no update, correct support, wrong-task
support, shuffled actions, and observations-only controls. Override
`inner_steps`, `inner_lr`, and `adapt_subset` for the planned 50--200 step and
parameter-subset sweeps.

Once Gate 1 establishes that support is informative, initialize MetaWorld KVB
meta-training from the competent query-only checkpoint. This starts a new TTT
step counter; it does not reuse the query-only optimizer:

```bash
export ICIL_ML1_REACH_QUERY_CHECKPOINT=/absolute/path/to/query_only/last.pkl

XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --frozen --group metaworld python \
  -m icil_jax_rlbench.train_metaworld_ttt \
  --config=icil_jax_rlbench/configs/metaworld_ml1_reach_kvb.py
```

The default meta-batch contains four training goals. Each goal contributes two
support and two different query demonstrations with padded shapes
`[task, demonstration, time, feature]`. The KVB writer processes support in
16-transition segments. The outer objective is action imitation only on the
query demonstrations, and full second-order gradients pass through every
WRITE update.

Checkpoints are written below
`outputs/metaworld_ml1_reach_ttt/<run-id>/`. Evaluate the 10 validation goals
before touching the 50-task test split:

```bash
CKPT=/absolute/path/to/outputs/metaworld_ml1_reach_ttt/<run-id>/last.pkl

MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --frozen --group metaworld python \
  -m icil_jax_rlbench.eval_metaworld_ml1_reach_ttt \
  --config=icil_jax_rlbench/configs/eval_metaworld_ml1_reach_ttt.py \
  --config.checkpoint_path="$CKPT"
```

The default evaluation sweeps one, two, and four support demonstrations over
all support controls. Every condition uses the same fresh closed-loop seeds;
fast state resets to `W0` independently for each goal and condition, then stays
frozen across that condition's query rollouts. Results are written to
`eval_outputs/metaworld_ml1_reach_ttt/<checkpoint>_<timestamp>/summary.json`.
After selecting all settings on validation, run the untouched test split once:

```bash
--config.split=test
```

To record a small qualitative sample without changing evaluator code, add:

```bash
--config.max_tasks=3 \
--config.support_counts='(2,)' \
--config.conditions='("no_update","correct_support","wrong_task_support")' \
--config.save_rollout_artifacts=True \
--config.record_video=True
```

The matched training ablations are:

```text
icil_jax_rlbench/configs/metaworld_ml1_reach_fomaml.py
icil_jax_rlbench/configs/metaworld_ml1_reach_action_bc.py
```

The synthetic benchmark remains useful for Gate 2 and fast-weight
implementation checks:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python -m icil_jax_rlbench.eval.ttt_state_gate2 \
  --config=icil_jax_rlbench/configs/ttt_state_gate2.py
```

## Synthetic Diagnostic

The controlled synthetic hidden-goal mechanism experiment remains the fast
autodiff and implementation diagnostic. MetaWorld is the policy experiment;
the synthetic result is not a substitute for held-out MetaWorld adaptation.

Train the main KVB method with at least three independent optimization seeds.
Keep `benchmark.seed` fixed so every seed uses the same disjoint task-latent
split:

```bash
for seed in 0 1 2; do
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run --frozen python -m icil_jax_rlbench.train_ttt_state \
    --config=icil_jax_rlbench/configs/ttt_state_kvb.py \
    --config.benchmark.seed=0 \
    --config.train.seed="$seed"
done
```

Each run writes `last.pkl` below `outputs/ttt_state/<run-id>/`. Select
hyperparameters using only the 12 validation goals:

```bash
CKPT=/absolute/path/to/outputs/ttt_state/<run-id>/last.pkl

XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --frozen python -m icil_jax_rlbench.eval.ttt_state \
  --config=icil_jax_rlbench/configs/eval_ttt_state.py \
  --config.checkpoint_path="$CKPT" \
  --config.split=validation
```

The default sweep evaluates 1, 2, and 4 support demonstrations, 20 matched
query episodes per latent, and all support controls. It resets to `W0` for
every task and condition, then freezes the adapted fast state across that
task's query episodes. Results are written below
`eval_outputs/ttt_state/<checkpoint>_<timestamp>/summary.json`. Once choices
are fixed, run the same command once with `--config.split=test`.

Train and evaluate the matched FOMAML, action-BC WRITE, and no-WRITE/no-READ
ablations by replacing the training config with, respectively:

```text
icil_jax_rlbench/configs/ttt_state_fomaml.py
icil_jax_rlbench/configs/ttt_state_action_bc.py
icil_jax_rlbench/configs/ttt_state_query_only.py
```

### Gate 3 Visualization

Visualization is a separate replay pipeline so plotting and video code does
not enter the trainer or quantitative evaluator. Install its direct
dependencies on a CUDA machine with
`uv sync --group visualization --extra cuda12`, then render a few selected
validation tasks using the same checkpoint. CPU-only installations can omit
`--extra cuda12`.

```bash
SUMMARY=/absolute/path/to/eval_outputs/ttt_state/<evaluation>/summary.json

XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --frozen --group visualization python \
  -m icil_jax_rlbench.visualize_ttt_state \
  --config=icil_jax_rlbench/configs/visualize_ttt_state.py \
  --config.checkpoint_path="$CKPT" \
  --config.evaluation_summary_path="$SUMMARY" \
  --config.task_ids='(0,1,2)' \
  --config.support_count=2 \
  --config.query_episodes=3
```

Each task directory contains matched trajectory overlays and MP4 animation,
support trajectories and goals, action changes at identical query
observations, per-WRITE-step diagnostics, per-tensor fast-weight deltas, policy
vector fields, and the underlying arrays in `trajectory_data.npz`. Goals are
read from privileged benchmark metadata only for rendering and never enter the
policy. The top-level `evaluation_summary.png` summarizes the quantitative
support-count sweep.

## Acceptance Signal

The intended Gate 3 result is:

```text
correct support > no update approximately wrong or shuffled support
```

Before moving to RLBench, require either at least 15 points of absolute
closed-loop success improvement or at least a 30% relative query-loss reduction,
with confidence intervals excluding zero and substantially smaller gains from
wrong or shuffled support.

## Outputs and Resume

Each MetaWorld query-only or KVB run creates a unique directory under its
`train.output_dir` and writes:

- `resolved_config.json`;
- `provenance.json`;
- `task_splits.json`;
- `normalization.json`;
- `dataset_integrity.json`;
- `training_contract.json` for KVB runs;
- periodic `step_XXXXXXX.pkl` checkpoints and `last.pkl`.

Checkpoints contain slow parameters, meta-learned `W0`, optimizer state, RNG,
configuration, and provenance. They never contain task-specific adapted fast
state. Resume with:

```bash
--config.train.resume_path=/path/to/checkpoint.pkl \
--config.train.num_steps=40000
```

`train.num_steps` is the final target step, not an additional-step count.
KVB resume restores model, adaptation, meta-batch, and optimizer
hyperparameters from the checkpoint. It also restores optimizer state, JAX
RNG, and train/validation sampler RNG states. Only runtime controls such as the
new target step, logging intervals, output directory, cache location, and W&B
settings are taken from the resume command.

## Repository Layout

- `icil_jax_rlbench/data/metaworld_ml1_reach.py`: task-aware phi cache and
  support/query sampler.
- `icil_jax_rlbench/train/metaworld_query_runner.py`: query-only MetaWorld BC.
- `icil_jax_rlbench/train/metaworld_ttt_runner.py`: MetaWorld KVB meta-training,
  exact resume, ledger, and checkpoints.
- `icil_jax_rlbench/eval/metaworld_ml1_reach_gate1.py`: ordinary-adaptation
  upper bound and closed-loop support controls.
- `icil_jax_rlbench/eval/metaworld_ml1_reach_ttt.py`: held-out KVB support
  controls and closed-loop evaluation.
- `icil_jax_rlbench/eval/metaworld_policy.py`: phi policy-interface adapter.
- `icil_jax_rlbench/eval/support_controls.py`: shared support perturbations and
  confidence intervals.
- `icil_jax_rlbench/data/hidden_goal.py`: synthetic controlled benchmark.
- `icil_jax_rlbench/models/fast_weight_ttt.py`: WRITE, READ, and fast-state logic.
- `icil_jax_rlbench/train/ttt_step.py`: meta-objective and JIT/pmap steps.
- `icil_jax_rlbench/train/ttt_runner.py`: training, validation, ledger, and resume.
- `icil_jax_rlbench/eval/ttt_state*.py`: Gates 1-3.
- `icil_jax_rlbench/visualization/`: deterministic Gate 3 rollout capture,
  plots, and animations.
- `icil_jax_rlbench/models/ttt_supernode.py`: visual event-register bridge.
- `icil_jax_rlbench/models/robotics_actions.py`: RLBench action geometry/losses.
- `icil_jax_rlbench/data/h5_cache.py`: retained dense RLBench H5 reader.

## RLBench Cache Contract

The retained reader expects `CACHE_ROOT/task_name/variationN.h5`, with episode
groups containing `xyz`, `valid`, `state`, and `action`; `rgb` and `mask_id` are
optional. Shapes must be inferred from data rather than hard-coded.

There is currently no RLBench TTT sampler, trainer, or online evaluator. Those
belong to the visual phase after Gate 3.

## Verification

```bash
uv run --frozen --group metaworld python -m compileall -q icil_jax_rlbench tests
uv run --frozen --group metaworld pytest -q
rg -n "^(from|import) (icil|icil_jax_query_memory|diagnostics|metaworld)(\\.|\\s|$)" \
  icil_jax_rlbench tests
```

See `IMPLEMENTATION_SUMMARY.md` for implemented gradient semantics, known limits,
and the current diagnostic status.
