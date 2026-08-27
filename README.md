# Fast-Weight Test-Time Training for Robot Imitation

This branch implements the adaptation-only fast-weight TTT program in
`ICIL_TTT_IMPLEMENTATION_PLAN.md`. The former direct-regression pretraining,
parameter-MAML, and memory-MAML paths have been removed deliberately.

The current executable benchmark is a self-contained hidden-goal state task. It
is not MetaWorld. The RLBench visual encoder and action-loss primitives are
present, but end-to-end RLBench TTT remains gated on successful held-out-latent
state experiments.

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

Run the implementation-correctness gate first:

```bash
PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m icil_jax_rlbench.eval.ttt_state_gate2 \
  --config=icil_jax_rlbench/configs/ttt_state_gate2.py
```

Train the query-only state policy:

```bash
PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m icil_jax_rlbench.train_ttt_state \
  --config=icil_jax_rlbench/configs/ttt_state_query_only.py
```

Run ordinary support adaptation on its checkpoint:

```bash
PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m icil_jax_rlbench.eval.ttt_state_gate1 \
  --config=icil_jax_rlbench/configs/eval_ttt_state_gate1.py \
  --config.checkpoint_path=/path/to/query_only/last.pkl
```

After Gate 1 passes, train the main KVB method:

```bash
PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m icil_jax_rlbench.train_ttt_state \
  --config=icil_jax_rlbench/configs/ttt_state_kvb.py
```

Evaluate held-out goals under matched support controls:

```bash
PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m icil_jax_rlbench.eval.ttt_state \
  --config=icil_jax_rlbench/configs/eval_ttt_state.py \
  --config.checkpoint_path=/path/to/kvb/last.pkl
```

Matched ablations use `ttt_state_fomaml.py` and
`ttt_state_action_bc.py`. Run at least three training seeds before drawing a
conclusion from held-out adaptation.

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

Each training run creates a unique directory under `train.output_dir` and writes:

- `resolved_config.json`;
- `provenance.json`;
- `task_splits.json`;
- `normalizer.json`;
- `benchmark_integrity.json`;
- periodic `step_XXXXXXX.pkl` checkpoints and `last.pkl`.

Checkpoints contain slow parameters, meta-learned `W0`, optimizer state, RNG,
configuration, and provenance. They never contain task-specific adapted fast
state. Resume with:

```bash
--config.train.resume_path=/path/to/checkpoint.pkl \
--config.train.num_steps=40000
```

`train.num_steps` is the final target step, not an additional-step count.

## Repository Layout

- `icil_jax_rlbench/data/hidden_goal.py`: controlled benchmark and meta-sampler.
- `icil_jax_rlbench/models/fast_weight_ttt.py`: WRITE, READ, and fast-state logic.
- `icil_jax_rlbench/train/ttt_step.py`: meta-objective and JIT/pmap steps.
- `icil_jax_rlbench/train/ttt_runner.py`: training, validation, ledger, and resume.
- `icil_jax_rlbench/eval/ttt_state*.py`: Gates 1-3.
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
python -m compileall -q icil_jax_rlbench tests
PYTHONPATH=. pytest -q
rg -n "^(from|import) (icil|icil_jax_query_memory|diagnostics|metaworld)(\\.|\\s|$)" \
  icil_jax_rlbench tests
```

See `IMPLEMENTATION_SUMMARY.md` for implemented gradient semantics, known limits,
and the current diagnostic status.
