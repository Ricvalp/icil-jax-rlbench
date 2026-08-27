# Fast-Weight TTT Implementation Summary

## Implemented Scope

This branch implements the mechanism-test portion of
`ICIL_TTT_IMPLEMENTATION_PLAN.md` as a separate path. The prediction API accepts
only a query observation and an explicit fast state. It cannot accept support
tensors, task IDs, language, or query actions.

The controlled benchmark is a self-contained hidden-goal reach-and-grasp task.
This is the plan's permitted equivalent to MetaWorld and respects the repository
constraint against importing MetaWorld code.

## Architecture

The state policy contains:

- separate support and query encoders;
- learned key, value, and query projections;
- a linear or two-layer MLP fast model;
- meta-learned initial fast weights `W0`;
- a positive learned rate for every fast tensor;
- per-fast-tensor gradient and update clipping;
- a near-zero tanh residual gate;
- separate normalized-Huber translation and BCE gripper heads.

KVB WRITE processes support segments sequentially:

```text
support (observation, action, transition)
  -> support encoder
  -> normalized K and V
  -> MSE(f_W(K), V)
  -> clipped learned-rate update of W
```

READ uses only the query observation and adapted state:

```text
query observation -> query encoder -> Q -> f_W(Q)
                  -> gated residual -> action heads
```

`action_bc` uses the same small fast model but replaces KVB WRITE with the
robotics-aware support imitation loss.

## Gradient Semantics

- Full mode differentiates the query imitation objective through every support
  WRITE update.
- FOMAML stops the computed fast gradient before applying it.
- Consequently, full mode gives key/value/support-encoder parameters a meta-gradient;
  the matched FOMAML mode removes those pathways exactly.
- Query projection, `W0`, learned rates, residual projection, gate, and policy heads
  retain their appropriate direct or first-order pathways.
- Support action losses never enter the outer objective.
- Transient task-specific fast states never enter checkpoints or the slow optimizer.

## Data and Evaluation

`HiddenGoalMetaSampler` returns explicit task, support-demo, support-time,
query-demo, and query-time axes. Support and query starts and episode IDs are
different. The latent goal appears only in metadata.

The benchmark checks:

- disjoint train/validation/test latents;
- no support/query episode or exact-start overlap;
- train-only normalizer fitting;
- reliable oracle expert;
- low linear goal predictability from the initial query observation;
- high goal predictability from support trajectory endpoints;
- zero support outer-loss mask.

Closed-loop evaluation uses matched query starts for:

- no update;
- correct support;
- wrong-task support;
- shuffled actions;
- shuffled temporal order;
- observations only;
- actions only;
- duplicated support;
- random fast updates matched in norm.

## Experiment Ledger and Checkpoints

Every TTT training run writes:

- `resolved_config.json`;
- `provenance.json` with commit, dirty state, dependency/runtime/device snapshot,
  parent checkpoint, adaptation mode, and reset policy;
- `task_splits.json`;
- `normalizer.json`;
- `benchmark_integrity.json`;
- pickle checkpoints containing slow parameters, `W0`, optimizer, RNG, config, and
  provenance identifiers.

Resume semantics are deliberately simple: `train.num_steps` is the final target
step, not an additional-step count. The checkpoint normalizer must match.

## Configs and Commands

Main full-second-order KVB training:

```bash
PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m icil_jax_rlbench.train_ttt_state \
  --config=icil_jax_rlbench/configs/ttt_state_kvb.py
```

Matched ablations:

```bash
# KVB FOMAML
python -m icil_jax_rlbench.train_ttt_state \
  --config=icil_jax_rlbench/configs/ttt_state_fomaml.py

# Support action-BC WRITE
python -m icil_jax_rlbench.train_ttt_state \
  --config=icil_jax_rlbench/configs/ttt_state_action_bc.py

# Query-only base policy
python -m icil_jax_rlbench.train_ttt_state \
  --config=icil_jax_rlbench/configs/ttt_state_query_only.py
```

Gate 1 ordinary adaptation, normally using a trained query-only checkpoint:

```bash
python -m icil_jax_rlbench.eval.ttt_state_gate1 \
  --config=icil_jax_rlbench/configs/eval_ttt_state_gate1.py \
  --config.checkpoint_path=/path/to/query_only/last.pkl
```

Gate 2 fixed-meta-batch correctness:

```bash
python -m icil_jax_rlbench.eval.ttt_state_gate2 \
  --config=icil_jax_rlbench/configs/ttt_state_gate2.py
```

Gate 3 support-control evaluation:

```bash
python -m icil_jax_rlbench.eval.ttt_state \
  --config=icil_jax_rlbench/configs/eval_ttt_state.py \
  --config.checkpoint_path=/path/to/ttt/last.pkl
```

## RLBench Bridge

The gated visual prerequisites are implemented and tested:

- component-specific delta-translation, 6D-rotation/geodesic, and gripper-BCE
  action utilities;
- continuous 6D rotation and XYZW quaternion conversion;
- local space-time supernodes with positive learned spatial/time bandwidths;
- occupancy, assignment entropy, and effective-supernode diagnostics;
- separate point, proprioception, demonstrated-action, and time token types;
- 8-32 style small event-register output;
- query encoding with no query-action input;
- the same encoded KVB WRITE and gated READ functions used by the state policy.

An end-to-end RLBench TTT trainer/evaluator is intentionally not enabled yet. The
plan explicitly requires held-out-latent Gate 3 to pass before spending effort or
compute on the visual phase. The bridge exists so that transition does not require
changing the fast-weight mechanism.

## Verification

- 18 tests pass, covering benchmark integrity, reset/carry, structural support
  isolation, support outer masking, full/FOMAML gradient paths, finite differences,
  eager/JIT consistency, deterministic updates, one-device JIT/pmap agreement,
  checkpoint contents, action geometry, and visual registers.
- Gate 2 reduced one fixed meta-batch from `0.508807` to `0.002243` (99.56%).
- After overfitting that batch, one forward WRITE reduced its query loss from
  `0.002803` to `0.000739`; reversing the update raised it to `0.006383`.
- One-step legacy pretrain, parameter-MAML, and memory-MAML jobs passed on real
  contextual RLBench cache data.
- A legacy online rollout reached simulator launch, but this host's CoppeliaSim
  OpenGL setup failed to load `swrast` and segfaulted before the episode began.

## Known Limits and Deliberate Deviations

- MetaWorld is replaced by the equivalent hidden-goal benchmark because this
  standalone repository prohibits MetaWorld imports.
- Gate 2 passing is an implementation result, not held-out adaptation evidence.
- Gate 1 and multi-seed Gate 3 experiments have not yet been run to scientific
  completion.
- Direct-context and recurrent state baselines remain future experiments; the
  existing RLBench direct-ICIL and legacy MAML paths remain available.
- Legacy online RLBench execution still needs verification in a working
  CoppeliaSim/OpenGL runtime; it was not counted as passing on this host.
- Future-effect, sparse trajectory, long-context TBPTT, hybrid direct-context TTT,
  and end-to-end RLBench TTT are gated on Gate 3 as required by the plan.
- No claim is made from old experiment numbers.

## Recommended First Experiments

1. Train `ttt_state_query_only.py`, then run Gate 1 with `action_heads` and
   `query_policy` subsets.
2. Train full KVB, KVB FOMAML, and action-BC WRITE with at least three seeds.
3. Run the complete support-control evaluator on held-out test latents and require
   the predefined correct-support gain before RLBench integration.
4. Inspect learned rates, per-tensor updates, gate magnitude, gradient alignment,
   and correct/wrong fast-state separation before tuning architecture size.
