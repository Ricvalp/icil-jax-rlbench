# AGENTS.md

Instructions for coding agents working on the `fast-weight-ttt` branch.

## Hard Constraints

- Keep the repository standalone. Do not import from `icil`,
  `icil_jax_query_memory`, `diagnostics`, or MetaWorld code.
- The legacy direct-regression pretraining, parameter-MAML, and memory-MAML paths
  were intentionally removed. Do not restore them unless the user explicitly
  asks.
- Keep implementation under `icil_jax_rlbench/` unless external tooling is
  explicitly requested.
- Do not add old PyTorch checkpoint conversion unless explicitly requested.
- Avoid destructive Git operations and never revert unrelated user changes.

## Project Purpose

This branch tests whether one or a few demonstrations can produce useful
gradient-based test-time adaptation for robot imitation. It starts with a
controlled hidden-goal state benchmark and moves to RLBench only after the state
diagnostic gates pass.

The main method uses a small fast-weight module, full second-order gradients, a
key-value reconstruction WRITE objective, and an outer action-imitation READ
objective. FOMAML and support action-BC WRITE are ablations.

## Scientific Invariants

- Support influences query predictions only through an explicit fast-state
  update. Do not add direct support cross-attention, support FiLM, task labels,
  language, or support-token initialization of READ memory to the main path.
- Query prediction accepts only slow parameters, fast state, and query
  observation. Query actions must never enter the query encoder or READ path.
- Support and query are different episodes with different starts but the same
  hidden task latent.
- Reset fast state to meta-learned `W0` at every task boundary. Adapt on support,
  then freeze and reuse that state for declared query episodes.
- Support WRITE losses train adaptation but are not added to the outer objective.
- Full second-order differentiation is the main method. `first_order=True` is a
  named FOMAML ablation and must visibly remove the expected meta-gradient paths.
- Never save transient task-adapted fast state in checkpoints or optimize it with
  the slow optimizer.

## Gate Order

1. Verify benchmark integrity and train a competent query-only policy.
2. Pass Gate 1: ordinary support adaptation improves independent held-out query
   episodes.
3. Pass Gate 2: fixed-meta-batch loss, finite-difference, JIT, reset, and update
   direction checks.
4. Pass Gate 3: held-out goals show support-specific adaptation under matched
   controls and multiple seeds.
5. Only then implement and run end-to-end RLBench TTT.

Do not interpret a passing Gate 2 as evidence of held-out adaptation.

## Directory Structure

- `data/hidden_goal.py`: controlled environment, task splits, normalization,
  samplers, and integrity checks.
- `models/fast_weight_ttt.py`: state-policy KVB WRITE and gated READ.
- `train/ttt_step.py`: full/FOMAML meta-objectives and train steps.
- `train/ttt_runner.py`: training, validation, provenance, checkpoints, resume.
- `eval/ttt_state_gate1.py`: ordinary-adaptation upper bound.
- `eval/ttt_state_gate2.py`: implementation-correctness overfit gate.
- `eval/ttt_state.py`: held-out closed-loop support controls.
- `models/ttt_supernode.py`: self-contained RLBench visual register encoder.
- `models/robotics_actions.py`: RLBench action encoding and component losses.
- `data/h5_cache.py`: retained dense-cache reader for the later visual phase.

## Controlled Benchmark Contract

The hidden latent is a 2D goal excluded from the four-dimensional observation
`[x, y, gripper, phase]`. The action is normalized planar delta plus a binary
gripper target. Train, validation, and test goals are disjoint. Support and query
episode IDs and initial states must not overlap. Fit normalization from training
goals only.

Meta-batches retain explicit task, support-demo, support-time, query-demo, and
query-time axes. Do not collapse those semantic axes in the sampler.

## RLBench Contract

The later visual path uses dense H5 episodes with `xyz`, `valid`, `state`, and
`action`, plus optional `rgb` and `mask_id`. Infer dimensions with
`RLBenchCacheStore.infer_dims()`.

Use space-time supernodes, separate state/action/time token types, and small event
registers. Query encoding must not accept demonstrated query actions. Use delta
translation, continuous 6D rotation with geodesic loss, and gripper BCE.

## Coding Guidelines

- Prefer explicit functional JAX code and static shapes inside JIT/pmap.
- Keep shape comments near nontrivial reshapes, scans, and vmaps.
- Add abstractions only when they remove real duplication or clarify semantics.
- Keep source ASCII and comments concise.
- Use `apply_patch` for manual edits.
- Do not commit generated outputs, checkpoints, caches, W&B data, or bytecode.

## Verification

Always run:

```bash
python -m compileall -q icil_jax_rlbench tests
PYTHONPATH=. pytest -q
rg -n "^(from|import) (icil|icil_jax_query_memory|diagnostics|metaworld)(\\.|\\s|$)" \
  icil_jax_rlbench tests
```

For model or training changes, also run Gate 2 or a one-step TTT smoke job. For
visual changes, run `tests/test_ttt_supernode.py` and
`tests/test_robotics_actions.py`.
