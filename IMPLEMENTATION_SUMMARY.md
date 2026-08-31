# Fast-Weight TTT Implementation Summary

## Scope

The branch implements the mechanism-test phases of
`ICIL_TTT_IMPLEMENTATION_PLAN.md`. The old direct-regression pretraining,
parameter-MAML, memory-MAML, their configs/evaluators, and their Slurm jobs were
removed at the user's request. Shared checkpointing, the dense RLBench H5 reader,
and generic visual primitives remain because they are needed by the TTT plan.

The repository now has two state paths. The synthetic hidden-goal 2D task is
retained for fast autodiff and mechanism checks. The first scientific policy
experiment uses the independent `phi-mujoco` `metaworld_ml1_reach` integration
for collection, processed data, task binding, and closed-loop evaluation.

## MetaWorld Policy Path

The policy project loads complete ML1 Reach episodes through the public
`phi_mujoco.offline` API and groups them with the integration's declared task
index. The sampler preserves explicit task, demonstration, and time axes and
draws support and query from different episodes. The 40 training, 10
validation, and 50 test task IDs are never replaced by an episode-level split.

Observation and action standardization is fitted only from episodes belonging
to training task IDs. The complete statistics, source episode indices, cache
SHA-256, and a normalizer identifier are stored in every checkpoint. Goals
remain provenance only and are never included in a policy batch.

The query-only policy consumes 39D hidden-goal state and predicts standardized
4D continuous actions. Cartesian and gripper components use separate Huber
terms. The `phi-mujoco` policy adapter denormalizes and projects each action at
the environment boundary.

Gate 1 performs ordinary support-action BC updates on an isolated parameter
copy, then evaluates fresh episodes from the same held-out goal. It implements
matched no-update, correct-support, wrong-task-support, shuffled-action, and
observations-only conditions. Closed-loop seeds are fresh relative to the
cache and identical across conditions. No task-specific adapted parameters are
written back to the query-only checkpoint.

MetaWorld KVB training starts from the query-only slow parameters but creates a
new AdamW state and TTT step counter. Each meta-batch retains explicit task,
support-demo, support-time, query-demo, and query-time axes. Support episodes
drive sequential KVB WRITE updates; the outer loss is computed only from
different query episodes. Full mode differentiates through all WRITE updates.

The held-out evaluator supports one, two, and four demonstrations and the full
no/correct/wrong/shuffled/partial-information/random-update control matrix. It
uses fresh rollout seeds matched across conditions, resets to `W0` at every
goal/condition boundary, freezes the resulting fast state across query
rollouts, and delegates simulation and optional video artifacts to public
`phi_mujoco` evaluation contracts.

## Adaptation Mechanism

The state policy contains separate support and query encoders, learned key/value/
query projections, a small linear or MLP fast model, meta-learned `W0`, positive
learned per-tensor update rates, clipped fast gradients/updates, and a gated READ
residual.

KVB WRITE is:

```text
(support observation, action, transition)
  -> support encoder -> normalized K,V
  -> MSE(fast_model_W(K), V)
  -> differentiable update of W
```

READ is:

```text
query observation -> query encoder -> Q -> fast_model_W(Q)
                  -> gated residual -> action heads
```

The query API has no support or query-action argument. Full mode differentiates
the outer imitation loss through all WRITE updates. FOMAML stops each computed
fast gradient before applying it, removing the key/value/support-encoder
meta-gradient pathways. Support WRITE loss is not part of the outer objective.

## Benchmark and Evaluation

The hidden 2D goal is shared by support and query but absent from observations.
Support and query starts and episode IDs differ. Train, validation, and test goals
are disjoint, and normalization is fit on training goals only.

Integrity checks cover split overlap, episode/start overlap, oracle success,
initial-query goal leakage, support-endpoint goal recovery, and support outer-loss
masking.

The held-out evaluator compares matched query starts under no update, correct
support, wrong-task support, shuffled actions, shuffled time, observations only,
actions only, duplicated support, and random updates matched in norm.

Gate 3 summaries report pooled Wilson intervals and task-paired bootstrap
intervals for success gain, offline-loss reduction, and final-distance
reduction. A separate visualization entry point replays deterministic held-out
tasks and writes matched trajectory plots and animations, support traces,
action-change plots at identical observations, sequential WRITE diagnostics,
fast-tensor deltas, vector fields, and the underlying arrays. Privileged goals
are used only to annotate these artifacts and never enter policy inference.

## Checkpoints and Provenance

Every run writes the resolved configuration, Git/runtime/device provenance, task
splits, normalizer, benchmark-integrity report, and pickle checkpoints. A
checkpoint stores slow parameters including `W0`, optimizer, RNG, step, config,
and provenance identifiers. Transient task-specific fast states are never saved.

Resume uses `train.num_steps` as the final target step. MetaWorld KVB resume
restores the checkpoint's scientific config, optimizer state, JAX RNG, and both
sampler RNG states; only runtime controls are overridable. It rejects a
mismatched cache, normalizer, model description, or training contract.

## RLBench Bridge

The gated visual prerequisites include component-specific translation, 6D
rotation/geodesic, and gripper losses; local space-time supernodes with positive
learned spatial/time bandwidths; occupancy diagnostics; separate point,
proprioception, demonstrated-action, and time token types; small event registers;
and query encoding with no query-action input.

The visual supernode path is self-contained and does not depend on the removed
legacy context encoder. There is not yet an RLBench TTT sampler, trainer, or
online evaluator because the plan requires Gate 3 first.

## Diagnostic Status

- The implementation test suite covers benchmark integrity, reset/carry,
  structural support isolation, outer masking, full/FOMAML gradient paths,
  finite differences, eager/JIT consistency, deterministic updates, pmap
  agreement, checkpoint contents, action geometry, and visual registers.
- Gate 2 previously reduced its fixed meta-batch loss from `0.508807` to
  `0.002243` (99.56%). A forward WRITE reduced query loss while reversing that
  update increased it.
- The MetaWorld task-aware loader, query-only trainer, policy interface, Gate 1,
  KVB trainer, and held-out KVB evaluator are implemented. Focused tests cover
  one-step full-second-order training, checkpoint contents, adapted policy
  inference, and exact resume semantics.
- A real 2,400-episode query-only run and Gate 1 evaluation were completed;
  ordinary correct-support adaptation materially exceeded no update on the
  validation goals.
- Two synthetic KVB optimization seeds show support-specific held-out
  adaptation. This is a mechanism result, not the final MetaWorld Gate 3.
- Multi-seed MetaWorld KVB training and held-out validation/test evaluation are
  the next scientific experiments.
- End-to-end RLBench TTT remains gated on a robust Gate 3 result.

## Next Experiments

1. Train `metaworld_ml1_reach_kvb.py` from the competent query-only checkpoint
   with at least three optimization seeds.
2. Select settings only on the 10 validation goals using the complete support
   control and support-count sweep.
3. Run matched FOMAML and support-BC WRITE ablations, then evaluate the frozen
   choices once on the 50 untouched test goals.
