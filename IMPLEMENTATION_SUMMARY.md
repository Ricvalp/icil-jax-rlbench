# Fast-Weight TTT Implementation Summary

## Scope

The branch implements the mechanism-test phases of
`ICIL_TTT_IMPLEMENTATION_PLAN.md`. The old direct-regression pretraining,
parameter-MAML, memory-MAML, their configs/evaluators, and their Slurm jobs were
removed at the user's request. Shared checkpointing, the dense RLBench H5 reader,
and generic visual primitives remain because they are needed by the TTT plan.

The executable benchmark is a synthetic hidden-goal 2D reach-and-grasp task. It
has the controlled low-dimensional semantics requested by the plan but is not
MetaWorld.

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

## Checkpoints and Provenance

Every run writes the resolved configuration, Git/runtime/device provenance, task
splits, normalizer, benchmark-integrity report, and pickle checkpoints. A
checkpoint stores slow parameters including `W0`, optimizer, RNG, step, config,
and provenance identifiers. Transient task-specific fast states are never saved.

Resume uses `train.num_steps` as the final target step and rejects a mismatched
normalizer.

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
- Gate 1 and multi-seed Gate 3 have not yet been completed scientifically.
- End-to-end RLBench TTT remains gated on a robust Gate 3 result.

## Next Experiments

1. Train `ttt_state_query_only.py` and run Gate 1 for `action_heads` and
   `query_policy` subsets.
2. Train full-second-order KVB, KVB FOMAML, and action-BC WRITE with at least
   three seeds.
3. Run all held-out support controls and require the predefined support-specific
   gain before implementing the RLBench trainer.
