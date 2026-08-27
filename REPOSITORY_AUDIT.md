# Repository Audit for Fast-Weight TTT

Audit date: 2026-08-27

Base commit: `187fa8bb353e7bb8cdb80d86651d344083a16d50`

Development branch: `fast-weight-ttt`

## Runtime Snapshot

- Python 3.13.5
- JAX 0.9.0
- Flax 0.12.0
- Optax 0.2.6
- ml-collections 1.1.0
- NumPy 2.1.3
- Checkpoints are standalone pickle payloads containing parameters, optimizer state,
  RNG, step, resolved config, and optional metadata.

Every new TTT run writes a fresh runtime snapshot, exact Git state, task split,
normalizer identifier, parent checkpoint, reset policy, and resolved config. The
values above are only the implementation-time snapshot.

## Legacy Support Paths

The direct-regression policy has four feed-forward support paths:

1. `ContextEncoder.encode_support` tokenizes support point clouds and passes the
   resulting visual tokens to the action decoder.
2. `TrajectoryTokenizer` embeds demonstrated support actions and concatenates them
   with visual support tokens.
3. `SupportSummaryHead` pools visual and trajectory tokens into a summary used by
   FiLM/AdaLN decoder blocks.
4. `ClassConditioner` can provide task and variation tokens instead of support.

The decoder exposes these paths through `single_ctx`, `two_ctx`, and
`query_film_support`. These are valid direct-ICIL baselines but are not suitable for
proving that a test-time update is necessary.

## Legacy MAML Semantics

Parameter MAML:

- The inner objective is the same action regression loss used for READ.
- Inner examples are leave-one-support-episode-out query chunks.
- A name or preset mask selects a subtree of policy parameters as fast parameters.
- Full MAML differentiates through all inner gradients.
- FOMAML applies `stop_gradient` to inner gradients before the parameter update.
- The outer objective is evaluated on a separate query episode, although several
  outer chunks can come from that one episode.

Memory MAML:

- Support tokens are encoded once into a memory array.
- The inner action loss updates that array rather than model parameters.
- The adapted memory is passed directly to decoder cross-attention.
- Full and first-order modes differ by the same inner-gradient stop.

Both legacy modes therefore retain a direct support representation at READ time;
neither is an adaptation-only mechanism test.

## Legacy Sampling and Actions

- Pretraining samples distinct support and query episodes from one variation.
- Parameter MAML samples support holdouts for WRITE and one independent query
  episode for READ.
- Cached state/action vectors are `[x, y, z, qx, qy, qz, qw, gripper_open]`.
- The legacy action loss is global MSE or L1 over the raw eight components.

## Regression Status

The new work is isolated from the legacy runner and policy. On the contextual V1
cache, one real-data step passed for each existing mode:

- direct-regression pretraining;
- exact parameter MAML;
- exact memory MAML.

These are execution checks only, not reproduced baselines or scientific results.
