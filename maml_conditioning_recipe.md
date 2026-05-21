# Recipe for full second-order MAML in an encoder-decoder transformer with cross-attention conditioning

## Goal
This note summarizes a practical recipe for **full second-order MAML** in an encoder-decoder policy with:
- an encoder that produces **query tokens** and **support memory / support summary**,
- a decoder that predicts action chunks,
- token-level **cross-attention** conditioning,
- optional **AdaLN/FiLM** conditioning.

The core recommendation is:
- use **query / current observation** via **cross-attention**,
- use **support trajectory summary** via **AdaLN/FiLM**,
- optionally use **support memory cross-attention** only when the support side contains rich demo / visual memory tokens,
- keep the inner-loop fast parameters **narrow**.

---

## High-level recommendation

### Conditioning split
Use the following split of conditioning mechanisms:

- **Query / current observation tokens** -> **cross-attention**
- **Support trajectory summary** (or task-like support summary) -> **AdaLN / FiLM modulation**
- **Support memory tokens** -> optional **second cross-attention**, but only if the support side contains rich visual/demo memory

### Why
- **Query conditioning** is the most spatially precise signal. It tells the model **where** the object is now. This should remain token-level and geometry-sensitive, so **cross-attention** is the right mechanism.
- **AdaLN / FiLM** is best for **global conditioning**: diffusion timestep, task embedding, compact support summary, trajectory summary.
- If support is only a compact trajectory summary, a full cross-attention layer in every block is usually not worth the cost.

So the preferred design is **not**:
- query via AdaLN/FiLM only,
- support via heavy cross-attention everywhere.

Instead, use:
- query via cross-attention,
- support summary via AdaLN/FiLM,
- support memory cross-attention only where justified.

---

## Recommended decoder block structure

A good decoder block ordering is:

1. **Self-attention** over action tokens
2. **Query cross-attention** (current observation/history tokens)
3. **Support cross-attention** (optional; rich support memory only)
4. **MLP**

All blocks are modulated by **AdaLN / FiLM** using a conditioning vector built from:
- diffusion timestep embedding,
- support/task summary embedding.

### Preferred conditioning inputs
Let:
- `Q` = query tokens from current observation/history
- `S` = support memory tokens
- `s` = pooled support summary vector
- `t` = diffusion timestep embedding

Then each block receives:
- token inputs: action tokens
- cross-attention memory: `Q` and optionally `S`
- modulation vector: `concat(t, s)` or a fused projection of `(t, s)`

---

## What to train in the inner loop (fast params)

For **full second-order MAML**, start narrow.

### Tier 1: fast state only
Recommended first experiment.

**Fast params**
- writable support memory tokens `M` (or writable support tokens derived from encoder output)
- optionally a learned memory prior `M0`

**Outer / slow params**
- full encoder weights
- full decoder weights
- memory initializer / projection layers

This is the highest-return first step if the goal is adaptation without destabilizing the whole network.

---

### Tier 2: fast state + modulation layers
If Tier 1 is too weak, add only small modulation pathways.

**Fast params**
- writable support memory tokens `M`
- **AdaLN / FiLM generator MLPs** in the **last 1/4 to 1/3** of decoder blocks
- final output modulation / final adaptive norm layer

**Still frozen in the inner loop**
- encoder weights
- self-attention QKV
- query cross-attention QKV
- main FFN weights

This is the strongest default recipe for full second-order MAML.

---

### Tier 3: add tiny adapters on support cross-attention only
Only if Tier 2 still does not move enough.

**Fast params**
- Tier 2
- plus **LoRA / adapters** on the **support cross-attention** of the top decoder blocks
- preferably only on `V` and output projection

**Still not adapted in inner loop**
- query cross-attention weights
- self-attention weights
- encoder weights

This keeps the inner loop targeted at support-memory usage rather than generic visual processing.

---

## What not to adapt in the inner loop first

Do **not** start by adapting:
- encoder weights,
- full decoder FFNs,
- self-attention weights,
- query cross-attention weights.

These options are:
- more expensive,
- less stable under second-order MAML,
- less specifically tied to task adaptation.

If adaptation is useful, the first thing that should move is usually **support memory usage**, not the entire scene-processing stack.

---

## Recommended encoder outputs
The encoder should produce three things:

1. **Query tokens `Q`**
   - current observation / recent history
   - high spatial fidelity

2. **Support memory tokens `S`**
   - rich support/demo tokens if available
   - not necessarily too many; a compact bank is fine

3. **Support summary `s`**
   - pooled summary of support trajectories/demos
   - used for AdaLN/FiLM conditioning

If support is only trajectory-like and not rich visual memory, it is often enough to produce:
- a support summary `s`,
- and skip `S` entirely.

---

## Best conditioning recipe by support type

### Case A: support side is only compact trajectory/task information
Use:
- **Query** -> cross-attention
- **Support summary** -> AdaLN/FiLM only
- **No support cross-attention**

This is cheaper and usually sufficient.

### Case B: support side includes rich demo memory / support visual tokens
Use:
- **Query** -> cross-attention in every block
- **Support summary** -> AdaLN/FiLM in every block
- **Support memory tokens** -> cross-attention in the **top few decoder blocks only**

This gives both:
- precise current-state grounding,
- and access to richer support memory when needed.

---

## Exact recipe I would implement first

### Encoder
- Encode **query/current observation** into query tokens `Q`
- Encode support demos/trajectories into:
  - support memory tokens `S`
  - pooled support summary vector `s`

### Decoder
Each decoder block gets:
- **AdaLN input** from support summary + diffusion timestep
- **query cross-attention** to `Q`
- optional **support cross-attention** to `S`

### Inner-loop fast params
Start with:
- writable support memory tokens `S` (or writable memory tokens derived from support encoding)
- AdaLN/FiLM generator MLPs in the top decoder blocks
- final adaptive norm / final modulation layer

### Outer-loop slow params
- full encoder
- full decoder
- memory initializer / support summary projection
- all remaining slow weights

---

## Concrete block-by-block recommendation

Suppose the decoder has `L` blocks.

### Bottom blocks: query-grounding only
For blocks `0 .. L-4`:
- self-attention
- query cross-attention
- MLP
- AdaLN/FiLM from `(t, s)`

No support cross-attention yet.

### Top blocks: query + support refinement
For blocks `L-4 .. L-1`:
- self-attention
- query cross-attention
- support cross-attention
- MLP
- AdaLN/FiLM from `(t, s)`

These are the blocks where support memory actually refines the action plan.

### Fast parameters in these top blocks
- AdaLN / FiLM MLPs
- optional support cross-attention LoRA on `V` and output projection
- writable support memory tokens

This is the most targeted full-second-order MAML recipe.

---

## Why not use query via AdaLN/FiLM only?

Because query/current observation carries **spatially precise** information.

AdaLN/FiLM is global modulation. It is excellent for:
- timestep,
- class/task embedding,
- compact trajectory summary,
- style / mode selection.

It is **not** the best mechanism for precise scene grounding.

The query path is the part that tells the decoder where the object is now. That should remain token-level and be read via **cross-attention**.

---

## Why support summary is a good fit for AdaLN/FiLM

Support trajectory summaries are often:
- low-entropy,
- global,
- task-like,
- not strongly spatial in the current frame.

That makes them ideal for global modulation.

AdaLN / FiLM can tell the decoder:
- what mode of behavior to use,
- which aspects of the query are relevant,
- how strongly to attend to certain structures.

Then the **query tokens** provide the actual spatial grounding.

---

## Final concise recommendation

If implementing one strong full second-order MAML variant, use this:

### Conditioning
- **Query**: cross-attention in every decoder block
- **Support summary**: AdaLN/FiLM in every decoder block
- **Support memory tokens**: cross-attention only in the **top 3–4 decoder blocks**

### Fast params (inner loop)
- writable support memory tokens
- AdaLN/FiLM MLPs in those top 3–4 blocks
- optional support cross-attention LoRA on `V` + out projection in those same blocks

### Slow params (outer loop)
- full encoder
- full decoder
- memory initializer / support summary projection
- all non-fast weights

This is narrow enough to be tractable and broad enough to actually change behavior.
