# Dr Moagi 3D Inward Document Swarm

## Status

Experimental deterministic reference model.

This module operationalizes a bounded 3D document-generation and self-optimization abstraction around a logical `4,843 x 4,843 x 4,843` lattice. The lattice capacity is exactly:

```text
4,843^3 = 113,590,865,107 logical token-ops per virtual tick
```

The word **virtual** is essential. The model does **not** claim that Python, Jarvis-X, a GPU, or a current transformer physically executes `113,590,865,107` full transformer tokens per nanosecond. `virtual_ns` is model-time used for architecture and scheduling experiments.

## Relationship to the existing inward optimizer

Jarvis-X already contains `hf_model/inward_self_optimizer.py`, which performs bounded gradient-driven parameter optimization by projecting model-parameter chunks into 3D control tokens, applying an inward fold, and pulling the resulting displacement back into parameter space.

`src/jarvisx/inward_document_swarm.py` operates at a different layer:

- `hf_model/inward_self_optimizer.py`: model-parameter refinement;
- `src/jarvisx/inward_document_swarm.py`: document workload partitioning, virtual 3D scheduling, configuration-space search, and deterministic fitness evaluation.

They are complementary rather than duplicate mechanisms.

## State

The document swarm state is

```text
S = [
  section_words,
  tokens_per_word,
  parallel_fraction,
  coherence_weight,
  verification_weight,
  novelty_weight,
  compression_weight,
  stability_weight,
  mutation_scale,
  memory_decay,
  refinement_passes,
]
```

The state is encoded into an 11-component bounded latent vector:

```text
Z_k = E(S_k)
```

and decoded back through:

```text
S_k = D(Z_k)
```

## 3D inward fold

For each recursive generation, the current latent state is treated as the center of a local `5 x 5 x 5` search cube:

```text
(x, y, z) in {-2, -1, 0, 1, 2}^3
```

so each generation evaluates exactly:

```text
5^3 = 125 candidate states
```

The axes are assigned operational roles:

```text
X -> partitioning / throughput
Y -> coherence / verification
Z -> compression / stability / recursion
```

A candidate is produced by:

```text
Z_k^(x,y,z) = Z_k + DeltaZ(x,y,z)
S_k^(x,y,z) = D(Z_k^(x,y,z))
```

Every decoded state is bounded by legal parameter ranges.

## Virtual document execution

For a document containing `W` words:

```text
N_tokens = ceil(W * tokens_per_word)
N_sections = ceil(W / section_words)
```

The model budgets three kinds of logical work:

```text
O_plan   = N_sections + 1
O_gen    = N_tokens
O_verify = N_tokens * refinement_passes
```

Total raw logical work is:

```text
O_raw = O_plan + O_gen + O_verify
```

The effective scheduled work is:

```text
O_eff = O_raw / parallel_fraction
```

with logical cube capacity:

```text
C = side^3
```

and logical ticks:

```text
ticks = max(1, ceil(O_eff / C))
```

The reference model defines virtual time as:

```text
T_virtual = ticks + 0.1 * refinement_passes
```

This is a scheduling metric, not wall-clock latency.

## Fitness and recursive selection

Every candidate is scored on bounded synthetic metrics for coherence, verification, stability, compression, novelty, memory burden, virtual speed, and logical utilization.

The fitness function is:

```text
J =
  0.24 * coherence
+ 0.23 * verification
+ 0.17 * stability
+ 0.11 * compression
+ 0.07 * novelty
+ 0.10 * speed
+ 0.04 * utilization
- 0.04 * memory_cost
```

The winning local state is:

```text
S_(k+1) = argmax J(S_k^(x,y,z))
```

and becomes the center of the next 3D search cube.

The closed loop is therefore:

```text
S_k
 -> Encode
 -> 3D expand
 -> Evaluate 125 candidates
 -> Select
 -> Decode
 -> S_(k+1)
 -> repeat
```

with the fixed-point target:

```text
S* ~= M(S*)
```

where `M` is the complete encode/fold/evaluate/select/decode operator.

## Million-word reference case

The default state for a one-million-word document starts at approximately:

```text
1,000,000 words
1,300,000 tokens
1,000 sections
3 refinement passes
```

At the requested logical lattice size, this workload fits inside one logical capacity tick in the synthetic scheduler. The model then adds the explicit refinement overhead and reports `1.3 virtual ns`.

Again, this number is **not** a measured physical generation time.

## Deterministic 3D addressing

No literal 113-billion-cell Python tensor is allocated. A logical cell index `n` maps reversibly to coordinates:

```text
x = n mod side
y = floor(n / side) mod side
z = floor(n / side^2)
```

and:

```text
n = x + side * (y + side * z)
```

This preserves the full logical cube without requiring physical RAM proportional to the virtual volume.

## Verification

Run the focused tests with:

```bash
pytest -q tests/test_inward_document_swarm.py
```

The tests verify:

- exact `4,843^3` capacity;
- reversible 3D addressing;
- exactly 125 local search directions at radius 2;
- bounded million-word scheduling;
- recursive optimization that does not reduce aggregate fitness;
- coordinate validation.

## Boundary between model and hardware

This reference deliberately separates three quantities:

1. **logical capacity**: number of virtual slots in the abstract 3D cube;
2. **virtual time**: scheduler model-time;
3. **wall-clock throughput**: what a real CPU/GPU/accelerator actually measures.

A future hardware backend must benchmark wall-clock execution independently and must not infer physical throughput from `logical_capacity` or `virtual_ns`.
