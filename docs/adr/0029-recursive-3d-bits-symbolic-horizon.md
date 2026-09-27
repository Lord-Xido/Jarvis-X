# ADR-029: Recursive 3D bitstream iteration adapter and symbolic septillion horizon

- **Status:** Proposed
- **Date:** 2026-09-21
- **Applies to:** 3D bit addressing, volumetric bytecode adapters, inward recurrence, symbolic iteration metadata, CTR receipts
- **Extends:** ADR-015, ADR-016, ADR-017, ADR-027

## Context

Jarvis-X already contains a bounded sparse volumetric bytecode reference whose
logical geometry can be much larger than its resident active set. The current
architecture also distinguishes symbolic iteration depth from measured physical
execution.

The next integration requirement is to make the linear bitstream-to-3D mapping
explicit and to carry the requested right-associated septillion tower

[
a=10^{24},
qquad
T=a^{(a^a)}
]

without constructing T or presenting it as a hardware throughput measurement.

The integration must preserve one authority boundary rather than introduce a
parallel executor.

## Decision

Jarvis-X adds a bounded adapter:

- `src/jarvisx/recursive_bits3d.py`;
- `tests/test_recursive_bits3d.py`;
- `docs/DR_MOAGI_3D_BITS_ITERATION_LOOP.md`.

The adapter wraps `VolumetricBytecodeVM`. It does not replace it.

### 1. Canonical linear-to-3D mapping

For side N, channels C and byte lane index i,

[
i=(((zN+y)N+x)C+c).
]

For an 8-bit lane, bit b is addressed by

[
j=8i+b.
]

The adapter exposes exact forward and inverse mapping functions and tests the
round trip at boundary indices.

### 2. Default bounded geometry

The reference defaults are

[
N=64,qquad C=4,qquad B=8.
]

Therefore

[
64^3cdot4=1,048,576
]

scalar byte lanes and

[
8,388,608
]

logical raw bits belong to one dense state.

Only non-zero active lanes are resident in the sparse wrapped executor.

### 3. Symbolic power tower

The power tower is stored as a structural value:

```text
a = 10^24
T = a^(a^(a))
```

with a symbolic base-10 rendering

```text
10^(24*10^(24*10^(24)))
```

The implementation SHALL NOT evaluate T.

The symbolic target SHALL NOT modify `max_iterations`, allocate an array of T
states, or be reported as iterations-per-second.

### 4. Physical execution

Each physical iteration remains the existing bounded volumetric cycle:

```text
active 3D state
  -> sparse mask
  -> octree accounting
  -> encode
  -> XYZ mirror fold
  -> mirror union
  -> decode
  -> XOR residual
  -> Hamming evidence
  -> CTR verification
  -> commit OR rollback
  -> permeation receipt
  -> recur
```

The adapter adds deterministic bit/byte addressing before this cycle and a
bounded sparse serialization view after it.

### 5. Authority boundary

The adapter SHALL NOT commit around or outside the wrapped VM.

The law is

[
S_{t+1}
=
V_t,Pi_Lambda(S^{m cand}_{t+1})
+
(1-V_t)S_t.
]

For this adapter, `VolumetricBytecodeVM` supplies the bounded candidate,
codec verification, state hash and commit decision. ADR-016 remains the
repository-wide transaction authority and ADR-017 remains the canonical
operational autoencoding contract.

### 6. Inward recurrence

Repeated execution is represented as

[
S_{t+r}=mathcal M^r(S_t)
]

for finite physical r bounded by `max_iterations`.

An implementation may later add a jump operator

[
J_{k+1}=J_kcirc J_k
]

only if it demonstrates a closed composable representation, cheaper
composition than replay, and verification of the resulting state.

The algebraic identity alone is not evidence of a speedup.

## Evidence boundary

This ADR establishes a software contract for:

- exact linear bit/byte to 3D addressing;
- sparse residency;
- bounded recursive execution;
- symbolic tower metadata;
- residual/Hamming accounting;
- CTR-gated promotion;
- deterministic state receipts.

It does not establish:

- septillion-tower physical iterations per second;
- a septillion-tower speedup;
- physical cubic memory;
- hardware electromagnetic computation;
- convergence of arbitrary learned nonlinear models;
- semantic correctness, AGI or consciousness.

Measured performance requires a named implementation, named hardware,
benchmark protocol and reproducible result.

## Consequences

### Positive

- makes the bit-level 3D mapping executable and testable;
- reuses the existing verified volumetric executor rather than duplicating it;
- carries astronomical iteration intent without numerical materialization;
- separates logical horizon, physical work and measured throughput;
- supplies a direct bridge from byte streams into the Dr Moagi sparse 3D family.

### Costs

- the current lane width is fixed to eight bits;
- the current inward operator remains the bounded mirror-union reference;
- the symbolic tower is metadata until a separately validated composable
  jump-ahead backend exists;
- dense export remains bounded and is not a mechanism for materializing huge
  virtual volumes.

## Validation

Conformance tests SHALL cover:

- default 64^3 x 4 state size;
- scalar-index address round trips;
- bit-index address round trips;
- symbolic tower rendering without evaluation;
- sparse zero elision;
- bounded convergence of the wrapped reference cycle;
- physical iteration-budget enforcement;
- malformed/out-of-range input rejection.

## Provenance

This ADR belongs to the Dr Moagi family and inherits the attribution and
provenance policy defined by ADR-015 and
`docs/attribution/PERMEATION_MANIFEST.md`.
