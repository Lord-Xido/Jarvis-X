# ADR-003: Adopt the same-space Dr Moagi field runtime v2

**Status:** Accepted  
**Date:** 2026-08-12  
**Extends:** ADR-002

## Context

The Dr Moagi 3D autoencoding/decoding work now has enough detail to separate four concerns that were previously mixed together: codec compression, volumetric state evolution, spatial residual correction, and bounded adaptive execution.

The previous symbolic form subtracted an encoder output from a decoded volume even though those values inhabit different spaces. The revised architecture requires every term in the field evolution law to inhabit the same volumetric state space. It also makes the fixed Moagi glyph kernel explicit as a six-face-neighbour operator and keeps scalar bottlenecks classified as deliberately lossy or restricted-family representations rather than general invertible encodings.

The deterministic 64-bit Jarvis-X VM remains the canonical authority boundary. Research runtimes may propose state transitions, but they may not silently replace the core bytecode format or bypass transaction, policy, resource, and provenance controls.

## Decision

Jarvis-X adopts the following same-space field law as the canonical Layer 4/5 Dr Moagi volumetric evolution contract.

Let

```text
A_theta = D_theta o E_theta
R_theta(Psi) = (I - A_theta)[Psi] = Psi - A_theta[Psi]
```

Then

```text
dPsi/dt = -alpha R_theta(Psi)
          + lambda * Delta_6 R_theta(Psi)
          + eta * G_moagi * Psi
```

where `Delta_6` is the six-face-neighbour discrete Laplacian and the fixed glyph kernel is

```text
G_moagi(0,0,0) = 1
G_moagi(+/-1,0,0) = -1/6
G_moagi(0,+/-1,0) = -1/6
G_moagi(0,0,+/-1) = -1/6
G_moagi(otherwise) = 0
```

With the convention

```text
Delta_6 Psi = sum(face_neighbours) - 6 Psi
```

the glyph operator satisfies

```text
G_moagi * Psi = -(1/6) Delta_6 Psi.
```

The relationship is recorded explicitly so implementations do not accidentally treat the two operators as independent mathematical primitives.

## Discrete operational form

The executable reference step is

```text
R_n       = Psi_n - A_theta(Psi_n)
H_n       = Delta_6 R_n
P_n       = G_moagi * Psi_n
candidate = Psi_n + dt * (-alpha R_n + lambda H_n + eta P_n)
Psi_n+1   = Pi_Lambda(candidate)
```

`Pi_Lambda` is a concrete admissibility projection. It validates finite values, coordinate bounds, resident-support ceilings, configured value bounds, version compatibility, and any caller-supplied acceptance predicate before commit.

## Sparse physical state

A logical `1000 x 1000 x 1000` field is an address space, not a dense allocation requirement.

The physical runtime operates on an active support

```text
A_n subseteq {0,...,side-1}^3
|A_n| <= max_active_cells
```

with deterministic zero background and, when enabled, a bounded one-cell face-neighbour halo. All operators for one step read a frozen snapshot and write only to a candidate map.

## Anchor and drift semantics

Each run freezes its initial projected field as an immutable anchor:

```text
Psi_anchor = Psi_0
```

Telemetry reports both local reconstruction error and drift relative to the anchor. Self-reference must never erase or overwrite the anchor within a run.

## Codec boundary

The field runtime does not prescribe one neural architecture. A codec must provide

```text
latent = encode(active_field)
reconstruction = decode(latent, requested_support)
```

and may only materialize the requested bounded support.

A scalar latent `z in R` is permitted only when the implementation explicitly declares either:

1. a restricted one-dimensional decoder family, or
2. lossy compression.

No implementation may claim `D o E = I` on arbitrary billion-voxel fields when the latent representation lacks sufficient information capacity.

## Adjoint decoder semantics

When decoder kernels are tied to encoder kernels, `K^T` means the true convolution adjoint: spatial reversal plus input/output channel transpose where channels exist. A plain matrix transpose is not an adequate definition for a 3D convolutional kernel.

## Runtime versus training

The runtime field law and the training objective are separate contracts.

A compatible training objective may be

```text
L = reconstruction_MSE
    + beta * ||z||^2
    + gamma * ||grad(Psi) - grad(Psi_hat)||^2
```

but `beta` and `gamma` do not become runtime PDE coefficients unless a specific implementation defines and validates such a coupling.

## Stability and resource guard

The reference explicit-Euler runtime exposes `dt` and rejects configurations outside a conservative operator-norm budget by default. For a non-expansive codec, the reference guard uses

```text
dt * (2 alpha + 24 lambda + 2 |eta|) <= 1.
```

This is a sufficient engineering guard for the reference model, not a proof of convergence for arbitrary learned codecs. Implementations may replace it with a stronger measured or analytically justified stability test, but may not remove bounded-step validation silently.

## Transaction boundary

Adaptive or self-modifying behaviour is candidate-first:

```text
snapshot
-> encode/decode
-> residual operators
-> candidate field
-> Pi_Lambda
-> optional validator / shadow evaluation
-> COMMIT or ROLLBACK
-> journal telemetry
```

Model, schedule, tile, or bytecode candidates must follow the same rule. Active authoritative code or state is never rewritten before validation.

## Architectural placement

The runtime is integrated as follows:

```text
Layer 0-3: canonical deterministic VM, policy, transactions, provenance
Layer 4:   sparse 3D support and spatial operators
Layer 5:   codec, residual dynamics, inward recurrence, bounded adaptation
Layer 6:   interfaces and visualization
```

The canonical 64-bit VM format remains unchanged by this ADR. Experimental 256-bit, 512-bit, DMEB-32, GPU, FPGA, or native-extension instruction formats are adapters/research targets until separately accepted and tested.

## Required invariants

1. Every additive term in the field evolution equation has the same field type and units.
2. The six-neighbour Laplacian and Moagi glyph stencil are defined with explicit sign conventions.
3. Sparse logical scale is reported separately from resident physical memory.
4. One step reads one frozen snapshot; partial writes are never authoritative.
5. Candidate state is finite, bounded, resource-admissible, and validator-approved before commit.
6. The run anchor is immutable.
7. Codec reconstruction is limited to requested support.
8. Latent information limits are stated honestly.
9. Virtual iteration depth is distinguished from measured physical throughput.
10. Visualization and speculative hardware descriptions cannot silently redefine canonical compute state.

## Consequences

### Positive

- the Dr Moagi equation becomes dimensionally consistent;
- the autoencoder is integrated as a same-space closure operator rather than an incompatible additive latent term;
- the glyph kernel receives exact discrete semantics;
- sparse execution, stability, anchor preservation, and rollback are first-class runtime requirements;
- neural, bytecode, C++, GPU, FPGA, and visualization implementations can share one operational contract.

### Negative

- a single scalar latent can no longer be presented as a general lossless billion-voxel representation;
- positive permeation gain may be anti-diffusive under the chosen Laplacian sign convention, so projection and timestep guards remain necessary;
- learned codecs still require empirical stability and reconstruction validation beyond the reference numerical guard.

## Validation

Acceptance requires an executable reference implementation with tests for:

- configuration and timestep rejection;
- same-space reconstruction residuals;
- exact six-face-neighbour glyph behaviour;
- sparse support confinement;
- immutable anchors;
- candidate rollback;
- deterministic state transitions for deterministic codecs.

The normative operational specification is maintained in `docs/DR_MOAGI_FIELD_RUNTIME_V2.md`.
