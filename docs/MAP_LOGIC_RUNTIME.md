# Map Logic Runtime

Map Logic is the deterministic mathematical-mapping layer for Jarvis-X. It
converts named numeric state into a bounded toroidal field, evolves that field
through a stable inward recurrence, feeds residual error into bounded memory,
and emits a verification receipt.

The implementation lives in `src/jarvisx/map_logic.py`.

## Operational pipeline

```text
named vectors
    |
    v
canonical encode (tanh)
    |
    v
stable SHA-256 coordinate map
    |
    v
T^3 = Z_N x Z_N x Z_N field
    |
    v
periodic 6-neighbour diffusion/relaxation
    |
    v
residual -> Omega feedback memory
    |
    v
constitutional admission gate
    |
    v
verified deterministic receipt
```

For field state `Psi_t(x)`, core signature `Psi_core`, and periodic
six-neighbour Laplacian `Delta_T`, the raw map update is

```text
Psi' = rho Psi_t + alpha dt Delta_T(Psi_t) + (1-rho) Psi_core
```

where `rho` is the target spectral radius inherited from
`PermeationConstitution`.

Residual memory is

```text
r_t       = Psi_core - Psi'
Omega_t+1 = beta Omega_t + (1-beta) r_t
Psi_t+1   = Psi' + gamma Omega_t+1
```

Every candidate is admitted only when the existing Jarvis-X constitutional
gate holds:

```text
J(Psi_t+1) <= J(Psi_t) + tolerance
rho < 1
hbar_semantic > 0
```

with

```text
J(Psi) = mean ||Psi(x) - Psi_core||^2.
```

Rejected candidates do not replace the accepted field.

## Geometry

Names are deterministically mapped with SHA-256 into

```text
(x, y, z) in Z_N x Z_N x Z_N.
```

All neighbourhood accesses wrap modulo `N`; therefore the execution surface
has no privileged spatial edge. Multiple names that hash to the same cell are
combined by channel-wise averaging.

The dense reference implementation is intentionally bounded and testable. It is
a semantic reference for future sparse, NumPy, Torch, WebGPU, CUDA, or C++
backends rather than a claim that arbitrarily large logical lattices are
materialized in memory.

## Verification receipt

Each execution emits:

- deterministic name-to-coordinate assignments;
- the encoded core signature;
- objective history;
- accepted/rejected fold counts;
- final fixed-point delta;
- saturation fraction;
- feedback-memory RMS;
- theoretical spectral ceiling;
- semantic uncertainty floor;
- a deterministic SHA-256 digest of the final field.

The receipt is marked verified when the accepted objective history is monotone,
the spectral ceiling remains below one, and semantic uncertainty remains
strictly positive.

## CLI

After installation:

```bash
jarvisx-maplogic
```

runs a deterministic smoke execution and prints the receipt as JSON.

## Architectural role

Map Logic sits between symbolic/numeric input and higher-level Jarvis-X
execution surfaces:

```text
description -> encode -> geometry -> state transition -> measurement
            -> residual memory -> verification -> geometry
```

This makes geometry part of the execution state rather than merely a rendering
of an already-computed answer.
