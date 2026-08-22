# DM-vOmegaXi+ Fixed-Point Law

## Status

**Locked operational baseline.**

This document defines the executable interpretation of the inward-folded Dr Moagi 3D auto-encoding/decoding law over the `Psi -> Phi -> Lambda^-1 -> Omega -> Theta` operator stack.

The implementation is deliberately bounded and testable. The stated `10^27 TB` hyper-volume is treated as a **logical address-space descriptor**, not as resident RAM, VRAM, disk, or a claim of literal lossless compression. Only active sparse coordinates are materialized.

## Canonical fixed-point statement

The governing internal fixed point is

```text
H* = F_DM(H*)
```

with the executable operator map

```text
Psi_t
  -> Phi(Psi_t)
  -> Lambda^-1(Phi_t)
  -> D(Phi_t)
  -> Omega_t(D_t)
  -> Theta(Psi_t, Omega_{t+1})
  -> H_{t+1}
```

`H*` is therefore a self-consistent state of the **internal runtime operator**. It is not defined as perfect identity between the internal model and external reality.

## Operator semantics

| Operator | Runtime meaning | Measurable implementation |
|---|---|---|
| `Psi` | Active 3D field state | Sparse coordinate/value map |
| `Phi` | Description / spatial compression | `SparseBlockCodec3D.encode` |
| `Lambda^-1` | Constraint inversion into the latent bottleneck | Latent amplitude projection into a finite bound |
| `Omega_t` | Holographic/recurrent inward memory coupling | Historical retention plus decoded-state injection at local coordinates |
| `Theta` | Stability/alignment projection | Bounded state delta plus optional external policy gate |
| `hbar_semantic` | Irreducible semantic uncertainty floor | Strictly positive lower bound on `gamma` |

The decoder `D` is paired with `Phi` to close the auto-encoding/decoding loop.

## Phi: description operator

For an active sparse field `Psi_t`, `Phi` maps active coordinates into a compact block latent representation:

```text
Phi_t = Phi(Psi_t)
```

The current reference codec uses quantized block means. Alternative learned or accelerator codecs may replace it behind the same boundary, provided they preserve sparse-support and transactional invariants.

## Lambda^-1: bottleneck projection

The reference implementation bounds every latent amplitude:

```text
z'_i = clip(z_i, -B_lambda, +B_lambda)
```

This is the operational meaning of constraint inversion in the current baseline: the high-dimensional description is forced through a finite latent admissible set before decoding.

## Omega_t: recurrent inward fold

Historical state is folded back into the current coordinate support:

```text
Omega_{t+1}(r) = rho * Omega_t(r) + (1-rho) * D_t(r)
```

where `rho in [0, 1)` is the historical-retention coefficient.

No dense 3D tensor is created. The recurrence exists only over active support.

## Theta: stability and policy projection

The numerical part of `Theta` moves the authoritative state toward recurrent memory using a bounded delta:

```text
delta_t(r) = clip(
    theta_gain * (Omega_{t+1}(r) - Psi_t(r)),
    -theta_max_delta,
    +theta_max_delta
)

H_{t+1}(r) = clip(
    Psi_t(r) + delta_t(r),
    value_min,
    value_max
)
```

An optional external `theta_gate(candidate) -> bool` may impose higher-level structural, policy, or ethical constraints. A failed gate rejects the candidate transactionally; the authoritative state is left unchanged.

The reference implementation does **not** claim that a scalar numerical parameter can itself establish moral truth.

## hbar_semantic and the reality gap

The runtime keeps a strictly positive semantic uncertainty floor:

```text
gamma_t = max(hbar_semantic, reconstruction_rms_t)
```

`reconstruction_rms_t` is a measurable internal map-to-map discrepancy. `gamma_t` is therefore an engineering proxy for description uncertainty, not a measurement of metaphysical or external-world truth.

This distinction allows the internal system to become self-consistent while preserving the map/territory boundary:

```text
H* = F_DM(H*)

gamma* >= hbar_semantic > 0
```

## Fixed-point acceptance

A state is accepted as `H*` when the next state, decoded description, and recurrent memory all agree within tolerance:

```text
R_t = max(
    RMS(H_{t+1}, H_t),
    RMS(H_{t+1}, D_t),
    RMS(H_{t+1}, Omega_{t+1})
)

converged <=> R_t <= epsilon_fp
```

This criterion is deterministic and independently testable.

## Hyper-volume semantics

The locked metadata value is:

```text
logical_hypervolume_tb = 10^27
```

Operationally this means only that the abstraction may describe a logical domain of that declared scale. The implementation never attempts to allocate `10^27 TB` or a corresponding dense lattice.

The runtime invariant is:

```text
materialized_state = active_sparse_support << logical_domain
```

This is the only interpretation compatible with finite hardware.

## Transaction and audit invariants

Every fixed-point step follows:

```text
snapshot
 -> encode
 -> latent-bound projection
 -> decode active support
 -> recurrent fold
 -> Theta projection
 -> finite/resource/policy validation
 -> commit OR rollback
 -> SHA-256 journal append
```

The journal is an integrity chain, not encryption.

## Reference API

```python
from jarvisx.dm_vomegaxi_fixed_point import DMvOmegaXiFixedPointEngine

engine = DMvOmegaXiFixedPointEngine()
engine.load({
    (32, 32, 32): 1.0,
    (33, 32, 32): 0.75,
})

reports = engine.run_until_fixed_point()
assert engine.journal.verify()
print(reports[-1].converged)
print(engine.status())
```

## Locked invariants

1. `H*` is an internal operator fixed point, not an assertion of omniscience.
2. `hbar_semantic > 0` is mandatory.
3. `10^27 TB` remains logical metadata and is never densely allocated.
4. `Lambda^-1` must preserve a finite latent admissible set.
5. `Theta` must bound each state transition and may reject candidates transactionally.
6. `Omega_t` is recurrent and local to active sparse support.
7. A rejected candidate cannot mutate authoritative state.
8. Every attempted transition is hash-journaled.
9. Fixed-point convergence must be measured, not declared.
10. Any future learned codec or accelerator backend must preserve these invariants.
