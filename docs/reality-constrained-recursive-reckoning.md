# ⟲⊙ — Reality-Constrained Recursive Reckoning

`⟲⊙` is the executable Jarvis-X contract for coupling recursive internal stabilization to an explicit external observation anchor.

It does **not** define an observation as absolute truth. Observation provenance remains the caller's responsibility. The runtime guarantee is narrower and testable: every admitted recursive transition is evaluated against the declared world anchor, and successful convergence requires both internal coherence and external correspondence.

## Canonical invariant

Let `H_t` be the authoritative sparse runtime state and `X_world` the declared observation field.

```text
Generate → Contrast → Reckon → Verify → Correct → Recur
```

The operational fixed-point condition is:

```text
H* = F_DM(H*; X_world)
||H* - F_DM(H*; X_world)|| <= epsilon_i
d(H*, X_world) <= epsilon_e
```

The first inequality is the internal fixed-point condition. The second is the external correspondence condition. Neither is substituted for the other.

## Commit rule

For a generated candidate `H'_t`, the runtime evaluates external residuals before and after correction:

```text
e_before    = d(H_t, X_world)
e_generated = d(H'_t, X_world)
H''_t       = Correct(H'_t, X_world)
e_after     = d(H''_t, X_world)
```

A candidate may commit only when all existing numerical and Theta-policy gates pass **and** correspondence is non-worsening:

```text
e_after <= e_before
```

This allows a state to advance toward reality over multiple bounded recursive cycles instead of requiring an exact world match in a single update.

Convergence is stronger than admission:

```text
internal_converged
AND external_residual <= epsilon_e
```

## Permeation stack

The runtime exposes the invariant across the canonical stack:

```text
Psi → Phi → Lambda^-1 → Omega → Theta → X_world
```

`Psi..Theta` remain the bounded internal operator. `X_world` is not collapsed into the latent model; it remains an external correspondence constraint. This preserves the model/territory distinction already present in the DM-vOmegaXi+ fixed-point runtime.

## Sparse reference implementation

```python
from jarvisx.dm_vomegaxi_fixed_point import DMvOmegaXiFixedPointConfig
from jarvisx.reality_reckoning import (
    RealityConstrainedDMvOmegaXiEngine,
    SparseRealityAnchor,
)

world = {
    (2, 2, 2): 0.25,
    (3, 2, 2): 0.25,
}

initial = {
    (2, 2, 2): 0.75,
    (3, 2, 2): 0.75,
}

anchor = SparseRealityAnchor(
    world,
    tolerance=1.0e-4,
    correction_gain=0.5,
)
engine = RealityConstrainedDMvOmegaXiEngine(
    anchor,
    DMvOmegaXiFixedPointConfig(fixed_point_tolerance=1.0e-4),
)
engine.load(initial)
reports = engine.run_until_fixed_point()

assert reports[-1].internal_converged
assert reports[-1].external_correspondence
assert reports[-1].converged
```

## Telemetry

Each cycle records:

- internal fixed-point residual;
- reconstruction, memory and Theta residuals;
- external residual before generation;
- external residual after internal generation;
- external residual after correction;
- whether reality correction changed the candidate;
- independent internal/external convergence flags;
- authoritative state hash and hash-chain journal head.

`status()` additionally exposes:

```text
glyph: ⟲⊙
operator: Reality-Constrained Recursive Reckoning
reckoning_cycle: Generate, Contrast, Reckon, Verify, Correct, Recur
permeation_stack: Psi, Phi, Lambda^-1, Omega, Theta, X_world
```

## Scientific boundary

Internal convergence alone remains insufficient:

```text
H* = F_DM(H*)  does not imply  H* corresponds to reality.
```

Likewise, a world anchor is only as reliable as its measurement and provenance. `⟲⊙` therefore operationalizes **reality-constrained correction**, not an oracle of truth.
