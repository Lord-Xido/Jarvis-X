# ADR-008: QSOL Kinetic 3D Research Surface

- **Status:** Proposed
- **Date:** 2026-08-16
- **Decision scope:** browser research visualization

## Context

Jarvis-X now has a bounded system control plane around the canonical deterministic VM. The QSOL kinetic processor introduces a complementary equation-phase-space model with position `X`, velocity `V`, residual `F`, residual force `G`, memory `Omega`, energy accumulators and an explicit `Pi_Lambda` projection operation.

The system also needs a geometrically explicit way to inspect those dynamics without weakening the trusted execution boundary.

## Decision

Add `apps/qsol-kinetic-3d/` as a self-contained browser research surface.

The visual mapping is:

```text
X       -> 3D state position and trajectory
V       -> velocity vector
F       -> residual vector
G       -> residual-force vector
Omega   -> toroidal memory geometry
R0      -> equation energy H_eq
PiLambda -> bounded projection cube
```

The browser interpreter may execute only its in-memory QSOL demonstration instruction set. It is not a canonical Jarvis-X executor and cannot directly commit system state.

## Geometric model

The authoritative research geometry exists in `R^3` before projection. The display applies a camera transform and perspective projection to the exact coordinates. The `Pi_Lambda` demonstration boundary is the cube `[-10,10]^3`; the Omega memory visualization uses the torus parameterization

```text
T(theta,phi) = ((R+r cos(phi)) cos(theta),
                (R+r cos(phi)) sin(theta),
                 r sin(phi)).
```

The visual trajectory is derived directly from the same `X` state updated by the embedded instruction interpreter.

## Core invariants

1. **Visualization is not authority.** Browser state is tentative research state.
2. **The canonical runtime is unchanged.** `jarvisx.system_runtime` remains the governed path to authoritative task state.
3. **No privileged side effects.** The QSOL surface has no network, shell, filesystem, market, medical, infrastructure or device capability.
4. **Projection remains explicit.** `PROJ_LAMBDA` is visible as a bounded state projection rather than an implicit clamp.
5. **One state drives all views.** Registers, trajectory, vectors, energy and memory geometry are rendered from the same interpreter state.
6. **Logical geometry does not imply physical allocation.** The visual model is a bounded browser simulation.

## Consequences

### Positive

- Kinetic equation dynamics become inspectable as a coherent 3D state-space system.
- QSOL assembly instructions and geometry stay synchronized.
- The existing deterministic production boundary is preserved.
- The surface is portable because it has no package or CDN dependency.

### Trade-offs

- The browser interpreter is intentionally small and illustrative.
- Perspective rendering is a visualization layer, not a proof of latent information geometry.
- Browser timing is not a real-time systems guarantee or performance benchmark.

## Validation

A repository conformance test checks that the app remains self-contained and retains the required QSOL, `Omega`, `PROJ_LAMBDA`, 3D canvas and trust-boundary markers.

## Promotion

Promote this ADR to **Accepted** after the pull request passes repository CI and the merged app is verified on `main`.
