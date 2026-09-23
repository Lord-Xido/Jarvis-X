# 3D-RADE Evolutionary Production Renderer

`evolution.html` operationalises the closed-loop renderer architecture as a standalone browser runtime with no build step and no external dependencies.

## End-to-end loop

```text
Seed pipeline
  -> population of N parameterised 3D variants
  -> evaluate each variant over a real NxNxN voxel lattice
  -> score fidelity, speed, novelty and geometric validity
  -> select elite variants
  -> crossover + bounded mutation
  -> repeat until convergence
  -> deploy champion
  -> generate synthetic 3D scenes
  -> independent validation gate
  -> continual parameter learning
  -> re-evaluate production champion
  -> periodic micro-evolution
  -> deploy improved champion
```

The optimisation objective is the weighted geometric mean

```text
J(P) = F(P)^0.48 * S(P)^0.14 * N(P)^0.12 * Q(P)^0.26
```

where:

- `F`: voxelwise fidelity against the target 3D density field.
- `S`: measured evaluation speed.
- `N`: parameter-space novelty relative to the current population centroid.
- `Q`: validity/coherence gate that penalises degenerate or unstable fields.

The geometric mean prevents one metric from completely masking collapse in another metric.

## 3D state

The runtime explicitly constructs `N^3` coordinates in `[-1,1]^3` and evaluates each candidate over that volume. The production view is a perspective projection of the champion's volumetric density field; the optimisation itself remains volumetric rather than screen-space.

## Convergence

A champion is considered converged after seven consecutive generations in which

```text
abs(J_t - J_(t-1)) < epsilon
```

The user can tune `epsilon`, population size, voxel resolution and mutation amplitude from the HUD.

## Validation-gated continual learning

Synthetic scenes are never treated as training truth automatically. Each generated scene must pass finite-value, bounds and probe-density checks before it can affect the deployed model.

Accepted scenes drive a small finite-difference parameter update. Every 12 accepted scenes, the deployed champion enters a local micro-evolution search. A candidate replaces production only when its composite fitness is higher than the current champion.

Invariant:

> No generated information becomes training truth without an independent validation gate.

## Run

Open:

```text
apps/3d-rade-swarm/evolution.html
```

Then choose **START** for autonomous operation or **STEP** to advance one kinetic stage at a time.

## Scope

This is a deterministic proof-of-concept architecture for evolutionary 3D rendering and continual optimisation. It does not claim that browser timing is a hardware benchmark or that synthetic validation proves real-world correctness. Production deployment should replace the analytic scene oracle with domain-specific reference data, tests and independent measurements.
