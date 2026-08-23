# Dr Moagi 3D Geometry Self-Optimizer

This runtime operationalizes the time-dependent Dr Moagi parametric geometry as a bounded, topology-preserving self-optimization loop.

## Refined geometry

The authoritative geometric state is

```text
θg = (R, Rmin, χ, ω, α, β, γ, δ, κ, λ)
```

with

```text
R(t) = Rmin + (R - Rmin) exp(-χ t)
q    = γ v sin(u/2 + β t)
r    = R(t) + v cos(u/2 + ω t)

x = r cos(u + α t) - q sin(u + α t)
y = v sin(u/2 + ω t) + δ sin(κ u + λ t)
z = r sin(u + α t) + q cos(u + α t)
```

`κ` is restricted to integer spatial modes so the Möbius seam remains continuous:

```text
P(0, v, t) = P(2π, -v, t)
```

The secondary perturbation also uses a half-angle phase so it does not break the seam.

## True inward motion

The earlier time-dependent geometry rotated and twisted but did not necessarily move inward. The refined law introduces a monotone radius schedule:

```text
R(t) -> Rmin as t -> infinity
```

for `χ > 0`.

## Geometry objective

The optimizer measures the geometry rather than assigning a synthetic quality score. The objective combines:

- seam RMS error;
- local area-element non-degeneracy;
- area coefficient of variation;
- temporal speed RMS;
- acceleration RMS;
- bounded inward contraction;
- bounded frequency cost;
- an expressivity floor that discourages the trivial zero-motion solution.

The score is an internal engineering objective, not an external SOTA benchmark.

## Bounded self-optimization

The parameter search is a 27-node local lattice:

```text
(shape, kinetics, inward) ∈ {-1, 0, +1}^3
```

The center is the incumbent and the remaining 26 nodes are counterfactual candidates.

- **shape** changes perturbation/vertical-wave amplitudes;
- **kinetics** changes twist/orbit/perturbation frequencies;
- **inward** changes contraction rate and radius floor.

Every candidate is measured in isolation. Promotion requires:

1. lower objective than the incumbent by the configured minimum;
2. seam RMS below the topology tolerance;
3. no excessive regression of the minimum area element;
4. a finite score.

The invariant is:

```text
PROVISIONAL GEOMETRY != AUTHORITATIVE GEOMETRY
```

until `PI_GEOM` accepts the candidate.

## Run tests

```bash
node --test apps/dr-moagi-geometry/test_core.mjs
```

## Browser runtime

Open `index.html` from a static server. The app provides:

- a live Three.js rendering of the authoritative manifold;
- direct parameter controls;
- measured geometric metrics;
- a 27-node candidate evaluation view;
- `AUTO OPTIMIZE` for one bounded geometry epoch;
- explicit accept/hold transaction trace.

The optimizer changes parameter values only. It does not rewrite source code, execute host commands, or infer that internal objective improvement is state-of-the-art performance.
