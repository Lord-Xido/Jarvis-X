# ADR-013: Add an inward multiparallel multimodal 3D swarm research runtime

**Status:** Proposed  
**Date:** 2026-09-05

## Context

Jarvis-X already contains bounded 3D geometric diffusion, kinetic runtimes, inward auto-encoding research, multimodal media processors, and a separate electromagnetic hardware-mapping specification. The next research step is to connect those ideas through one executable contract without merging experimental semantics into the canonical VM core.

The required architecture has four coupled properties:

- heterogeneous modality adapters encode inputs into a common bounded 3D control chart;
- many modality-tagged hypotheses evolve concurrently rather than as one 3-vector;
- generated/projected outputs are decoded and re-encoded to form an inward fixed-point loop;
- the graph/inward relaxation admits a circuit-level RC analogue while preserving the distinction between virtual semantic geometry and physical electromagnetic fields.

ADR-001 requires spatial and adaptive research layers to remain isolated until their contracts and evidence justify promotion.

## Decision

Add `src/jarvisx/inward_multimodal_swarm3d.py` as a dependency-free research runtime with the following public boundary:

1. `ModalityCodec` adapts one media representation to/from the shared 3D chart.
2. `Particle3D` carries bounded position, shared feature state, modality identity, confidence, time coordinate and source identity.
3. `InwardMultimodalSwarm3D` implements dynamic cross-modal adjacency, local Riemannian task preconditioning, decode/re-encode contraction, graph consensus, optional bounded memory forcing and consensus decoding.
4. `ElectricalAnalogueConfig`, `electrical_rhs` and `electrical_step` expose only the structural RC-network mapping of inward feedback and graph relaxation.
5. The runtime remains independent of Torch, FAISS, transformer libraries and concrete media generators. Those systems integrate through adapters rather than becoming package-level dependencies.

The local kinetic equation is

```text
dz_i/dt =
    - k_task G_i^-1 grad J_i
    + lambda (Phi_i(z_i) - z_i)
    + gamma sum_j A_ij (z_j - z_i)
    + rho (z_memory_i - z_i),
```

with

```text
G_i = I + alpha grad(phi_i) grad(phi_i)^T
Phi_i = E_i o D_i.
```

The circuit analogue is

```text
C dV_i/dt =
    g_phi (V_phi_i - V_i)
    + g_c sum_j A_ij (V_j - V_i)
    + I_ext_i.
```

## Physical boundary

The 3D chart is an algorithmic state space. It is not physical spacetime and is not assumed to have Planck-scale or quantum-gravity structure.

The electrical helper models a bounded circuit-level correspondence. It does not solve Maxwell's equations and cannot establish electromagnetic acceleration, energy efficiency or hardware feasibility without a concrete device/netlist implementation and measurement.

The existing electromagnetic research specification remains the lower-level reference for field, circuit and device semantics.

## Consequences

### Positive

- turns the inward multimodal/swarm equations into executable behavior;
- creates a narrow adapter boundary for text, image, audio, video, geometry, code and data heads;
- keeps the 3D control manifold distinct from high-dimensional media payloads;
- provides deterministic, dependency-free tests for the mathematical contract;
- gives future analog/mixed-signal work a precise RC equation to target;
- preserves the canonical-core isolation required by ADR-001.

### Negative

- the local chart uses Euclidean/tangent approximations for consensus and displacement rather than a global geodesic solver;
- the reference runtime does not itself provide trained modality codecs or high-quality media generation;
- dynamic all-to-all attention is quadratic in particle count;
- explicit Euler integration requires conservative bounds for stable parameter regimes;
- circuit correspondence alone does not prove hardware advantage.

## Validation

The decision is successful when tests demonstrate that:

- the rank-one inverse metric produces the expected Riemannian preconditioning;
- graph coupling reduces consensus error for a simple symmetric swarm;
- a contractive decode/re-encode codec converges toward its fixed point;
- heterogeneous inputs can share a common feature width and decode through multiple heads;
- the RC analogue relaxes coupled voltage differences while preserving symmetry in the unforced pair case;
- invalid bounds and incompatible modality/feature inputs are rejected.

## Promotion criteria

Before this layer can influence canonical VM state, require:

- reproducible concrete modality adapters;
- benchmarked stability and performance against simpler baselines;
- a provenance/transaction integration design;
- resource-accounting limits for large swarms;
- measured hardware evidence for any electromagnetic-compute claim.
