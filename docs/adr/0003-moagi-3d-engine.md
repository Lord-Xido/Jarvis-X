# ADR-003: Adopt the Moagi-3D recursive multi-channel attractor engine

**Status:** Accepted  
**Date:** 2026-09-07

## Context

ADR-002 established the Dr Moagi 3D adaptive codec-runtime as a bounded research architecture. The next layer formalizes the user-facing four-subsystem blueprint as an executable orchestration runtime: Volumetric Ingestion Fabric (VIF), K-channel Moagi Manifold, Fusion & Routing Matrix (FRM), and Recursive Attractor Controller (RAC).

The design must preserve the repository's existing distinction between mathematical architecture and measured hardware capability. It must also remain bounded, observable and deterministic in its portable reference implementation.

## Decision

Jarvis-X adopts the Moagi-3D Engine with transition

```text
Z_k^(n) = E_k(P^(n))
R_k^(n) = D_k(Z_k^(n))
alpha^(n) = softmax(-tau * MSE(P^(n), R_k^(n)))
F^(n) = sum_k alpha_k^(n) R_k^(n)
P^(n+1) = (1-g)P^(n) + gF^(n)
```

and RAC condition

```text
Delta_n = ||P^(n+1) - P^(n)||_2
halt if Delta_n <= epsilon OR n == N_max.
```

The implementation keeps `P^(0)` immutable as an anchor and emits `||P^(n)-P^(0)||_2` as drift telemetry.

K denotes the number of parallel channels; it is not the per-channel latent dimensionality. Each channel uses a distinct deterministic encoder/decoder instance. Independent seeds create distinct kernels, but strict orthogonality is a separate training/verification requirement and must not be inferred from seed diversity.

## Required invariants

1. deterministic replay for fixed configuration, seeds and input;
2. finite-state checks at input, latent, reconstruction and recursive-state boundaries;
3. normalized non-negative FRM weights;
4. hard `N_max` resource bound in addition to epsilon convergence;
5. immutable source anchor for generational-drift measurement;
6. explicit convergence versus budget-exhaustion telemetry;
7. no claim that the portable C++ runtime is itself HBM/tensor-core/crossbar silicon;
8. no claim that a self-consistent attractor is automatically physical ground truth.

## Consequences

The Moagi-3D blueprint now has an executable reference layer that reuses the existing 3D autoencoder substrate and can be compiled, smoke-tested and regressed in CI. Future GPU/FPGA/ASIC implementations can preserve the same operator while replacing the portable VIF/FRM/RAC realization with physical accelerators.

The bounded feedback gain `g` is an under-relaxation control. It improves operational stability but does not by itself prove global contraction; formal contraction claims require a bound such as `rho(J_T(P*)) < 1` or an equivalent Lipschitz certificate.

## Reference

`docs/research/MOAGI_3D_ENGINE.md`
