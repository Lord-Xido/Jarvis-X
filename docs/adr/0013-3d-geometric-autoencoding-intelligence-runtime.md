# ADR-013: Adopt a bounded 3D geometric autoencoding intelligence runtime

**Status:** Proposed  
**Date:** 2026-09-06  
**Extends:** ADR-003, ADR-006, ADR-007, ADR-010

## Context

Jarvis-X already separates the deterministic VM core from sparse-field, kinetic and adaptive research layers. The next integration step is to give the 3D autoencoding/decoding research path one explicit state model, one geometric transition law and one latency interpretation.

The term **sub-nanosecond** is treated as a target for an innermost local state-transition primitive only. It is not a claim that end-to-end model inference, global memory reconciliation, parameter learning or external I/O completes in less than one nanosecond.

## Decision

Adopt the following Layer-5 research state:

```text
S_t = (X_t, Z_t, V_t, Phi_t, Omega_t, Theta)
```

where `X` is the observed 3D field, `Z` the latent 3D field, `V` latent velocity, `Phi` nested phase state, `Omega` residual memory and `Theta` the parameter set.

The canonical research flow is:

```text
Reality
-> quantize / normalize
-> encode
-> geometric latent state
-> local kinetic update
-> predict
-> decode
-> compare with observation
-> geometric correction
-> residual memory
-> bounded candidate update
-> verify
-> commit | rollback
-> repeat
```

### Geometric state

The latent state is interpreted on a product manifold

```text
M = R^d x T^2 x Omega_3
```

with metric

```text
ds^2 = dZ^T G(Z) dZ.
```

A discrete manifold step is written

```text
V_(k+1) = V_k - dt M^-1 [Gamma V_k + grad_M U(Z_k)]
Z_(k+1) = Exp_(Z_k)(dt V_(k+1)).
```

The dependency-free reference implementation uses the flat Euclidean specialization `G = I`, so `Exp_Z(v) = Z + v`. Curved backends may replace that map only when they define and test their metric, tangent representation and exponential/retraction operation.

### Objective

The research potential is decomposed rather than hidden in one score:

```text
U = U_reconstruction
  + U_cycle
  + U_prediction
  + U_geometry
  + U_spatial
  + U_constraints.
```

A backend may omit terms that are not implemented, but it must report that omission and may not claim the corresponding property.

### 3D locality

Local spatial coupling uses an explicit neighborhood operator. The baseline reference uses the six-neighbour Laplacian

```text
Delta_6 Z[i,j,k] =
    Z[i+1,j,k] + Z[i-1,j,k]
  + Z[i,j+1,k] + Z[i,j-1,k]
  + Z[i,j,k+1] + Z[i,j,k-1]
  - 6 Z[i,j,k]
```

with declared boundary behavior.

### Multi-timescale operation

The runtime has three distinct loops:

```text
micro kinetic loop      tau_mu  : local encode/state/decode/correction primitive
memory/reasoning loop   tau_M   : residual aggregation and bounded global reconciliation
learning loop           tau_L   : parameter candidate generation and admission
```

with

```text
tau_mu << tau_M << tau_L.
```

Parameter learning is not required, or assumed, to run at the micro-kernel period.

### Latency contract

A backend that advertises a micro-kernel target `tau_mu < 1 ns` must report:

- the exact operation boundary being timed;
- clock source and timing method;
- warm/cold conditions;
- resident working set;
- routing and synchronization assumptions;
- percentile distribution, not only a minimum;
- hardware and compiler identity.

The physical propagation ceiling is recorded as

```text
d_max = c * tau_mu.
```

This ceiling is not a performance prediction. Real interconnect propagation, gates, memory and synchronization are slower.

### Transaction boundary

Adaptive changes remain candidate-first under the canonical kinetic runtime:

```text
snapshot
-> propose parameter/topology/mechanics candidate
-> shadow
-> named validators
-> commit | rollback
-> journal
```

No neural, geometric or self-referential layer may bypass the authoritative commit boundary.

## Consequences

1. The 3D geometric intelligence model becomes executable and falsifiable without redefining the VM core.
2. Geometric language must map to explicit state types, metrics and operators.
3. Sub-nanosecond language is restricted to measured local primitives.
4. Large virtual 3D extents remain distinct from resident state and measured throughput.
5. The word `intelligence` remains a research label; capability claims require task-level evidence.
6. Curved-manifold, binary/XNOR-popcount, CUDA, WebGPU, FPGA and custom-silicon implementations remain adapters until independently validated.

## Acceptance criteria

ADR-013 is ready for promotion only when the repository contains:

- a deterministic reference kernel;
- exact six-neighbour boundary tests;
- encode/decode round-trip tests;
- state-transition determinism tests;
- residual-memory tests;
- explicit propagation-bound calculations;
- rejection of invalid sub-nanosecond timing declarations;
- integration with candidate/verify/commit semantics before adaptive parameters become authoritative.
