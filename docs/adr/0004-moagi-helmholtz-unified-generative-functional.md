# ADR-004: Adopt the Moagi-Helmholtz unified generative functional

**Status:** Accepted  
**Date:** 2026-08-12  
**Extends:** ADR-003

## Context

ADR-003 established a dimensionally consistent sparse volumetric field law for the Dr Moagi Layer 4/5 runtime. The wider Jarvis-X research architecture also requires one contract that connects multimodal conditioning, 3D geometry encoding, conditional generation, geometric refinement, rendering, multimedia archival, inverse inference, and transactional adaptation.

Earlier descriptions mixed several distinct operations: gradient flow was called Hamiltonian flow, a global DCT was identified with MP4, RGB rendering was treated as globally invertible, and uniqueness was claimed without contractivity or convexity assumptions. Those shortcuts are not suitable as canonical operational semantics.

The new functional must preserve the existing Jarvis-X authority boundary: the deterministic VM, policy, transaction and provenance layers remain canonical. Generative or codec subsystems may propose candidate states but cannot bypass validation or silently redefine execution semantics.

## Decision

Jarvis-X adopts the **Moagi-Helmholtz unified generative functional** as the canonical orchestration contract for conditional 3D generation and multimedia archival.

The forward cycle is

```text
M
-> c = Phi(M)
-> z = E_thetae(G)
-> V0 = D_thetad(z, c)
-> V* = Refine_E(V0)
-> F = Render(V*)
-> B = VideoEncode_q(F)
-> A = ContainerMux(B, side_info)
```

The reverse cycle is inference, not an assumed exact inverse:

```text
A
-> (B, side_info) = ContainerDemux(A)
-> F_hat = VideoDecode(B)
-> (z_hat, c_hat) = I_phi(F_hat, side_info)
-> V_hat0 = D_thetad(z_hat, c_hat)
-> V_hat* = Refine_E(V_hat0)
-> cycle verification
```

The complete candidate is then admitted through the existing projection and transaction boundary:

```text
candidate
-> Pi_Lambda
-> validator / shadow evaluation
-> COMMIT or ROLLBACK
-> journal
```

## Geometric objective

The conditional geometry is defined variationally as

```text
V* = argmin_{V in G_adm} [
       E_MH(V)
       + (xi/2) ||V - D(E(V), c)||_F^2
     ]
```

with

```text
E_MH = E_membrane
     + E_local
     + E_bend
     + E_area
     + E_barrier.
```

Refinement follows dissipative gradient flow:

```text
dV/dtau = - grad_V E_MH(V)
```

or an explicitly documented numerical approximation. A cotangent Laplacian that depends on geometry must not be differentiated as though it were constant unless the implementation declares a lagged-Laplacian approximation.

## Rendering and codec boundary

Rendering, codec transforms, compressed video bitstreams and file containers are separate contracts:

```text
geometry -> renderer -> frame sequence -> codec -> bitstream -> container
```

An implementation may use MP4 as a container, but MP4 is not defined as a global 3D DCT. Backend-specific codecs, transforms, prediction and rate-control tools remain adapter concerns.

## Reconstruction boundary

Ordinary RGB projection is many-to-one. Jarvis-X therefore does not define a global inverse renderer for arbitrary scenes. Reconstruction uses an inverse inference model and may consume explicit side information such as:

- calibrated camera parameters;
- depth or normal maps;
- segmentation/material metadata;
- latent codes;
- conditioning embeddings;
- geometry or topology sidecars;
- model/version/integrity metadata.

Exact deterministic replay is claimed only when the archive contains sufficient information for that claim. Otherwise reconstruction is an inferred geometry compatible with the decoded observations and declared priors.

## Rate-distortion contract

Archival quality is evaluated as a rate-distortion problem:

```text
J_RD = Distortion(F, F_hat) + lambda_R * Rate(B)
```

Video MSE alone is not a complete codec objective. The selected distortion metric, bit accounting method and operating point must be observable.

## Unified training objective

A compatible training objective may combine:

```text
L_MH =
    w_CD      * L_chamfer
  + w_normal  * L_normal
  + w_edge    * L_edge
  + w_KL      * L_KL
  + w_render  * L_render
  + w_rate    * Rate
  + w_station * ||grad E_MH(V*)||^2
  + w_cycle   * L_cycle
  + w_cond    * L_condition.
```

Training losses do not automatically become runtime coefficients. Runtime and training contracts remain separately versioned.

## Fixed-point and convergence semantics

The orchestration state is

```text
Xi = (M, c, z, G, Theta, Omega, Archive, telemetry)
```

and the bounded transition is

```text
Xi_(t+1) = Pi_Lambda(M_MH(Xi_t)).
```

A fixed point satisfies

```text
Xi* = Pi_Lambda(M_MH(Xi*)).
```

Unique convergence is claimed only when the relevant map is proven contractive on the admissible domain or when another sufficient theorem applies. Otherwise the implementation reports measured convergence/stopping criteria without elevating them into a universal theorem.

## Transactional adaptation

Parameter, architecture, scheduler, codec, renderer, tile and bytecode adaptation is candidate-first:

```text
active
-> candidate
-> benchmark / verify
-> Pi_Lambda
-> COMMIT or ROLLBACK.
```

No research layer may rewrite authoritative code or state before validation.

## Reference implementation

`src/jarvisx/moagi_helmholtz.py` provides a dependency-free orchestration reference with typed protocol boundaries and deterministic conformance components. The conformance archive is JSON, not MP4, and the reference geometry codec is lossless, not a learned compressor. Their purpose is to test orchestration invariants rather than model quality or hardware throughput.

## Required invariants

1. Conditioning, latent, geometry, frame, bitstream and archive states have explicit boundaries.
2. Geometric candidate states are finite and resource bounded.
3. Refinement is gradient flow or a named approximation, not mislabeled Hamiltonian dynamics.
4. Rendering and multimedia coding remain separate operators.
5. Lossy RGB/video archival is not called globally invertible.
6. Exact replay requires sufficient side information.
7. Rate and distortion are both measurable for codec optimization.
8. The original anchor may not be silently overwritten by self-reference.
9. Candidate adaptation is transactional and reversible.
10. Logical scale and measured physical throughput are reported separately.
11. Unique convergence requires explicit sufficient assumptions.
12. Backend adapters cannot redefine the canonical VM authority boundary.

## Consequences

### Positive

- multimodal generation, geometry and codec research now share one end-to-end contract;
- the reverse path becomes a measurable inference problem rather than a false inverse assumption;
- archive side information and rate-distortion accounting become first-class;
- C++/CUDA, neural, browser, DMEB, FPGA and native backends can share one orchestration state machine;
- adaptation remains auditable and rollback-safe.

### Negative

- the unified system has more explicit interfaces and versioned state;
- exact replay may require additional archive side information;
- learned inverse reconstruction requires empirical evaluation;
- convergence and performance claims remain backend-specific and evidence-gated.

## Validation

Canonical promotion requires:

- end-to-end deterministic reference-cycle tests;
- malformed geometry and resource-budget rejection;
- immutable-anchor tests;
- cycle-error telemetry;
- validator rejection with atomic rollback;
- archive-size bounds;
- clear distinction between reference archive fixtures and production codecs;
- CI across supported Python versions.

The detailed operational specification is maintained in `docs/DR_MOAGI_MOAGI_HELMHOLTZ.md`.
