# ADR-0013: Lock the Dr Moagi Engine as a 3D Cognitive-Space Network of Networks

**Status:** Accepted  
**Date:** 2026-08-17  
**Extends:** ADR-002, ADR-003, ADR-006, ADR-0010, ADR-0011, ADR-0012

## Context

Jarvis-X contains several bounded research runtimes: sparse 3D field evolution, codec recursion, geometric diffusion, epistemic verification, and a master transaction membrane. The neural core itself must now have one narrow definition that does not conflate ANN computation with bytecode authority, Helmholtz permeation, rendering, hardware acceleration, or deployment infrastructure.

The canonical Dr Moagi Engine is therefore defined around the 3D auto-encoding/decoding ANN itself.

The engine is not one monolithic network and its latent state is not treated as passive storage. The intended research architecture is a **Network of Networks** operating over a recurrent, multiscale 3D **Cognitive Space**.

"Cognitive" is an architectural term for the shared latent computational state. It does not assert consciousness, subjective experience, AGI, or biological equivalence.

## Decision

Jarvis-X adopts the following as the canonical research definition of the **Dr Moagi Engine**:

> A recursive, multiscale 3D auto-encoding/decoding neural architecture whose shared latent computational manifold is a Cognitive Space and whose transformations are implemented by a dynamically routed Network of Networks.

The compact neural-core operator is

```text
DM_3D = D_3D o R_C^* o E_3D
```

with

```text
Z_t = E_3D(X_t)
C_t^(0) = Lift(Z_t, G_t, Omega_t, A_t, H_t)
C_t^(m+1) = Pi_Lambda_C[R_C(C_t^(m))]
C_t^* = FixedPoint(R_C, C_t^(0); epsilon_fp, I_max)
X_hat_t = D_3D(C_t^*)
E_t = X_t - X_hat_t
```

The fixed point is an operational stopping criterion, not a truth guarantee:

```text
||C_t^(m+1) - C_t^(m)|| <= epsilon_fp
```

or the configured iteration ceiling is reached.

## Cognitive Space

The Cognitive Space is the typed recurrent latent state

```text
C_t^3D = (Z_t, G_t, Omega_t, A_t, H_t)
```

where:

- `Z_t` is the volumetric latent field;
- `G_t` is the relational/topological graph or neighborhood structure;
- `Omega_t` is recurrent ANN memory;
- `A_t` is attention/routing state;
- `H_t` is the bounded set of active hypotheses/candidate latent interpretations.

A concrete tensor realization may use

```text
Z_t in R^(B x C_z x D_z x H_z x W_z)
```

and a multiscale pyramid

```text
C^(0) <-> C^(1) <-> ... <-> C^(L)
```

so local, regional, and global representations can exchange information without flattening the entire volumetric state into one undifferentiated vector.

## Network of Networks

The neural core is composed from specialist subnetworks. A conforming implementation may include:

```text
N_local       local 3D convolutional refinement
N_regional    regional aggregation/message passing
N_global      global context integration
N_attention   spatial/semantic routing
N_memory      recurrent latent memory
N_prediction  bounded candidate prediction
N_correction  reconstruction/error correction
N_decoder     volumetric reconstruction heads
```

The set is extensible, but every participating network must declare its input/output type and may only combine states through compatible typed transforms.

The generic Cognitive-Space update is

```text
C_(m+1) = Pi_Lambda_C[
    C_m
    + T_L(N_local(C_m))
    + T_R(N_regional(C_m))
    + T_G(N_global(C_m))
    + T_A(N_attention(C_m))
    + T_M(N_memory(Omega_t, C_m))
    + T_P(N_prediction(C_m))
    - T_E(N_correction(E_t, C_m))
]
```

where every `T_*` is an explicit transport/lift into the declared Cognitive-Space component being updated. This prevents direct addition of incompatible tensors, logits, parameter gradients, rendered pixels, bytecode words, or provenance records.

## Dynamic routing

The Network of Networks may use sparse or mixture-of-experts routing. For expert networks `N_k`, routing weights satisfy a declared normalization rule such as

```text
w_k >= 0
sum_k w_k = 1
```

and the routed update may be

```text
Y = sum_k w_k N_k(X).
```

Routing may depend on latent state, 3D location, memory, uncertainty, or task conditioning, but resource use must remain explicitly bounded.

## 3D locality and relational geometry

A local latent element may be represented as

```text
z_i = (h_i, p_i, omega_i)
```

with feature state `h_i`, spatial coordinate `p_i`, and recurrent memory `omega_i`.

Neighbor interaction may use

```text
m_ij = phi(h_i, h_j, p_i - p_j)
h_i' = psi(h_i, sum_j m_ij, omega_i).
```

This provides explicit computational 3D geometry without claiming that ordinary ANN hidden states are physically embedded in Euclidean space.

## Multiscale hierarchy

The preferred architecture is hierarchical:

```text
fine local field
    <-> regional field
        <-> global latent field
```

with explicit down/up transforms between levels. The engine should preserve enough spatial structure in latent space to support volumetric reconstruction, local error correction, and spatially selective computation.

## Auto-encoding/decoding closure

The canonical end-to-end ANN loop is

```text
X_t
 -> E_3D
 -> Cognitive Space C_t
 -> recursive Network-of-Networks refinement
 -> C_t^*
 -> D_3D
 -> X_hat_t
 -> reconstruction residual E_t
 -> encode residual / recurrent correction
 -> next refinement cycle
```

A compatible training objective may combine

```text
L = lambda_rec * L_rec
  + lambda_grad * L_grad
  + lambda_cycle * L_cycle
  + lambda_fp * L_fixed_point
  + lambda_sparse * L_sparse
  + lambda_route * L_route_balance.
```

Training losses and runtime coefficients remain separately typed/versioned.

## Memory boundary

Within the Dr Moagi Engine neural core, `Omega_t` denotes recurrent ANN memory only. Provenance journals, VM audit chains, security logs, and authoritative persistent state remain outside this neural memory and must not be conflated with it.

## Relationship to I AM = I DESCRIBE

The phrase

```text
I AM = I DESCRIBE
```

is retained only as a constitutional interpretation of autoencoding closure:

```text
state -> internal description -> reconstruction.
```

It is not a theorem of consciousness or identity. A fixed point means representational stability under the declared operator; it does not imply factual truth or subjective awareness.

## Architectural boundary

The Dr Moagi Engine ends at the ANN research interface:

```text
Input -> E_3D -> Cognitive Space -> Network of Networks -> D_3D -> Output
```

The following remain downstream adapters or enclosing control planes, not neural-core operands:

- canonical 64-bit bytecode/VM execution;
- `SystemRuntime` authority and audit commit;
- epistemic provenance/authentication infrastructure;
- Helmholtz/Green permeation;
- rendering and multimedia containers;
- native swarm/VRAM accelerators;
- browser/GUI visualization;
- network/filesystem/device/tool side effects.

They may consume or host the ANN, but they may not silently redefine its neural state equation.

## Required invariants

1. **True volumetric latent state:** the reference ANN preserves a declared 3D latent structure rather than treating "3D" as naming alone.
2. **Network-of-Networks composition:** specialist subnetworks communicate through explicit typed interfaces.
3. **Bounded recursion:** recursive refinement has a measured stopping criterion and iteration/resource ceiling.
4. **Same-space updates:** additive updates are transported into compatible Cognitive-Space components before combination.
5. **Parameter/state separation:** `grad_Theta L` updates parameters; latent/state gradients update latent/state unless an explicit transport is declared.
6. **Multiscale locality:** local, regional, and global states have explicit transforms and shapes.
7. **Bounded routing:** active expert count, attention/window extent, and compute budget are observable and bounded.
8. **Memory separation:** ANN recurrent memory is distinct from provenance/audit memory.
9. **No truth-by-fixed-point:** convergence is not epistemic verification.
10. **Honest scale:** virtual spatial extent and symbolic recursion depth are reported separately from resident memory and measured throughput.
11. **No consciousness claim:** the Cognitive Space is a computational architecture term only.
12. **Adapter separation:** bytecode, rendering, permeation, hardware, and authority control remain outside the neural-core equation unless connected through explicit adapters.

## First reference implementation target

The first trainable reference should be deliberately bounded, for example:

```text
input:            B x C x 64 x 64 x 64
encoder pyramid:  32@32^3 -> 64@16^3 -> 128@8^3
latent:           B x 128 x 8 x 8 x 8
refinement:       4-8 recurrent 3D blocks with local convolution + windowed attention + memory
routing:          bounded sparse experts
output:           mirrored volumetric decoder
```

This is a reference target, not a requirement that all implementations use these exact dimensions.

## Validation strategy

The architecture should be benchmarked against named baselines rather than claimed superior by construction. Minimum comparisons should include a conventional 3D autoencoder and, where appropriate, 3D VAE and 3D attention variants.

Measure at least:

```text
reconstruction MSE / L1
3D structural/perceptual quality where defined
geometry-specific error where defined
latent bitrate or representation size
fixed-point residual and actual iterations
VRAM / resident memory
latency / throughput
routing utilization
anchor or cycle drift for recursive experiments
```

Any beyond-baseline or SOTA claim requires measured evidence on a declared dataset and protocol.

## Consequences

### Positive

- the Dr Moagi Engine now has one precise neural-core definition;
- Cognitive Space becomes a typed latent computational substrate rather than an undefined metaphor;
- the Network-of-Networks concept becomes trainable through specialist subnetworks and bounded routing;
- 3D geometry, recurrence, memory, attention, and autoencoding fit one coherent ANN contract;
- downstream VM, security, rendering, and accelerator work remain modular.

### Trade-offs

- the architecture is more complex than a conventional 3D autoencoder;
- multiscale volumetric attention can be expensive and requires sparse/windowed designs;
- fixed-point stability is not guaranteed for arbitrary learned recurrent blocks;
- multiple specialist networks increase optimization, routing, and ablation requirements;
- the architecture must prove practical value empirically.

## Canonical designation

```text
Dr Moagi Engine
= Recursive Multiscale 3D Auto-Encoding/Decoding Cognitive Space
  implemented as a Dynamically Routed Network of Networks.
```
