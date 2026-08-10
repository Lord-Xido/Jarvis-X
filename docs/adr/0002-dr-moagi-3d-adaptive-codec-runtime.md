# ADR-002: Adopt the Dr Moagi 3D adaptive codec-runtime as a bounded research architecture

**Status:** Accepted  
**Date:** 2026-08-10

## Context

Jarvis-X requires a precise architectural home for the Dr Moagi continuous 3D auto-encoding/decoding system. The system combines a 3D transform codec, quantization, entropy coding, reconstruction-error feedback, rate-distortion control, persistent statistics, candidate architecture adaptation, inward latent recursion and a virtual macro-step that may summarize up to 1,000,000 logical refinement transitions.

The repository's canonical architecture separates the deterministic VM core from spatial, adaptive and visual research layers. This decision preserves that boundary: the Dr Moagi system is accepted as a canonical **research architecture and contract**, not as proof that the current VM core physically executes one million dense 3D codec cycles per displayed step.

## Decision

Jarvis-X adopts the Dr Moagi 3D adaptive codec-runtime as a Layer 4/5 research subsystem with the following authoritative state abstraction:

```text
Xi_t = [X_t, F_t, Z_t, B_t, X_hat_t, E_t, lambda_t, Theta_t, Omega_t, Sigma_t]
```

Its codec transaction is:

```text
X_t
  -> Encode_Theta(X_t)
  -> Quantize_Delta(F_t)
  -> EntropyEncode_Omega(Z_t)
  -> bitstream B_t
  -> EntropyDecode_Omega(B_t)
  -> Dequantize_Delta(Z_hat_t)
  -> Decode_Theta(F_hat_t)
  -> X_hat_t
  -> Measure(D_t, R_t, J_t)
  -> Update(lambda, Omega)
  -> Evaluate candidate Theta
  -> Lambda projection / validation
  -> Atomic commit
  -> inward 3D re-entry
```

The compact state transition is:

```text
Xi_(t+1) = G_DrMoagi^3D(Xi_t)
```

A macro-step with virtual depth M is defined by function composition:

```text
Xi_(t+1)^macro = (G_DrMoagi^3D)^M(Xi_t^macro)
```

For the current accelerated visualization and research mode:

```text
M = 1,000,000
```

This value denotes logical/virtual refinement depth unless measured execution demonstrates equivalent physical throughput.

### Codec equations

```text
F_t       = E_Theta_t^3D(X_t)
Delta_t   = Delta(q_t)
Z_t       = round(F_t / Delta_t)
B_t       = C_Omega_t(Z_t)
Z_hat_t   = C_Omega_t^-1(B_t)
F_hat_t   = Delta_t * Z_hat_t
X_hat_t   = D_Theta_t^3D(F_hat_t)
E_local_t = X_t - X_hat_t
D_local_t = ||E_local_t||_2^2 / (HWD)
R_t       = |B_t| / (HWD)
J_t       = D_local_t + alpha*D_anchor_t + lambda_R*R_t + gamma*C_t
```

The entropy codec is lossless with respect to the discrete latent representation:

```text
C^-1(C(Z_t)) = Z_t
```

Therefore quantization and model approximation control distortion; entropy coding controls representation length.

### Anchor-preserving self-reference

The evolving working state may feed back into itself, but the original source state remains immutable for drift detection:

```text
X_anchor = X_0
E_anchor_t = X_anchor - X_hat_t
D_anchor_t = D(X_anchor, X_hat_t)
```

The next working state is produced through a bounded inward spatial operator:

```text
X_(t+1) = I_t^3D(X_hat_t)
```

with a contractive geometric form:

```text
r_(t+1) - c_t = s_t R_t^3D (r_t - c_t),  0 < s_t < 1
```

### Adaptive state

Persistent state `Omega_t` is decomposed into explicit domains:

```text
Omega_t = {
  error_history,
  entropy_model,
  rate_distortion_history,
  architecture_history,
  scheduler_history
}
```

Architecture adaptation is transactional:

```text
Theta_candidate = Mutate(Theta_active)
-> benchmark
-> reconstruction test
-> rate test
-> stability test
-> Lambda gate
-> atomic commit or rollback
```

An encoder and decoder participating in one transaction must use the same architecture and entropy-model versions.

### Lambda projection

`Pi_Lambda` is the admissibility gate, not a metaphor. It enforces bounded numerical state, valid serialization, resource ceilings, distortion ceilings, mutation limits, compatible model versions and fail-closed behavior.

### Preferred million-step implementation

The millionfold recursion should occur primarily in compact latent space rather than as one million full dense 3D encode/decode passes:

```text
X -> Encode -> Z_0
              |
              +-> R(Z_0) -> R(Z_1) -> ... -> Z_M
                                            |
                                            +-> Quantize/Compress/Decode -> X_hat
```

Implementations may use exact serial iteration, parallel candidate evaluation, sparse refinement, operator exponentiation/acceleration or learned macro-transition approximation, but must state which method is used and report measured throughput separately from virtual depth.

## Required invariants

1. **Deterministic codec contract:** fixed authoritative inputs and versions produce reproducible discrete latents and bitstreams where deterministic mode is selected.
2. **Entropy round-trip:** `decode(encode(Z)) == Z`.
3. **Version coherence:** encoder, decoder and entropy model versions are bound to the transaction/bitstream.
4. **Anchor preservation:** self-reference never deletes the immutable anchor used to detect generational collapse.
5. **Bounded adaptation:** candidate architecture changes cannot directly mutate active state before validation.
6. **Bounded resources:** resident memory, iteration budget and compute budget are explicit.
7. **Honest acceleration:** `M = 1,000,000` is reported as virtual/logical depth unless supported by measured physical execution.
8. **Fail-closed projection:** invalid, divergent, non-finite or incompatible candidate states are rejected.
9. **Separation of authority:** visualization does not become authoritative compute state merely by representing the process.
10. **Observable operation:** distortion, anchor drift, rate, latency, memory, convergence and mutation decisions are emitted as telemetry.

## Consequences

### Positive

- the Dr Moagi equation gains an executable architectural interpretation;
- the inward 3D visualization maps to a contractive latent-state operator rather than decorative geometry;
- the millionfold mode has a precise virtual-time meaning;
- repeated lossy self-reference is guarded by an immutable anchor;
- adaptation becomes testable, transactional and reversible;
- the research subsystem remains compatible with the deterministic Jarvis-X core boundary.

### Negative

- the complete architecture requires several state domains and explicit versioning;
- a literal dense 1,000,000-cycle implementation is computationally prohibitive for large volumes;
- architecture search and entropy adaptation introduce additional validation and persistence requirements;
- convergence of the self-referential codec does not by itself prove preservation of source information or task utility.

## Validation

This decision is considered operationally validated when an implementation provides:

- deterministic unit tests for encode/quantize/dequantize/decode;
- entropy coder round-trip tests;
- bitstream version and integrity tests;
- local and anchor distortion telemetry;
- bounded latent recursion with configurable `M`;
- explicit measured versus virtual throughput reporting;
- candidate architecture shadow evaluation and rollback;
- `Pi_Lambda` rejection tests for invalid states;
- memory and cycle ceilings;
- reproducible fixtures for fixed-point/convergence experiments.

## Canonical research specification

The detailed systems contract is maintained in:

`docs/research/DR_MOAGI_3D_CODEC_RUNTIME.md`
