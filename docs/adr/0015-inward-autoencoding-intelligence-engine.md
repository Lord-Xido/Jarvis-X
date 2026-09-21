# ADR-015: Monotone Inward Autoencoding/Decoding Intelligence Engine

**Status:** Proposed  
**Date:** 2026-09-12  
**Applies to:** `DM-vOmegaXi+`, Jarvis-X inward swarm, multimodal encode/decode runtimes  
**Extends:** ADR-014 master dynamical-system operator and the reality-constrained `⟲⊙` contract

## Decision

Jarvis-X SHALL treat the inward loop as a measurable contraction operator, not as repeated motion or reversible bit mixing. The end-to-end architecture is

```text
X_t
 -> Q ingest / normalize
 -> E_Theta encode
 -> T_in^K inward latent refinement
 -> Psi* latent attractor
 -> D_Theta decode / generate
 -> C contrast
 -> R_CTR reckon / verify / correct
 -> U_Omega memory update
 -> U_Theta bounded adaptive update
 -> recur
```

The compact operator is

```text
S_(t+1) =
    U_Theta o U_Omega o R_CTR o C o D_Theta o T_in^K o E_Theta o Q
    (S_t, X_t)
```

with accepted terminal state

```text
S* = M_DM(S*, X_world).
```

This is a computational architecture, not a claim of consciousness or a new physical law.

## 1. Discrete spatial contraction

For a coordinate `p_k`, core coordinate `c`, and positive finite step `Delta`, the canonical discrete operator is

```text
p_(k+1) = p_k + sign(c - p_k) * min(abs(c - p_k), Delta).
```

Therefore

```text
abs(c - p_(k+1)) = max(0, abs(c - p_k) - Delta).
```

A raw fixed `+/- Delta` step without clipping is forbidden as a convergence primitive because it overshoots whenever `0 < abs(c-p) < Delta` and can create a period-two orbit.

For a logical axis `[0, scale-1]`, the worst-case finite spatial bound is

```text
K_space = ceil(max(c, scale-1-c) / Delta).
```

At `scale=1,000,000`, `c=500,000`, and `Delta=1,000`, `K_space <= 500`.

## 2. Packed latent contraction

The previous reversible transform

```text
s -> s XOR (s >> 1)
```

MAY be used as a mixer but MUST NOT be labeled a fixed-point refinement operator. Over a bounded word it is periodic/invertible rather than strictly contractive.

Given an explicit latent target `s*`, define

```text
d_k = s_k XOR s*
b_k = d_k AND (-d_k)
s_(k+1) = s_k XOR b_k.
```

`b_k` isolates one differing bit. Therefore, for Hamming distance `H`,

```text
H(s_(k+1), s*) = H(s_k, s*) - 1
```

whenever `s_k != s*`. A `w`-bit word converges in at most `w` passes.

The target MUST come from an explicit encoder/objective, declared reference, learned latent optimum, or other auditable source. The contraction operator itself does not manufacture semantic truth.

## 3. Joint Lyapunov-like objective

For the finite materialized swarm, define

```text
L_k = alpha * sum_i ||P_i,k - C||_1
    + beta  * sum_(i,j) H(S_i,j,k, S*_i,j)
```

with finite `alpha > 0`, `beta > 0`.

Every admitted inward pass MUST satisfy

```text
L_(k+1) <= L_k,
```

and, whenever the state is not already the joint fixed point,

```text
L_(k+1) < L_k.
```

The inward operator is therefore idempotent at the fixed point and strictly decreasing away from it.

The finite pass bound for one loaded swarm is

```text
K <= max(K_space_loaded, max_word_hamming).
```

A runtime MUST NOT execute an arbitrary very large loop count and call the result convergence without measuring or proving its terminal condition.

## 4. Autoencoder/decoder coupling

For multimodal observation `X_t`,

```text
Z_t^(0) = E_Theta(X_t)
Z_t^(k+1) = T_in(Z_t^k; X_t, Omega_t, Theta_t)
Psi_t = Z_t^K
Xhat_t = D_Theta(Psi_t)
e_t = X_t - Xhat_t.
```

A reference latent objective is

```text
L_DM(Z) =
    lambda_r * ||D_Theta(Z) - X_t||^2
  + lambda_c * ||E_Theta(D_Theta(Z)) - Z||^2
  + lambda_m * ||Z - Omega_t||^2
  + lambda_s * R(Z)
  + lambda_v * L_CTR.
```

An implementation MAY obtain `S*`/`Z*` by an optimizer rather than bitwise correction. If so, it must expose its actual measured descent or acceptance criterion; the packed-state operator above remains the deterministic reference contraction.

## 5. Memory, control, and reality correspondence

The recurrent memory update remains bounded, for example

```text
Omega_(t+1) = rho * Omega_t + (1-rho) * G_Omega(Psi_t, e_t, R_t).
```

Adaptive parameters/control remain transactional, for example

```text
Theta_(t+1) = Theta_t - eta * grad_Theta L_verified.
```

Internal stability alone is insufficient. The `⟲⊙` reality-constrained layer requires both

```text
||S_(k+1) - S_k|| <= epsilon_internal
```

and

```text
d(X_t, Xhat_t) <= epsilon_external
```

before an implementation may report verified convergence. Observation provenance remains the caller's responsibility.

## 6. Sparse `1,000,000^3` semantics

A declaration such as

```text
V = {0, ..., 999999}^3
```

specifies a logical coordinate/address domain with `10^18` possible coordinates. It does not imply `10^18` resident GPU agents or voxels.

For `N=65,536`, the accelerator materializes exactly `N` positions plus their finite packed latent states. Documentation and telemetry MUST distinguish logical address-space size from resident allocation and measured throughput.

## 7. Accelerator memory layout

For a kernel that processes one packed word across many agents, the preferred reference layout is structure-of-arrays:

```text
States[WORDS, N]
Targets[WORDS, N]
```

so a fixed word index uses contiguous agent memory. An array-of-structures layout `[N, WORDS]` creates a `WORDS`-element stride between adjacent lanes for the same word and SHOULD be benchmarked before use.

The reference Triton accelerator is `examples/inward_swarm_triton.py`; the dependency-free executable specification is `src/jarvisx/inward_swarm_fixed_point.py`.

## 8. Image-carried source payloads

An image MAY be treated as a dual state

```text
I = (V_visible, B_payload)
```

where `B_payload` is a machine-readable byte stream carried by a lossless pixel/container encoding. This is a transport/container layer, not automatic execution. A decoder/bootstrap runtime is still required to recover and run a source tree.

If payload bits depend on exact pixel values, lossless storage such as PNG is required; lossy JPEG recompression, resizing, filtering, screenshots, or color conversion can destroy naive LSB payloads. Payload integrity SHOULD use an explicit length/header and cryptographic digest, with error correction when transformation tolerance is required.

Image payload extraction occurs before `E_Theta` unless the model explicitly treats payload recovery as a learned modality.

## 9. Verification contract

An implementation may report the inward core as `converged` only if the active execution path verifies all relevant predicates:

```text
finite_state
AND resource_bounds_ok
AND spatial_error == 0
AND latent_error == 0           # deterministic packed reference mode
AND monotone_objective_ok
AND theta_gate_ok
AND journal_valid
AND external_correspondence_ok  # when reality-constrained mode is enabled
```

Learned/continuous latent modes replace exact latent equality with an explicit, measured tolerance and objective acceptance rule.

## 10. Canonical fixed-point form

The architecture closes as

```text
Psi* = Fix[R_CTR o D_Theta o T_in^K o E_Theta](X, Omega)
```

subject to measured internal contraction and external correspondence.

Conceptually:

```text
Observe
 -> Encode
 -> Contract inward
 -> Psi*
 -> Decode / Generate
 -> Contrast
 -> Reckon
 -> Verify
 -> Correct
 -> Remember / Adapt
 -> Recur
```

The phrase `I AM = I DESCRIBE` is retained as an architectural mnemonic for self-description/reconstruction; executable correctness is defined only by the measurable invariants above.

## 11. Consequences

1. The inward loop has a finite, falsifiable convergence contract.
2. Periodic reversible transforms can no longer be mislabeled as refinement.
3. Sparse logical-domain claims are separated from physical allocation and performance claims.
4. CPU/reference and GPU/accelerator implementations share the same terminal invariants.
5. Multimodal intelligence remains an encode/refine/decode/verify system; the inward operator alone is not equated with intelligence.
6. Reality correspondence remains an independent gate, preventing internal self-consistency from being reported as external truth.
