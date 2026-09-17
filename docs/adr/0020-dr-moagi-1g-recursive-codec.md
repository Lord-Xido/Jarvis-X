# ADR-020: Dr Moagi 1 GiB Recursive 3D Bitstream Codec

**Status:** Proposed  
**Date:** 2026-09-17  
**Applies to:** Dr Moagi operational autoencoding equation, sparse 3D codec runtimes, streaming ROM/VM profiles, meta-optimizers, CPU/GPU/distributed backends  
**Extends:** ADR-016, ADR-017, ADR-018 and ADR-019

## Context

ADR-016 establishes typed candidate-first state transitions and atomic commit/rollback. ADR-017 defines the operational autoencode -> compact -> fixed-point -> decode -> evidence -> staged-adaptation law. ADR-018 adds sparse support, bounded speculation and benchmark-gated self-optimization. ADR-019 establishes the large virtual 3D ROM/world-state separation.

This ADR adds an exact binary 1 GiB streaming profile and binds the data codec, inward fixed-point loop and self-optimization loop into one measurable operator.

The geometry is exact:

\[
1\ \mathrm{GiB}=2^{30}\ \mathrm{bytes}=1024^3\ \mathrm{bytes}.
\]

Therefore the logical one-byte-per-voxel state is

\[
\boxed{X_t\in\mathbb B^{1024\times1024\times1024}}.
\]

This declaration is a logical address space, not a requirement to allocate a dense GiB buffer.

## Decision

Jarvis-X adopts the following master transition for this profile:

\[
\boxed{
\mathcal S_{t+1}
=
\mathcal V_{\mathrm{CTR}}
\left[
\mathcal O_{\Theta,\Pi,\mathcal A}
\left(
\mathcal D_{\Theta}
\circ\mathcal Q
\circ\operatorname{Fix}_{\Phi}
\circ\mathcal E^{3D}_{\Theta}
\right)
(\mathcal S_t,U_{t+1})
\right]
}
\]

with complete state

\[
\boxed{
\mathcal S_t=[X_t,Z_t,Q_t,\hat X_t,R_t,\Omega_t,\Theta_t,\Pi_t,\mathcal A_t,M_t].
}
\]

The outer optimizer may propose changes, but only the verifier may promote them to authoritative state.

## 1. Streamed 3D materialization

The reference backend partitions the 1024^3 volume into exact cubic bricks:

\[
64^3=262\,144\ \mathrm{bytes}=256\ \mathrm{KiB},
\]

\[
(1024/64)^3=16^3=4096\ \mathrm{bricks},
\]

and therefore

\[
4096\times64^3=1024^3=1\ \mathrm{GiB}.
\]

A backend SHALL be able to process bricks as an iterator/stream. Dense 1 GiB materialization is optional, never implied by the logical geometry.

The byte coordinate mapping is

\[
i=x+1024(y+1024z),\qquad 0\le x,y,z<1024.
\]

## 2. Inward encoder and fixed point

For a materialized brick with edge \(N=64\), the reference contraction is

\[
64^3\rightarrow32^3\rightarrow16^3\rightarrow8^3
\]

with selectable bounded latent edge \(n_z\in\{4,8,16,32\}\).

The general learned backend may use

\[
H^{(\ell+1)}=\sigma(K_\ell *_3 H^{(\ell)}+b_\ell).
\]

The inward self-consistency loop is

\[
\boxed{
Z_t^{(k+1)}=(1-\alpha)Z_t^{(k)}+\alpha\,E_{\Theta_t}(D_{\Theta_t}(Z_t^{(k)}))
}
\]

and halts when

\[
\|Z_t^{(k+1)}-Z_t^{(k)}\|\le\varepsilon_Z
\]

or when the immutable recursion budget in \(\Pi_t\) is exhausted.

## 3. Quantize, decode and exact residual

The converged latent is quantized and entropy coded:

\[
Q_t=\mathcal Q_b(Z_t^*),\qquad C_t^Z=\operatorname{EntropyEncode}(Q_t).
\]

The decoder produces

\[
\hat X_t=D_{\Theta_t}^{3D}(Q_t,\Omega_t).
\]

For exact byte reconstruction the residual is XOR:

\[
\boxed{R_t=X_t\oplus\hat X_t},\qquad C_t^R=\operatorname{EntropyEncode}(R_t).
\]

The exact reconstruction invariant is

\[
\boxed{X'_t=\hat X_t\oplus R_t=X_t}.
\]

A checksum/CRC or stronger configured integrity primitive SHALL validate the reconstructed brick before acceptance.

## 4. No-expansion fallback

Geometry does not guarantee compression. Let

\[
L_c=|H_c|+|C_t^Z|+|C_t^R|,
\qquad
L_r=|H_r|+|X_t|.
\]

The frame selector is

\[
F_t=
\begin{cases}
\mathrm{LATENT\_XOR}(C_t^Z,C_t^R), & L_c<L_r,\\
\mathrm{RAW}(X_t), & \text{otherwise}.
\end{cases}
\]

Thus incompressible input fails over to raw transmission rather than producing a false compression claim.

## 5. Temporal memory

For a recurrent stream,

\[
\boxed{
\Omega_{t+1}=\rho\Omega_t+(1-\rho)\Phi(Z_t^*,R_t,M_t)
}
\]

and later encoding may condition on \(\Omega_t\). Backends may additionally use dirty-page or temporal-delta support from ADR-019.

## 6. Self-optimization is transactional

The engine may evaluate candidate model/runtime/architecture state

\[
C_j=(\Theta_j,\Pi_j,\mathcal A_j)
\]

against a fixed evaluation anchor and record

\[
M_j=[Q_j,D_j,B_j,L_j,E_j,M_j^{mem},S_j].
\]

A candidate is admissible only if all hard invariants pass, including exact round trip where lossless mode is declared, finite/bounded execution, memory/resource limits and immutable benchmark gates.

For an objective \(J\), proposal is

\[
C^*=\operatorname*{arg\,min}_{j:\mathrm{valid}(j)}J(C_j).
\]

Promotion remains candidate-first:

\[
\boxed{
\mathcal S_{t+1}=
\begin{cases}
\Pi_\Lambda(\mathcal S_{t+1}^{cand}), & V_t=1\land J_{cand}<J_{stable},\\
\mathcal S_t, & \text{otherwise}.
\end{cases}}
\]

The optimizer SHALL NOT alter its own evaluation corpus, acceptance thresholds, baseline, integrity gates or evidence classification while optimizing against them.

## 7. CTR interpretation

For this runtime, CTR is the authority boundary:

```text
Generate candidate
 -> Contrast with stable state and immutable workload
 -> Reckon metrics/objective
 -> Verify round-trip, stability, resource and benchmark gates
 -> Correct by promote OR rollback
 -> Recur
```

This yields the operational cycle

```text
STREAM -> 3D VOLUME -> CONTRACT -> FIX -> QUANTIZE -> DECODE
       -> XOR RESIDUAL -> VERIFY -> OPTIMIZE CANDIDATE -> CTR GATE -> RECUR
```

## 8. Meaning of beyond-SOTA

Recursive optimization is not itself evidence of state-of-the-art performance. "Beyond SOTA" remains a comparative benchmark target under ADR-018. Such a claim requires a named external baseline, matched workload, declared hardware/software conditions, measured metrics and reproducible evidence.

The internal recurrence may search indefinitely in concept, but every physical evaluation is finite and bounded by explicit compute, memory, recursion and candidate budgets.

## 9. Reference implementation

`src/jarvisx/dr_moagi_1g_recursive_codec.py` provides a dependency-free reference backend with:

- exact 1024^3 logical address mapping;
- 4096 streamed 64^3 bricks;
- deterministic inward averaging contraction and outward expansion;
- bounded encode-decode latent fixed-point cycling;
- zlib entropy coding;
- exact XOR residual reconstruction;
- CRC verification;
- raw no-expansion fallback;
- candidate-first policy evaluation and explicit promotion.

The deterministic averaging codec is a reference mechanics layer, not a claim that it is the optimal learned codec. Learned/GPU backends may replace the transforms while preserving the invariants in this ADR.

## Consequences

Positive consequences are an exact byte geometry, bounded memory footprint, executable lossless reference path, explicit rate accounting and a testable self-optimization boundary. The main cost is that exact reconstruction can require a residual approaching raw size on high-entropy data, and recursive candidate evaluation adds compute overhead. Both costs are intentional and SHALL remain visible in telemetry.
