# ADR-013: Add the bounded Fourier-Markov-diffusion-resonance inward loop

**Status:** Proposed  
**Date:** 2026-09-05  
**Extends:** ADR-003, ADR-006, ADR-007

## Context

Jarvis-X already has a same-space sparse 3D field runtime, a bounded geometric-diffusion research runtime, and candidate-first control-plane semantics. The next step is to make a recurring analytical pattern explicit:

```text
field history
-> Fourier modal decomposition
-> Markov mode-transition model
-> diffusion loss estimate
-> temporal resonance detection
-> bounded modal feedback
-> projection / validation
-> commit or rollback
-> repeat
```

Without a precise contract, Fourier analysis, stochastic transition modeling, diffusion and resonance can be conflated into one metaphorical operator. This ADR separates them and defines exactly where feedback may re-enter the field runtime.

The new layer is a numerical analysis and control reference. It does not claim that a neural model physically stores its hidden state in Euclidean 3D space, that a short-window DFT identifies a unique physical resonance, or that modal reinforcement is universally beneficial.

## Decision

Jarvis-X adopts a Layer 5 **Fourier-Markov-diffusion-resonance (FMDR) analysis loop** over bounded histories of sparse 3D scalar fields.

For a configured spatial wavevector

```text
k = (kx, ky, kz)
```

and sparse scalar field `Psi_t(r)`, the reference spatial coefficient is

```text
A_k(t) = (1 / |support_t|) * sum_r Psi_t(r) exp(-i k dot r).
```

The active-support normalization is part of the reference implementation. It is not interchangeable with full-volume FFT normalization when support changes.

### Markov mode dynamics

At every observation, the configured mode with the largest current Fourier amplitude becomes the discrete dominant-mode state

```text
q_t = argmax_k |A_k(t)|.
```

The empirical transition law is

```text
P_ij = count(q_t=i, q_t+1=j) / count(q_t=i)
```

for rows with observed outgoing transitions. Unobserved rows use an identity row in the reported transition matrix so the matrix remains stochastic without inventing cross-mode evidence.

For resonance ranking, a mode-specific persistence estimate is kept separate from that reporting convention:

```text
p_k = count(k -> k) / count(outgoing from k),  if outgoing evidence exists
p_k = 0.5,                                      otherwise.
```

The neutral `0.5` fallback prevents an unobserved mode from receiving artificial resonance reinforcement merely because its displayed Markov row is the identity.

This is a finite-window empirical Markov model. It is not a claim that the complete underlying process is first-order Markov.

### Diffusion loss

For a diagonal anisotropic diffusion tensor

```text
D = diag(Dx, Dy, Dz),
```

the per-mode linear diffusion rate is

```text
lambda_D(k) = k^T D k
```

and the exact one-step attenuation for the linear diffusion equation is

```text
a_D(k) = exp(-k^T D k * dt).
```

The reference implementation reports this attenuation; it does not silently apply diffusion to the authoritative field.

### Temporal resonance analysis

For each configured spatial mode, a short positive-frequency temporal DFT is applied to the complex coefficient history `A_k(t)`.

Let

```text
omega_k* = argmax_omega |A_tilde_k(omega)|^2.
```

Define spectral coherence as

```text
C_k = peak_power_k / total_nonzero_frequency_power_k
```

when nonzero-frequency power exists.

The reference resonance score is

```text
R_k = C_k * sqrt(peak_power_k) * S_k / (gamma + k^T D k)
```

where

```text
S_k = 0.5 + 0.5 p_k
```

and `gamma > 0` is a configured damping floor.

This score is an engineering ranking functional, not a universal physical definition of resonance. It deliberately rewards temporally concentrated, empirically persistent energy and penalizes damping plus diffusive loss.

### Inward feedback

The highest-scoring mode may propose a same-space field correction. The reference modal component is reconstructed on the currently active support from the current complex Fourier coefficient.

The feedback sign is determined by

```text
s_k = 2 p_k - 1.
```

Persistent modes (`p_k > 0.5`) are eligible for reinforcement; rapidly switching modes (`p_k < 0.5`) are eligible for suppression; modes without transition evidence are neutral.

The raw feedback is scaled by the configured feedback gain and normalized resonance score, then clipped per coordinate:

```text
|Delta Psi(r)| <= max_feedback_delta.
```

Every initial field and every observation is also projected into the configured value interval before publication, including cycles that have not yet accumulated enough history to activate resonance feedback.

### Candidate-first recurrence

The operational recurrence is

```text
observation_t
-> finite/resource validation
-> value projection Pi_Lambda
-> append to bounded pending history
-> Fourier coefficients
-> dominant-mode sequence
-> empirical Markov matrix and persistence
-> diffusion rates
-> temporal resonance scores
-> selected modal correction when min_history is satisfied
-> bounded value projection
-> optional external validator
-> COMMIT state and history or ROLLBACK both.
```

A rejected candidate does not mutate either the published FMDR state or its history buffer.

## Relationship to the Dr Moagi field and autoencoding loops

FMDR is an analysis/control operator, not a replacement codec. It may be inserted after an encoder has produced a same-space sparse latent field and before a decoder, or it may analyze the field-runtime state directly.

A compositional form is

```text
Z_t        = E_theta(X_t)
A_k(t)     = F_3D[Z_t]
P_t        = Markov(argmax_k |A_k|)
R_t        = Resonance(A_k history, P_t, D)
Z_t'       = Pi_Lambda[Z_t + bounded_feedback(R_t)]
X_hat_t    = D_phi(Z_t')
e_t        = X_t - X_hat_t
parameters = candidate-first update(e_t, telemetry)
```

Encoder, decoder, spectral analysis, Markov estimation, diffusion accounting and feedback remain distinct contracts.

The higher-level self-referential recurrence may therefore be written

```text
S_t+1 = G(S_t, spectrum(S_t), transitions(S_t), diffusion(S_t), resonance(S_t), error(S_t))
```

but every executable term must still satisfy the same-space/type/resource rules established by ADR-003 and the bounded authority rules established by ADR-007.

## Reference implementation

The dependency-free reference is:

```text
src/jarvisx/dr_moagi_fmdr.py
```

with focused tests in:

```text
tests/test_dr_moagi_fmdr.py
```

The reference uses direct sparse Fourier sums and a naive short-window temporal DFT. It is correctness-oriented and intentionally does not claim production FFT throughput.

## Required invariants

1. Every configured wavevector is finite, three-dimensional and nonzero.
2. Diffusion coefficients are finite and non-negative.
3. Damping is finite and strictly positive.
4. Field values are finite and active support is resource-bounded.
5. Reported Markov rows are stochastic.
6. Unobserved transition rows do not create artificial persistence in resonance ranking.
7. Resonance is reported separately from Fourier amplitude and diffusion attenuation.
8. Feedback is same-space and bounded per active coordinate.
9. Value projection applies before every published state, even before feedback activation.
10. The history window is explicitly bounded.
11. Validator rejection atomically preserves the prior published state and history.
12. No FMDR result silently rewrites the canonical VM, source code, policy, provenance or transaction layers.
13. Physical resonance, production performance and learned-model quality require separate evidence.

## Validation

Acceptance requires focused tests for:

- axis wavevector construction;
- sparse spatial Fourier mode discrimination;
- diffusion attenuation against the analytic exponential;
- deterministic temporal resonance detection on a synthetic traveling mode;
- stochastic Markov rows and persistent-mode self-transition probability;
- neutral persistence when a mode has no outgoing transition evidence;
- bounded per-cell reinforce/suppress feedback;
- value projection before minimum-history feedback activation;
- delayed feedback until the configured minimum history exists;
- bounded history retention;
- atomic rollback of state and history on validator rejection;
- malformed configuration rejection.

## Consequences

### Positive

- Fourier, Markov, diffusion and resonance semantics become individually testable;
- the engine can identify coherent spatiotemporal modes without treating resonance as raw amplitude alone;
- diffusion provides an explicit modal dissipation term;
- mode switching supplies a measurable stochastic stability signal;
- feedback remains bounded, same-space and reversible by transaction rollback;
- the loop composes naturally with existing encoder/decoder and field-runtime boundaries.

### Negative

- the direct sparse Fourier sum and naive temporal DFT scale poorly for large mode sets and long windows;
- active-support normalization can change amplitude when the support changes and must therefore be interpreted carefully;
- an empirical dominant-mode chain may discard important multimodal structure;
- the reference resonance ranking is an engineering heuristic, not a proof of a physical eigenfrequency;
- reinforcing a persistent mode can still be undesirable for a task objective, so external validation remains necessary.
