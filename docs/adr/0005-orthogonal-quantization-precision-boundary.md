# ADR-005: Adopt the orthogonal quantization precision boundary

**Status:** Accepted  
**Date:** 2026-08-12  
**Extends:** ADR-004

## Context

The Moagi-Helmholtz archival path permits transform/quantization backends, but a recent two-coordinate verification exposed an important distinction between quantization error and transform-normalization error.  A matrix whose second DCT row was scaled as `[0.5, -0.5]` was described as orthonormal and then inverted with its transpose.  It is not orthonormal, so the resulting approximately `[-0.1,+0.1]` spatial residual was primarily a basis/inverse mismatch rather than evidence that the nearest-neighbour quantization bound should be doubled.

Jarvis-X therefore needs one canonical precision contract for any backend that claims orthogonal transform quantization.

## Decision

For an input `x in R^M`, an orthonormal transform `D`, and positive coefficient steps `delta_k`, define

```text
X       = D x
A_k     = round_nearest(X_k / delta_k)
Xhat_k  = delta_k A_k
xhat    = D^T Xhat
```

with the invariant

```text
D^T D = I.
```

The reference rounding rule is nearest neighbour with exact half-step ties away from zero.  Backends may use another declared deterministic tie rule, but they must preserve the nearest-level residual property.

For every coefficient,

```text
|X_k - Xhat_k| <= delta_k / 2.
```

Therefore

```text
||x - xhat||_2
= ||X - Xhat||_2
<= 0.5 * sqrt(sum_k delta_k^2).
```

For a uniform step `Delta`, this reduces to

```text
||x - xhat||_2 <= Delta * sqrt(M) / 2.
```

This is a deterministic worst-case bound.  It does not rely on independent, zero-mean, uniformly distributed, or cancelling quantization errors.

## Precision gate

Define

```text
Lambda_Q = ||x - xhat||_2 / B_Q
B_Q      = 0.5 * sqrt(sum_k delta_k^2).
```

The transform/quantization candidate is admissible only when

```text
Lambda_Q <= 1
```

within the declared floating-point verification tolerance.

If the gate fails, the implementation must first diagnose:

1. transform normalization;
2. inverse pairing;
3. rounding/quantizer convention;
4. payload corruption;
5. precision, overflow or non-finite values;
6. coefficient-step/version mismatch;
7. substitution of a non-orthogonal transform.

The deterministic gate may not be silently widened merely because a malformed transform produced a larger error.

## Canonical two-coordinate fixture

For

```text
x       = [1.5, 1.9]^T
Delta   = 0.1
D       = (1/sqrt(2)) [[1,1],[1,-1]]
```

the verified reference trace is

```text
X       = [ 2.4041630560342613, -0.2828427124746189 ]
A       = [ 24, -3 ]
Xhat    = [ 2.4, -0.3 ]
xhat    = [ 1.48492424049175, 1.9091883092036785 ]
e       = [ 0.0150757595082500, -0.0091883092036786 ]
||e||_2 = 0.0176551281720920
B_Q     = 0.0707106781186548
Lambda_Q= 0.249681217064078
```

The transition therefore commits with substantial deterministic margin.

## Non-uniform precision

For frequency-selective steps `delta_k`, the exact reference envelope is

```text
B_Q = 0.5 * sqrt(sum_k delta_k^2).
```

This permits lower-frequency coefficients to use smaller steps while high-frequency coefficients use larger steps, provided the complete step vector is versioned and included in the archive/receipt information budget.

## Non-orthogonal transforms

A non-orthogonal but invertible transform is a different contract.  If `T` is used, the safe induced bound requires the inverse operator norm, for example

```text
||x - xhat||_2 <= ||T^-1||_2 * 0.5 * sqrt(sum_k delta_k^2).
```

Such a backend may not claim the orthogonal bound unless it proves the required norm property.  ADR-005 does not make non-orthogonal transforms canonical; it merely prevents accidental transpose-as-inverse reasoning.

## Transaction and provenance boundary

Every orthogonal quantization receipt records at least:

```text
transform/version or basis identity
orthogonality tolerance and measured error
coefficient step(s)
rounding convention
quantized coefficient payload
reconstruction residual norm
B_Q
Lambda_Q
validator decision
COMMIT / ROLLBACK outcome
```

When used inside Moagi-Helmholtz, this precision receipt is subordinate to the existing candidate-first transaction:

```text
render/latent state
-> orthogonal transform
-> quantize
-> reconstruct
-> precision gate
-> archive/cycle validation
-> COMMIT or ROLLBACK
-> journal.
```

## Architectural effect

The canonical 64-bit VM is unchanged.  ADR-005 adds a reusable Layer 5 numerical verification primitive for DCT/tensor/archive backends.  C++, CUDA, DMEB, FPGA, native SIMD and video-codec adapters may implement accelerated versions only if they preserve the same declared normalization, error-bound and transaction semantics.

## Validation

Acceptance requires executable tests for:

- the exact two-coordinate DCT fixture;
- rejection of the incorrectly normalized basis;
- uniform and non-uniform bounds;
- declared half-step rounding behavior;
- malformed/non-finite inputs;
- a second orthogonal basis demonstrating norm preservation.

The reference implementation is `src/jarvisx/orthogonal_quantization.py`.
