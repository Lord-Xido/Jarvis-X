# Dr Moagi Orthogonal Quantization Precision Boundary

## Status

Canonical numerical verification contract for orthogonal transform quantization inside Jarvis-X Layer 5 research runtimes and codec/archive adapters.

This specification extends ADR-005 and the Moagi-Helmholtz archive contract.  It does not claim that a DCT is itself an MP4 codec, nor that every backend transform is orthogonal.

## 1. State and transform

Let

```text
x in R^M
D in R^(M x M)
D^T D = I
```

and let each transformed coefficient have a positive quantization step `delta_k`.

The complete transform boundary is

```text
X       = D x
A_k     = round_nearest(X_k / delta_k)
Xhat_k  = delta_k A_k
xhat    = D^T Xhat.
```

For a uniform precision step `Delta`, every `delta_k = Delta`.

## 2. Canonical rounding

The dependency-free reference uses nearest-neighbour quantization with exact half-step ties away from zero:

```text
+0.5 -> +1
-0.5 -> -1
```

A hardware or codec backend may use a different deterministic tie rule only when that rule is explicitly versioned and still reconstructs the nearest quantization level.

## 3. Deterministic coefficient envelope

For every coefficient,

```text
epsilon_k = X_k - Xhat_k
|epsilon_k| <= delta_k / 2.
```

Therefore

```text
||epsilon||_2 <= 0.5 * sqrt(sum_k delta_k^2).
```

Because `D` is orthonormal,

```text
||x - xhat||_2 = ||D^T epsilon||_2 = ||epsilon||_2.
```

Hence

```text
B_Q = 0.5 * sqrt(sum_k delta_k^2)
||x - xhat||_2 <= B_Q.
```

For uniform `Delta`,

```text
B_Q = Delta * sqrt(M) / 2.
```

No statistical cancellation assumption is required.

## 4. Precision gate

Define

```text
Lambda_Q = ||x - xhat||_2 / B_Q.
```

The quantized transform candidate is admissible when

```text
Lambda_Q <= 1.
```

The candidate then proceeds to the enclosing Jarvis-X validator.  A passing precision gate does not itself authorize a complete archive, model or geometry transaction.

```text
transform candidate
-> verify D^T D ~= I
-> quantize
-> reconstruct
-> compute Lambda_Q
-> precision COMMIT candidate
-> enclosing Pi_Lambda / policy / cycle checks
-> authoritative COMMIT or ROLLBACK.
```

## 5. Canonical arithmetic fixture at Delta = 0.1

Use

```text
x = [1.5, 1.9]^T
M = 2
Delta = 0.1
```

with the exact orthonormal two-point DCT-II basis

```text
D = (1/sqrt(2)) * [[1, 1], [1, -1]].
```

The forward coefficients are

```text
X = [
   2.4041630560342613,
  -0.2828427124746189
].
```

Nearest-neighbour quantization gives

```text
A = [24, -3].
```

Dequantization gives

```text
Xhat = [2.4, -0.3].
```

The exact transpose inverse reconstructs

```text
xhat = [
  1.48492424049175,
  1.9091883092036785
].
```

The spatial residual is

```text
e = x - xhat
  = [
      0.0150757595082500,
     -0.0091883092036786
    ].
```

Thus

```text
||e||_2 = 0.0176551281720920
B_Q     = 0.1 * sqrt(2) / 2
        = 0.0707106781186548
Lambda_Q= 0.249681217064078.
```

Therefore

```text
Lambda_Q < 1
PRECISION GATE: PASS.
```

## 6. Normalization-failure diagnosis

The matrix

```text
D_bad = [[1/sqrt(2), 1/sqrt(2)], [0.5, -0.5]]
```

is not orthonormal:

```text
D_bad D_bad^T = diag(1, 0.5).
```

Using `D_bad^T` as though it were `D_bad^-1` introduces deterministic transform distortion before quantization.  A resulting error near `sqrt(0.1^2 + 0.1^2)` is therefore not evidence that the correct nearest-neighbour error envelope should be doubled.

The runtime must fail closed on this basis unless it is explicitly handled as a non-orthogonal transform with a separately justified inverse-norm bound.

## 7. Frequency-selective precision

Low-frequency structural coefficients may use smaller steps than high-frequency coefficients:

```text
delta_0 <= delta_1 <= ...
```

The deterministic spatial bound remains

```text
B_Q = 0.5 * sqrt(sum_k delta_k^2).
```

All steps are part of the payload/version contract.  Side-information bytes used to describe them count toward the rate/information budget.

## 8. Backend requirements

Any backend advertising this orthogonal precision contract must expose:

```text
basis or basis version
orthogonality error
orthogonality tolerance
input dimension M
coefficient step vector or uniform Delta
rounding convention
quantized integer coefficients
reconstruction residual norm
B_Q
Lambda_Q
floating-point verification tolerance
```

Production implementations may replace explicit dense matrices with factorized DCT/FFT-like kernels, provided the implemented transform is equivalent to the declared orthonormal operator within the tested tolerance.

## 9. Moagi-Helmholtz integration

The orthogonal precision gate is inserted inside transform-based archive or latent adapters:

```text
Vstar / F / z
-> declared orthogonal transform D
-> X
-> Q_delta
-> integer payload A
-> dequantize
-> D^T
-> reconstruction
-> Lambda_Q
-> rate/distortion + cycle verification
-> Pi_Lambda
-> COMMIT / ROLLBACK.
```

For an MP4/video backend, this contract applies only to the transform stage that the backend explicitly maps to it.  The MP4 container, temporal prediction, entropy coding and codec syntax remain separate concerns.

## 10. Reference implementation

`src/jarvisx/orthogonal_quantization.py` provides:

```text
dct2_orthonormal_basis(size)
orthogonal_quantization_trace(values, transform, delta)
uniform_error_bound(delta, dimension)
OrthogonalQuantizationTrace
```

The implementation is dependency-free and correctness-oriented.  It is not intended to replace optimized numerical libraries or production codec kernels.

## 11. System invariant

The canonical invariant is

```text
||x - D^T [delta * round_nearest(Dx / delta)]||_2
<= delta * sqrt(M) / 2
```

for a uniform scalar `delta` and an orthonormal `D`.

A failed invariant triggers diagnosis and rollback; it does not automatically authorize a wider threshold.
