# ReflexVM-Toroidal v3 — isolated T³ fixed-point reference

This NumPy implementation is a three-axis periodic geometric engine, not a
bytecode interpreter, trained ANN, LLM, or learned compressor. It coexists
with jarvisx.toroidal_feedback, a separate T² × (0,1] spectral model.

## Operator

For period R > 0, fixed anchor a, and 0 < q < 1, all three axes use:

    fold(x)   = x mod R
    encode(x) = (R/2π) [cos(2πx/R), sin(2πx/R)] on each axis -> R⁶
    decode(z) = (R/2π) atan2(z_sin, z_cos) mod R
    F_a(x)    = [a + qR/(2π) sin(2π(x-a)/R)] mod R

The encoder is a local isometry of the flat T³ into R⁶, not a compression
map or global Euclidean-distance isometry. Decoding is invertible on the
encoded product-of-circles manifold up to floating-point error.

The magnitude of every axis derivative of F_a is <= q, so the intrinsic
flat-torus Lipschitz constant is at most q. Banach's theorem guarantees a
unique fixed point a and at least geometric convergence when anchors are
fixed. The guarantee does not apply to anchors changing each iteration.

A shared anchor collapses the whole cloud by design. Per-point reference
anchors preserve distinct points but need external reference data; this
does not constitute learning or demonstrate prediction accuracy.

## Run

Install dependencies: pip install -e '.[test]'

    python -m jarvisx.reflexvm_toroidal
    python -m jarvisx.reflexvm_toroidal --benchmark
    python -m jarvisx.reflexvm_toroidal --benchmark --explicit
    pytest -q -o addopts='' tests/test_reflexvm_toroidal.py

Benchmark figures are hardware and workload dependent, not claims of an
unmeasured speedup. Float64 cannot preserve arbitrarily huge input
coordinates exactly. No trainable weights or virtual petascale memory are
instantiated.
