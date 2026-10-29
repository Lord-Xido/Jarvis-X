# JARVIS X: 1,000-iteration 3D neural verification reference

**Status:** reproducible CPU research reference; not a global contraction proof.

The standalone source is `cpp_runtime/src/jarvis_x_1000_loop.cpp`, pinned to the
original local source SHA-256:

`b0daaa6d089eeb3f1971521a290f48cb052e7021bc6507c673b955c403eb7388`

The source contains a trainable 399-parameter 2×2×2 patch autoencoder,
nonlinear 8→3→8 encoding/decoding, 3D latent projections, Hopf visualization,
reversible 60-bit Morton address interleaving, and a separately reversible
binary codec. `loop1000` freezes learned parameters and measures 1,000 inward
inference transitions in both real-valued and byte-quantized paths.

## Mathematical contracts

Neural continuous trajectory (approximate, lossy):

\[
x_{k+1}=D_\phi(E_\theta(x_k)),\qquad k=0,\ldots,999.
\]

Quantized byte trajectory (finite but generally **not** lossless):

\[
b_{k+1}=Q_{8}\big(D_\phi(E_\theta(b_k/255))\big).
\]

Lossless binary path is verified **separately**:

\[
\operatorname{Decode}_{3D}(\operatorname{Encode}_{3D}(B))=B.
\]

- `continuous_step_rmse` measures \(F(x_k)-x_k\); it is not preservation of \(x_0\).
- `continuous_origin_rmse` measures drift from \(x_0\); the drift is nonzero in the supplied experiment.
- Quantized recurrence and numerical thresholds are observed per run, not globally proven.
- Jacobian norm samples and loose global bounds do not establish Banach contraction.
- Morton interleaving is exact for valid discretized coordinates; mapping continuous
  geometric positions to integer coordinates is lossy.
- The virtual \(10^6\times10^6\times10^6\) address space is **not** physically allocated.
- The binary stream's integrity checksum is not a cryptographic signature.

## Build and run

```bash
cmake -S cpp_runtime -B build/recursive3d -DCMAKE_BUILD_TYPE=Release -DJARVISX_BUILD_GL_VISUALIZER=OFF
cmake --build build/recursive3d --target jarvisx-recursive3d-1000 -j2
ctest --test-dir build/recursive3d -R recursive3d-1000 --output-on-failure

# Trained checkpoint is required for the loop: create one from user-supplied bytes.
./build/recursive3d/jarvisx-recursive3d-1000 train INPUT.bin model.jxnm 8 128
./build/recursive3d/jarvisx-recursive3d-1000 loop1000 INPUT.bin model.jxnm results 16
```

`cpp_runtime/tests/test_recursive3d_1000.py` creates deterministic synthetic
input, trains a temporary model, runs two 1,000-transition experiments, checks
1,001 metric rows and the expected geometric snapshots, verifies exact source
restoration, and compares the two traces byte-for-byte. All test data lives in a
private temporary directory and is deleted on completion.

## Historical reference measurement (not CI expected values)

The separately completed 2026-10-29 sample experiment used 1,024 sampled
8-byte patches over 1,000 transitions. The continuous residual decreased from
0.07782206058 to approximately 9.93e-11 (normalized RMSE) while the final
continuous origin drift was approximately 0.18692245. The byte-quantized
trajectory was stationary by iteration 121 on that sample. These figures are
**not** the expected results on a new dataset or checkpoint; regression tests
assert invariants and reproducibility instead of hardcoded convergence.

## Interoperability and scope

This module is intentionally independent of the existing
`cpp_runtime/include/jarvisx/bit_self_loop3d.hpp` and the DOCX `v1000_adapter`.
The former provides exact XOR/popcount binary fixed-point contracts; the latter
validates a different document-container VM format. Adding this neural benchmark
does not change or replace either contract. No trained model, copyrighted input,
or benchmark-derived binary is committed.
