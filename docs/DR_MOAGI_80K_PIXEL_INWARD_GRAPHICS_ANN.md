# Dr Moagi 80K-Pixel Inward Graphics ANN

**Status:** bounded C++17 reference laboratory  
**Framebuffer:** `400 x 200 = 80,000` logical pixels  
**Neural field:** `16 x 16 x 16 x 8 = 32,768` hidden scalars  
**Authority:** CTR-style same-frame candidate verification

## Purpose

This runtime closes a software 3D graphics pipeline around a recurrent 3D ANN. The renderer no longer terminates at display output: the exact 80,000-pixel RGB/depth framebuffer becomes the observation consumed by the neural field, its reconstruction residual is back-projected into the 3D lattice, and the accepted latent state deforms the next rendered geometry.

`80K-pixel` means exactly **80,000 total logical framebuffer pixels**. It does not mean an `80,000 x N` display and it is distinct from the repository's separate `80,000^3` sparse logical-volume/MP4 runtime.

## State and cycle

```text
G_t
 -> software 3D rasterization + Z buffer
 -> X_t = {RGB, depth}
 -> depth-aware E_3D
 -> H_t
 -> six-neighbour recurrent ANN
 -> D_3D(H_t)
 -> Xhat_t
 -> R_t = X_t - Xhat_t
 -> B_3D(R_t)
 -> shadow parameter candidate
 -> CTR same-frame verification
 -> COMMIT or ROLLBACK
 -> bounded latent deformation of G_(t+1)
```

The recurrent update is:

```text
h'ijk = tanh(Wself*hijk + Wnbr*mean(neighbours)
             + Win*encoded_ijk + Wfb*feedback_ijk + b)
```

## Transaction boundary

Let `L_base` be same-frame reconstruction MSE before a proposed decoder/density update and `L_cand` its MSE after the update. A proposal becomes authoritative only when:

```text
finite(candidate)
AND L_cand <= L_base + 1e-7
AND feedback_energy <= 0.5
```

Otherwise the adaptive parameters roll back. Wall-clock timing is telemetry only and never an acceptance criterion.

## Geometry feedback

The authoritative ANN state samples the 3D toroidal model at world coordinates. Hidden channels produce bounded normal displacement and colour modulation. Global residual energy also contracts the inward feedback stream after clamping.

```text
geometry -> pixels -> ANN -> residual -> ANN -> geometry
```

This is numerical feedback closure, not unrestricted source-code mutation or unbounded self-improvement.

## Build and run

```bash
cmake -S cpp_runtime -B build/cpp-runtime \
  -DCMAKE_BUILD_TYPE=Release \
  -DJARVISX_BUILD_GL_VISUALIZER=OFF
cmake --build build/cpp-runtime --target jarvisx-80k-pixel-inward-ann --parallel

./build/cpp-runtime/DrMoagi-80K-Pixel-Inward-ANN --self-test
./build/cpp-runtime/DrMoagi-80K-Pixel-Inward-ANN --headless --frames 30 --json
./build/cpp-runtime/DrMoagi-80K-Pixel-Inward-ANN
```

Interactive mode uses ANSI true-colour half-block cells. Terminal output may be resampled to fit the physical console, while the internal framebuffer remains exactly `400 x 200`.

## QSOL deterministic observation integration

The 80K framebuffer also drives a separate 16³ Q16.16 reverberation field
via fixed RGB sampling. The additional non-mutating observation yields a
256-unit 8×8×4 pooled signal and canonical integer render hash. It is
**read-only**: no QSOL observer output is routed to the ANN candidate verifier,
render geometry, or the reverberation state update. The ANN's existing
guarded learning remains unchanged.

For headless JSON runs, the fields `qsol_state_hash` and
`qsol_render_hash` are decimal strings; the observer's integer camera offsets
are derived solely for deterministic layout telemetry. Equal quantized input
sequences and frame indices produce equal hashes. Floating-point rasterization
may differ across architectures, so cross-device pixel identity is not claimed.

See [ADR-0036](adr/0036-qsol-deterministic-3d-observer.md) for the exact
`||A||∞ ≤ 15/16` operator, quantization, hashing order, and test vectors.

## Verification

The deterministic self-test checks exact framebuffer and latent cardinality, projection/unprojection consistency, finite metrics, feedback energy `<= 0.5`, non-regressing MSE for every committed parameter candidate, and at least one accepted candidate. Focused CI covers GCC, Clang, ASan and UBSan.

## Relationship to existing systems

| Subsystem | Relationship |
|---|---|
| `DrMoagi-80K3-MP4` | Separate `80,000^3` sparse logical-volume engine; a bounded projection adapter may feed this ANN without making the full field resident. |
| C++ 3D autoencoder | Precedent for volumetric encode/decode; this runtime adds graphics-world feedback closure. |
| DM3D operational engine | Shares Generate/Contrast/Verify/Commit discipline; numerical operators are not claimed identical. |
| Exact bitwise fixed-point verification | Complementary binary closure; floating reconstruction tolerance here is not exact bitwise closure. |
| 3D ANN Cloud Graphics Processor | Executable bounded backend for Acquire -> Map3D -> Encode -> ANN -> Graphics -> Decode -> Verify -> Repeat. |

## Complexity and evidence boundary

Per-frame work is bounded by the fixed framebuffer and latent support:

```text
raster/contrast: O(80,000)
latent recurrence: O(16^3 * 8^2)
decode/backprojection: O(16^3 + 80,000)
```

This is a CPU software reference. It does not establish GPU-equivalent throughput, differentiable-rasterizer correctness, trained generalization, AGI, consciousness, autonomous authorization, or hardware performance. Measured `step_ms` is telemetry only and is excluded from candidate acceptance.
