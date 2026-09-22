# Dr Moagi 80K^3 Sparse Auto-Encoding/Decoding MP4 Compute Runtime

**Status:** bounded executable reference  
**Authority:** ADR-016 + ADR-017  
**Logical geometry:** `80000 x 80000 x 80000` voxels

## Contract

This specialization instantiates the canonical Dr Moagi operational equation as

```text
80K^3 logical address field
  -> sparse 32^3 active brick
  -> 32^3 -> 16^3 -> 8^3 -> 4^3 -> 2^3 -> 1 encoder pyramid
  -> latent inward contraction
  -> bounded fixed-point refinement
  -> selective 3D decode
  -> residual measurement
  -> residual correction
  -> CTR verification
  -> staged Omega / Theta adaptation
  -> commit OR rollback
  -> 3D -> 2D RGB projection
  -> frame sequence
  -> external MP4 codec adapter
  -> recur
```

The logical field contains exactly

```text
80000^3 = 512,000,000,000,000 voxels
```

but the runtime materializes only bounded `32^3` bricks. Logical extent is not resident memory.

## Equation

For active support `A_t`,

\[
X_t^{A}
\xrightarrow{E_\Theta}
Z_t
\xrightarrow{\Phi_{in}}
Z_t^*
\xrightarrow{D_\Theta}
\hat X_t
\xrightarrow{e_t=X_t^A-\hat X_t}
\Pi(e_t)
\xrightarrow{CTR}
S_{t+1}^{cand}.
\]

Authoritative promotion remains

\[
S_{t+1}
=
V_t\Pi_\Lambda(S_{t+1}^{cand})
+
(1-V_t)S_t.
\]

The reference runtime treats a candidate as admissible only when reconstruction
metrics are finite, residual correction is non-worsening, and MSE remains under
the declared ceiling.

## Build

```bash
cmake -S cpp_runtime -B build/cpp-runtime -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp-runtime --config Release --parallel
ctest --test-dir build/cpp-runtime -C Release --output-on-failure
```

Run:

```bash
./build/cpp-runtime/DrMoagi-80K3-MP4 \
  --cycles 60 \
  --active-bricks 32 \
  --frames build/80k-frames \
  --width 1920 \
  --height 1080 \
  --fps 30
```

The runtime emits deterministic PPM frames when `--frames` is supplied.
A standards-compliant MP4 bitstream requires an actual video codec backend.
The CLI prints an FFmpeg command that converts the emitted frame sequence to
H.264/MP4 when FFmpeg is installed:

```bash
ffmpeg -y -framerate 30 \
  -i build/80k-frames/frame_%06d.ppm \
  -c:v libx264 -pix_fmt yuv420p output.mp4
```

## Evidence boundary

This is an executable sparse reference, not evidence that 512 trillion voxels
are simultaneously resident or that an 80K x 80K video frame can be encoded by
a given MP4 profile. The 80K^3 quantity is the logical 3D compute domain; the
video adapter receives bounded 2D projections.

Originator of the Dr Moagi 3D Ephemeral-Notion Intelligence Framework:
Matladi Maxwell Moagi (Lord-Xido). Canonical provenance:
`docs/attribution/DR_MOAGI_EPHEMERAL_NOTION_FRAMEWORK.md`.
