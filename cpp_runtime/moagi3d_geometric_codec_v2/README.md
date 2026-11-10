# Dr Moagi 3D Operational Auto-Encoding / Decoding Engine

A **tested, local, geometric codec reference implementation** constructed from the
uploaded 3D orchestration dashboard. This is not the original bundled React app;
it replaces claimed numerical operations with actual computations and an auditable
byte format while preserving the 7-stage operational pipeline.

## Run the interactive 3D model

Open `Moagi_3D_Operational_Engine.html` in a web browser. It contains the entire
JavaScript codec and its 25,088-byte frozen basis embedded as a Base64 string;
there are **no package installations, CDNs, cloud dependencies or servers**.

* Orbit the source and reconstructed 3D voxel fields by dragging either panel.
* Use the slider to control the contractive latent feedback (rho).
* Regenerate torus, sphere, helix or wave inputs; single-step, solve or auto-feed.
* Export a verified `.dm3d` binary file, then import it to validate the
  exact-byte reconstruction.
* Source-space MSE measures error **before** XOR residual correction.

## Native C++17 codec

The C++ encoder/decoder uses the same frozen signed-int16 Q14 codebook
`basis.bin`. Supply that file to the executable:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -pedantic moagi_3d_runtime.cpp -o moagi3d
./moagi3d basis.bin selftest
./moagi3d basis.bin decode test_torus.dm3d decoded.raw
./moagi3d basis.bin encode test_input.raw recompressed.dm3d
```

`input.raw` must contain **exactly 784 bytes**. Both platforms can decode each
other's `.dm3d` v2 packages. `basis.bin` has SHA-256:

`0427c0b5ad72e6c1ed51a514a8470cc0a62ab5ae5294a1cbfb8e8f0cd144773c`

## Algebra and geometry

The 3D lattice is 14 × 8 × 7, with 784 scalar byte-valued voxels and 16
analytic Gaussian dictionary functions derived from 3D toroidal centers.
The immutable Q14 dictionary is used as B ∈ ℝ^(784×16).

A deterministic 3D source field is sampled at r=(x,y,z):

    torus distance: d(r)=sqrt((sqrt(x²+y²)-0.67)²+z²)
    field: X(r)=round(clamp(250 exp(-d(r)²/(2 sigma²))+seeded_noise,0,255))

Other demonstrator fields use sphere, helix, and wave distance functions.

Encoding is ridge-regularised geometric projection:

    z* = (Bᵀ B + 0.025 I)⁻¹ Bᵀ X

A Cholesky solver is used rather than forming an explicit matrix inverse.

Finite inward feedback, with 0≤rho<1, is:

    z[t+1] = rho z[t] + (1-rho) z*
    ||z[t] - z*||₂ = rho^t ||z[0] - z*||₂  (exact arithmetic)

This is a contraction in 16D latent space, not an infinite CPU loop. The
UI convergence threshold is 1e-3 in latent L2 norm, with a finite-step cap.

Quantisation is Q10 fixed point for the 16 coefficients:

    q_k = round(1024 z_k)          // signed int32
    B_int = round(16384 B)         // signed int16, frozen and shared
    pred_i = clamp(floor((Σ B_int[i,k] q_k + 2²³)/2²⁴),0,255)

The integer summation and prescribed rounding make the decoder prediction
reproducible across JavaScript and C++17 (under the checked arithmetic bounds).

Exact recovery uses a bytewise residual:

    residual_i = source_i XOR pred_i
    restored_i = residual_i XOR pred_i
    restored_i = source_i

The residual is optionally run-length encoded if smaller, otherwise stored raw.
Hence **exact reconstruction is guaranteed only when the residual is included**.
The latent representation by itself is approximate and not lossless.

## DM3D version 2 binary format

All multibyte integers are little-endian:

| Offset | Size | Meaning |
| --- | ---: | --- |
| 0 | 4 B | ASCII `DM3D` |
| 4 | 1 B | Version = 2 |
| 5–7 | 3 B | NX=14, NY=8, NZ=7 |
| 8 | 1 B | K=16 |
| 9 | 1 B | Residual codec: raw=0, zero-run/literal RLE=1 |
| 10–11 | 2 B | Residual payload length |
| 12–15 | 4 B | FNV-1a-32 checksum of reconstructed raw bytes |
| 16–79 | 64 B | 16 signed int32 Q10 latent coefficients |
| 80 onward | variable | 784 raw XOR residual bytes or RLE bytes |

The FNV-1a checksum detects routine data corruption but is **not** a
cryptographic authenticity guarantee or IP-protection mechanism.

## Reproducible validation

```bash
node test_core.js
./moagi3d basis.bin selftest
./moagi3d basis.bin decode test_torus.dm3d decoded.raw
```

The core test covers 20 deterministic geometry/seed round-trips, three
contraction coefficients, and seven malformed-packet cases.

Independent browser tests confirmed: 6/6 built-in checks; a working 3D canvas;
actual export/import; and successful C++ decode of the browser-produced file.
Cross-language interoperability was tested **in both directions**.

### Recorded outputs (seed 17; all four shapes)

* Source: 784 bytes = one byte per 3D voxel.
* Latent: 16 int32 values (64 bytes); 49× **dimensionality** reduction.
* Lossless file with raw residual: 864 bytes.
* Storage ratio source/file = 784/864 = **0.907407×** (10.204% expansion).
* Exact residual corrected error: 0/784 byte mismatches.
* Uncorrected torus source-space MSE: ~1017.2 squared byte-intensity units.

The measured ratio is **not 49× file compression**; the baseline has no
validated speedup, billion-inference throughput, GPU parallelism, cloud
execution, learned model, actual revenue generation or deployed billing.

## Context from publicly available research

* Cornell Numerical Methods for Data Science: contraction mapping theory
  https://www.cs.cornell.edu/~bindel/nmds/00-Background/04-Calculus.html
* van den Oord et al., VQ-VAE (NeurIPS 2017): learned discrete latent codes
  https://arxiv.org/abs/1711.00937
* Ballé et al., Integer Networks for Data Compression with Latent-Variable Models
  https://research.google/pubs/integer-networks-for-data-compression-with-latent-variable-models/
* Minnen, Ballé & Toderici, learned entropy priors for real compression
  https://research.google/pubs/joint-autoregressive-and-hierarchical-priors-for-learned-image-compression/

## Technical boundaries / next stage

This is a **reproducible prototype**, not a trained neural autoencoder. To
become a competitive multimodal codec, it would need trained transforms,
quantised learned codebooks, entropy coding, image/audio/video input/output,
rate–distortion benchmarking on held-out datasets, memory caps, and CPU/GPU
profiler reports. Avoid inferring a throughput number from UI animation speed.
## Repository integration

This module is deliberately isolated from pre-existing Jarvis-X codecs, canonical equations and deployment/billing subsystems. GitHub Actions verifies the 784-byte geometric reference and JS/C++ interchange. Production throughput, 49× lossless compression, self-improving ANN training and actual revenue are not claimed.
