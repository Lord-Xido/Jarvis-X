# MagnusStream — independently verified 3D recursive feedback

**Status: numerical reference and browser demonstration. Not a trained ANN or operational multimodal codec.**

This isolated Jarvis-X app preserves the distinction between the 1000³ *conceptual* MagnusStream graphics swarm and an actually executed numerical encode/refine/decode loop. The original 2026-11-01 HTML contains 1,000 simulated particles, 10,000 + 100,000 rendered background points, and visualization-derived compression/loss readings. The uploaded original application has **not** been proved to encode or reconstruct actual image, audio and text inputs.

The separate verifier is executable and reproducible:

```bash
python -m pip install numpy pytest
python apps/magnusstream-3d/verify_magnusstream_3d.py --loops 1000 --json /tmp/magnusstream.json
python -m pytest -q -o addopts='' apps/magnusstream-3d/test_runtime.py
```

Open [index.html](./index.html) locally in a modern browser for interactive 3D voxel views, 100/1000 recursion controls, live error telemetry, an MS3D binary export/import, and nine self-checks. No external scripts or graphics libraries are required.

## Operator contract

Let `X_t` be the current `16 × 16 × 16` scalar volume. Define

```text
X_(t+1) = F(X_t) = (R ∘ D ∘ M ∘ E)(X_t), where R = identity.
E: blockwise 2×2×2 mean, 16³ -> 8³.
M: periodic six-neighbor diffusion, (1 - α) z + (α / 6) sum(neighbors), α = 1/4.
D: replicate each latent scalar into 2×2×2 reconstructed voxels, 8³ -> 16³.
```

Every next step consumes the preceding **decoded** state. Constant fields are invariant; the periodic mean is conserved; spatial variance is non-increasing. Nonconstant modes decay toward the global mean. Convergence therefore does **not** establish preservation of multimedia detail.

The binary `MS3D` v1 container is 4 magic bytes + 7 little-endian uint32 fields (version and spatial dimensions) + 512 little-endian float32 latent samples. Its 2,080 bytes compared with 16,384 original float32 bytes give a **fixed-width lossy payload ratio of 7.8769:1**. It is not an entropy-coded or lossless compression claim. The Python transport rejects malformed length, magic, version/dimensions and nonfinite samples; the browser does likewise.

## Evidence boundary

The [audit](./OPERATIONAL_VERIFICATION.md) documents original artifact limitations, measured Python reference behavior and prior browser results. The [reference receipt](./reference_results.json) captures the November 1, 2026 execution. CI runs a fresh 1,000-loop computation and tests for shape, numerical invariants, binary transport, corruption rejection, model convergence and reconstruction loss.

**Architectural compatibility:** This is an isolated specialization of `R ∘ D ∘ M ∘ E`; it does **not** claim full conformance with [ADR-017](../../docs/adr/0017-dr-moagi-operational-autoencoding-equation.md) or [ADR-016](../../docs/adr/0016-canonical-typed-state-transaction-and-geometry-profiles.md), which additionally require typed state, staged updates, verification gates and atomic rollback. Numerical near-fixed-points are not the exact binary fixed-points of [ADR-034](../../docs/adr/0034-exact-bitwise-fixed-point-verification.md).

This research app falls under the repository's [Dr Moagi provenance framework](../../docs/attribution/PERMEATION_MANIFEST.md) when used as a designated Dr Moagi derivative. Graphics are illustrative and do not imply one billion active neural parameters or 1000³ actual computation.
