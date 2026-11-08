# Ω-ENGINE v2 — isolated trained 3D neural auto-codec reference

**Status:** Research prototype. This directory is a separate, non-authoritative implementation; it does not modify Jarvis-X's canonical runtime, C++ operational codec, API, bytecode, or production deployment.

## Operational pipeline

```
3D float32 source (.npy)
  → 16×16×16 spatial tiles
  → learned separable rank-one 3D factors (four vectors)
  → least-squares latent projection (K=4)
  → per-channel signed-int16 quantization
  → ONN3 v2 (header + learned model + scales + latent bytes + CRC32)
  → independent NumPy decoder
  → full-volume tile-by-tile RMSE verification
  → (quality AND rate) transaction gate
  → COMMIT / polynomial v1 fallback / REJECT
```

Trains a small spatial-factor ANN with a 4→8→4 nonlinear decoder correction using PyTorch; the read-only ONN3 decoder needs only NumPy. Neither K-adaptation for the ANN nor an entire 10^18-voxel processing runtime is implemented. The fallback is a **different fixed polynomial codec** with an explicit descending-K sweep.

## Local setup and reproduction

Run from this directory (Python 3.10+):

```bash
python -m pip install numpy torch
python -m unittest -v test_omega_ann.py
python benchmark_streaming.py
python omega_ann_codec.py --steps 220 --side 128 --output-dir results
```

Own float32-compatible volume, nonempty XYZ dimensions divisible by 16:

```bash
python omega_stream.py encode input.npy output.onn3
python omega_stream.py verify input.npy output.onn3
python omega_stream.py decode output.onn3 reconstructed.npy
python omega_controller.py input.npy --output-dir accepted
```

**Important:** Standalone `encode` writes an experimental candidate even if quality fails. Only `omega_controller.py` commits an accepted bitstream after actual round-trip reconstruction and full-tile verification. Rejects are not certified. The implementation uses a two-pass quantizer and an additional full-volume verification pass. No arbitrary untrusted input should be considered hardened; CRC32 is **not authenticated encryption**.

## ONN3 packet accounting

- 35-byte little-endian header; 1,088-byte embedded decoder parameters and quantization scales; 8 latent bytes per 16³ tile.
- Header records magic/version, 3D dimensions, tile edge, latent K, number of tiles, model bytes, payload bytes and CRC32 of model plus latents.
- Byte compression ratio = (source float32 voxel bytes)/(entire serialized packet bytes), including decoder model.
- The source uses a 2³¹-voxel demonstrator limit, and the header has a 32-bit payload-length field. **Exavoxel capacity is an extrapolation only**, not implemented.
- Integrity catches accidental corruption; no malicious-tamper or secrecy guarantee.

## Local held-out synthetic measurements (2026-11-07)

These results were observed from the supplied prototype, on synthetic 128³ float32 volumes with 512 tiles each, seed 11 for evaluation and seed 4 for training. The repository branch's CI must independently verify its copied sources before merge.

| Dataset | Recorded ONN3 size | Ratio with model | Mean normalized-RMSE quality | Worst tile | Gate |
|---|---:|---:|---:|---:|---|
| Piecewise affine | 5,219 B | 1,607.32× | 99.994% | 99.978% | PASS |
| Smooth nonlinear | 5,219 B | 1,607.32× | 97.755% | 95.145% | FAIL |
| Random noise | 5,219 B | 1,607.32× | 71.123% | 70.425% | FAIL |
| Affine + noisy tile | 5,219 B | 1,607.32× | 99.938% | 71.421% | FAIL |

Quality `Q_tile = clip(1 - tile_RMSE / tile_range, 0, 1)`, constant tiles normalized to the whole-volume range. This is a *custom* metric, **not SSIM, perceptual similarity, or clinical diagnostic equivalence**. The general-purpose 1,000× @ 98% target is **not established**; no real-world held-out scientific/medical volumes or zfp/SZ3 measurements are supplied.

## Review gates

- [ ] GitHub Actions 7-test suite green on this exact commit.
- [ ] Inspect the ONN3 parser, memory limits, filesystem atomicity and input hardening.
- [ ] Independently replicate measured rates and error distributions on real held-out 3D datasets.
- [ ] Compare against zfp and SZ3 under a shared distortion criterion.
- [ ] Add authenticated encryption and sound key lifecycle before use with sensitive volumes.
- [ ] Define intended usage and regulatory/compliance review before touching medical/patient data.

Do **not** merge or deploy as a production service solely because these synthetic tests pass.
