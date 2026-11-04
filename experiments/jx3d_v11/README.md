# Jarvis-X Native 3D Operational Engine v11 (isolated candidate)

This directory integrates the verified local `JarvisX_3D_Operational_v11` CPU prototype as a **source-only vendored archive** so existing Jarvis-X runtime branches and builds remain unaffected.

## Reproduce

```bash
mkdir -p /tmp/jx3d_v11
tar -xJf experiments/jx3d_v11/jx3d_v11_source.tar.xz -C /tmp/jx3d_v11
cd /tmp/jx3d_v11
bash run_tests.sh
python3 serve.py
# Browser: http://127.0.0.1:8765
```

Requires Python 3.10+ and a C++20 compiler. No third-party Python/Node dependencies.

## Verified prototype scope

- Native 3D Q8 lattice and six-face-neighbor stable stencil
- True 2×2×2 spatial tiling; 76-parameter shared 8→4→8 neural model and training
- Q8 bitplanes and bit-transition counters; byte-exact XOR residual decoding + CRC32
- A* routing, delayed voxel fault recovery; local HTTP and WebGL projection interface
- Local validation: native codec / ANN / fault tests, 600 path comparisons, six HTTP tests

**Not claimed:** trillion-parameter trained model, hardware-level photonics, physical power measurements, cloud rollout, production network security, compression gain or browser rendering certification.

**Note:** archive is used for a minimal, isolated GitHub integration. Extract before reviewing source. The source of truth is the v11 source distribution, not the original static OS v9 dashboard. Release excludes native binaries, `__pycache__`, large original v9 static export, and generated samples.

### Source integrity

The uncompressed tar contains `src/`, `serve.py`, `web/`, `tests/`, `run_tests.sh`, `benchmark.py`, `README.md` and `VALIDATION_REPORT.md`. SHA-256 hash of the compressed tar is recorded in `SHA256SUMS.txt`. The CI workflow extracts and compiles it on Ubuntu for verification.
