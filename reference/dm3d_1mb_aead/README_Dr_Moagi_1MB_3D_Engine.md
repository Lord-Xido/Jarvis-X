# Dr Moagi 1 MB Python 3D Auto-Encoding / Decoding Engine

**Release:** DM3D-AEAD-v2 · **Artifact:** `Dr_Moagi_1MB_3D_Engine.py` · **Exact source size:** 1,000,000 bytes (decimal MB). Standard-library core with an optional PyTorch neural experiment.

## What is actually implemented

- **Logical 3D sparse addressing:** `8,000,000,000,000` bit coordinates per axis; arbitrary-precision Morton coding and reversible decode. This is virtual *addressability*, not physical RAM or processed voxels.
- **Physical processing:** bounded 32×32×32 **byte** tiles (32,768 bytes = 32 KiB each); the default workload is 1 MiB (32 tiles). Distinguish this from the original bit-packed 4 KiB chunks; this runtime represents one byte per voxel to preserve arbitrary input bytes. Spatial chunk coordinates are implicit in a bounded 3D grid.
- **Geometric transforms:** identity + 90-degree rotations around three axes, bijective reversible XYZ modular-delta, a learned 3D local predictor with exact modulo-256 residual, and ordinary zlib/raw candidates. The smallest byte-exact verified candidate is recorded independently for every tile.
- **Automatic updating weights:** a four-parameter Q8.8 predictor trained on held-out-separated points of each tile by bounded ridge regression; commits only if the tile's held-out squared error falls and the immutable anchor does not degrade by more than 3%. All packet weights are embedded in the packet so previous compressed files remain decodable after further adaptation. This is a limited adaptive signal predictor, **not a pretrained semantic ANN**.
- **Integrity & transaction:** bounded versioned archive `DM3D-AEAD-v2`, archive-header SHA-256, body SHA-256, per-packet encoded/decoded SHA-256, complete restored-source SHA-256, reject malformed/truncated/oversized payloads, temporary output and atomic replace only after successful verification. Hashes provide accidental corruption and consistency detection, **not authentication against an attacker capable of rewriting the entire archive**.
- **Optional neural experiment:** PyTorch CNN encoder/decoder, 4×4×4 toroidal latent fixed-point map with conservative contraction gain 0.4; train/validation MSE-gated Adam weight updates with rollback of optimizer state. It writes a separate PyTorch checkpoint and **does not modify the lossless archive packet format**.
- **Bounded WWW ingestion:** `fetch` accepts only the HTTPS GitHub public `/repos/{owner}/{repo}/events` API host (`api.github.com`) with no redirects, bounded bytes and timeout. Data is packed as 3D bytes with provenance; successful reconstruction of the JSON bytes is not semantic understanding of those events.
- **Embedded fixture index:** 11,706 deterministic 8³ volume reference hash vectors in source comments. These supply a meaningful regression data appendix to satisfy the explicit 1 MB source-size requirement. The executable logic is much smaller than 1 MB.

## End-to-end law

Let `X` be a byte-exact 3D tile, `T_r` a reversible cube rotation, and `Δ_xyz` an invertible periodic-independent, modular spatial difference transform. The four reversible coding proposals are:

```
raw(T_r(X))
zlib(T_r(X))
zlib(Δ_xyz(T_r(X)))
zlib((T_r(X) - Pred_{w_t}(T_r(X))) mod 256)
```

For each tile, the encoder selects the candidate with minimum actual payload size. The decoder uses the stored mode, orientation and immutable integer predictor weights. It verifies reconstruction against the encoded source hash before promotion:

```
source bytes → bounded 32³ tile → 3D permutations and codecs
             → candidate selection → packet integrity/metadata
             → archive and candidate policy checkpoint
             → independent packet decoder
             → source SHA-256 and resource verification
             → atomic file commit OR rollback
```

Per-chunk `w_t` adapts only *after* the current chunk was encoded and is never needed from a mutable global checkpoint to decode an earlier chunk.

For the separate optional neural model:

\[
Z_0=E_\theta(X),\quad
Z_{k+1}=Z_0+0.4\tanh(P_{\mathbb T^3}(Z_k-Z_0)+0.2\tanh Z_0),\quad
\hat X=\sigma(D_\phi(Z_K)).
\]

The six-neighbor periodic averaging operator has Euclidean induced norm ≤1, so the fixed-input iterative map has Lipschitz bound ≤0.4. Contraction of this internal neural field is **not** proof of good reconstruction or external factual correctness.

## Usage

Only Python 3.10+ is required for exact encode/decode:

```bash
python Dr_Moagi_1MB_3D_Engine.py selftest --vectors
python Dr_Moagi_1MB_3D_Engine.py demo --output-dir 3d_demo --size 1048576
python Dr_Moagi_1MB_3D_Engine.py encode your_file.bin your_file.dm3d --state predictor_policy.json
python Dr_Moagi_1MB_3D_Engine.py decode your_file.dm3d restored.bin
python Dr_Moagi_1MB_3D_Engine.py inspect your_file.dm3d
python -m unittest -v test_Dr_Moagi_1MB_3D_Engine.py
```

Optional PyTorch ANN demonstration:

```bash
pip install torch
python Dr_Moagi_1MB_3D_Engine.py neural-demo --steps 5 --checkpoint dm3d_neural.pt
```

Optional live GitHub public-event ingestion (requires internet):

```bash
python Dr_Moagi_1MB_3D_Engine.py fetch \
  https://api.github.com/repos/Lord-Xido/Jarvis-X/events \
  github_events.dm3d --max-bytes 1048576
python Dr_Moagi_1MB_3D_Engine.py decode github_events.dm3d github_events.json
```

The HTTPS API example is implemented but **has not been validated against a live upstream response in the offline execution environment**. GitHub API limits, upstream JSON changes and network access still apply.

## Verification receipt · 19 October 2026

| Property | Verified outcome |
|---|---:|
| Single Python program size | **1,000,000 bytes exactly** |
| Embedded 3D regression vectors | 11,706 |
| Self-test assertions | 11,769 passed |
| Additional independent unittest scenarios | 11 passed |
| Synthetic demo original payload | 1,048,576 bytes |
| Synthetic demo archive | 18,122 bytes |
| Synthetic ratio (input/archive) | 57.862:1 |
| Predictor weight mutations accepted/rejected | 3 / 29 |
| Demo restored SHA-256 | `2678a89a2be251584d9fe68691c6227e31f5552ecbd3a57641d58de37ec16f85` |
| Neural 4-step held-out MSE | 0.16237 → 0.16190 (synthetic shapes) |
| Local source SHA-256 | `d5c1312b6acf0d44b0fc66772f3eccabaea0ab46676a6125476da93311f2fdf0` |

**Important:** The high compression ratio is from a deliberately structured synthetic spatial pattern. Random or already-compressed data is not expected to compress similarly and can become slightly larger from archive headers and frame overhead. This standalone v2 archive is an experimental protocol, not a canonical Jarvis-X wire format or production-hardened secure container.

## Boundaries / next integration gates

1. Python local-file commit is atomic per output, but **not** a multi-file, distributed ADR-016 transaction coordinator. Optional model-policy checkpoints are separate state files.
2. `--fetch` is a one-time bounded network ingest, not a perpetually active hosted service.
3. Neural training is a separate lossy experiment; the reference codec's guaranteed byte recovery comes from exact reversible residuals and independent verification.
4. An archive SHA-256 is not a MAC or digital signature. Apply trusted signatures and provenance anchoring before exposing packet input to untrusted production systems.
5. The enormous address domain and speculative recursive iteration counts are symbolic and do not imply physical memory allocation or actual iteration throughput.
6. Model packet identity, cross-backend compatibility, native SIMD/CUDA, external benchmarks, enterprise IAM and deployment attestation remain future integration work.
