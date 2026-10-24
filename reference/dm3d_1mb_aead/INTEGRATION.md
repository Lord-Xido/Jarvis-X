# DM3D 1MB lossless codec — isolated research integration

This directory imports the exact byte-for-byte **1,000,000-byte** Python reference
release from 2026-10-19, its independent tests, release manifest, and operational
README. The original source SHA-256 must remain:
`d5c1312b6acf0d44b0fc66772f3eccabaea0ab46676a6125476da93311f2fdf0`.

## Review-first boundary

- This code is **not** installed as the canonical `jarvisx` package, VM opcode,
  CodexVM interpreter, production service, or new public codec wire protocol.
- `DM3D-AEAD-v2` is the reference's historical version string. Despite the name,
  SHA-256 alone is **not authenticated encryption** or attacker-resistant integrity.
  Do not advertise it as secure AEAD or expose archives to untrusted production inputs.
- Virtual axes of 8 trillion bit coordinates are *address space semantics*, not
  physically allocated memory or an execution-speed claim. Bounded 32³ tiles store
  **bytes**, not packed individual bits.
- The local lossless file transaction is atomic per output path; it is **not**
  ADR-016 distributed multi-domain atomic promotion. The predictor state file is
  stored separately from the archive.
- The optional PyTorch fixed-point/ANN experiment is **lossy**, untrained in this
  repository, and outside the byte-exact archive format. No PyTorch dependency
  is required for the CI-gated lossless path.
- WWW `fetch` is one bounded public GitHub events request. Network ingestion is
  optional and is not exercised in hermetic CI.
- No pretrained semantic capability, AGI, MP4 compatibility, GPU scaling,
  verified world understanding, or perpetual autonomous operation is claimed.

## Run locally

```bash
cd reference/dm3d_1mb_aead
python Dr_Moagi_1MB_3D_Engine.py selftest --vectors
python -m unittest -q test_Dr_Moagi_1MB_3D_Engine.py
python Dr_Moagi_1MB_3D_Engine.py demo --output-dir demo_out
python Dr_Moagi_1MB_3D_Engine.py encode input.bin input.dm3d
python Dr_Moagi_1MB_3D_Engine.py decode input.dm3d restored.bin
```

The [path-scoped CI](../../.github/workflows/dm3d-1mb-reference.yml)
verifies the exact file size and source SHA-256, executes the 11 independent
regression scenarios, and runs 11,769 embedded assertions (as recorded in the
release manifest) across Python 3.10, 3.12 and 3.13.

## Canonical integration contract

The Jarvis-X canonical scope remains:

`typed input -> encode -> refine -> decode -> CTR -> candidate verify
-> atomic commit OR rollback`.

For a production integration, implement an ADR-016/ADR-017 conformance adapter
with typed model/codec identifiers, receipt lineage, authentication,
fuzzing, hard resource caps, external evidence gates, and independent
compatibility/recovery testing. Never substitute local numerical convergence
for independent world correspondence or authorization.

References:
- [ADR-016](../../docs/adr/0016-canonical-typed-state-transaction-and-geometry-profiles.md)
- [ADR-017](../../docs/adr/0017-dr-moagi-operational-autoencoding-equation.md)
