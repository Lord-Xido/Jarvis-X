# React Artifact (9): Jarvis-X Quine-ROM Compatibility Bridge

**Status:** Executable research adapter, not a replacement for canonical authority VM.  
**Source:** Uploaded React Artifact (9).html SHA-256 `7904a388fbb68b4fe4434f7f687ab745e1d35eae4ffac16c4f944401d632f554` (2026-10-30).  
**Module:** `jarvisx.quine_rom_bridge`. **Tests:** `tests/test_quine_rom_bridge.py`.

The uploaded browser artifact renders an animated sparse cube and instruction trace labeled QUINE ENGINE ROM. The screen's simulated CPU address pointer is not evidence of physically materialized 1GB-cubed RAM/ROM or independently executing machine instructions. This adapter gives its six named instructions testable, finite semantics.

## Logical geometry

The UI's `1,073,741,824³ bits` is `2**90` logical bit positions, addressed as three virtual `2**30`-bit axes. A bijective Morton interleave maps `[0,2**90)` to coordinate triples. Only finitely touched chunks are materialized. The existing `jarvisx.dm3d_rom` declares a distinct `1,000,000,000³` virtual **byte-coordinate** volume; the two profiles are not conflated.

## ISA operational semantics

| Opcode | Hex | Effect |
|---|---|---|
| Q_ENC | E1 | Split payload into active 64³-byte chunks, encode using existing `dm3d_rom.encode_tile` to 8³ latent and retain bytewise XOR residual |
| Q_FOLD | F0 | Emit Morton descriptors of active chunk starts; model 64³ → 8³ latent geometry |
| Q_DEC | D1 | Decode with `dm3d_rom.decode_tile` and XOR residual; verify chunk and full-source SHA-256 |
| Q_OPT | 0E | Select zlib compression level for latent and residual, accepting only identical decode |
| Q_JMP_SELF | CC | Enforce verified reconstruction equality; next cycle runs only within explicit max 32-cycle bound |
| Q_NOP | 00 | No operation |

Fixed point here means the **finite byte-exact identity** `D(E(X),R)=X` because a full residual is retained. It does not prove that an arbitrary neural map converges, that the UI is a true self-replicating code quine, or that a physical 2**90-bit memory exists. This profile is not an autonomous optimizer and does not mutate canonical source code.

## Execute

```bash
python -m jarvisx.quine_rom_bridge integrate 'React Artifact(9).html' --output react_artifact9.qrom --cycles 2
python -m jarvisx.quine_rom_bridge restore react_artifact9.qrom --output restored.html
sha256sum 'React Artifact(9).html' restored.html
pytest -q -o addopts='' tests/test_quine_rom_bridge.py
```

The QROM archive contains a bounded binary header and JSON manifest with zlib-compressed latents and XOR residuals, enough to restore the exact source without executing its embedded JavaScript. Decoded bytes remain a **candidate** for the enclosing authority VM's normal validation/commit/rollback policy. No training, networking, remote commands, or unbounded recursion is performed.
