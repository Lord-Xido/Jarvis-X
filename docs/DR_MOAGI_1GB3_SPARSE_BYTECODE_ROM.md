# Dr Moagi 1 GB × 1 GB × 1 GB Sparse 3D Bytecode ROM

**Status:** Reference implementation profile  
**Scope:** Bounded sparse AE/AD ROM demonstrator  
**Canonical authority:** The Jarvis-X authority VM and its normal validation/commit rules remain authoritative.

## Purpose

This profile makes the 1 GB × 1 GB × 1 GB formulation operational as a sparse virtual byte-address space.

Each axis contains one billion byte coordinates:

\[
\mathcal V = \{0,\ldots,10^9-1\}^3,
\qquad
|\mathcal V| = 10^{27}.
\]

The implementation does **not** allocate a dense \(10^{27}\)-byte cube. It materializes only explicitly active 64³ tiles and executes a compact ROM program over those tiles.

This follows the same governing distinction used by the canonical cloud ROM profile:

\[
\boxed{\text{ROM describes the machine; sparse state supplies the active world.}}
\]

## Relationship to existing Jarvis-X bytecode

This ROM is a bounded reference profile, not a replacement for the canonical Jarvis-X authority VM and not a competing top-level ISA. Its decoded or corrected state remains a candidate until the enclosing runtime's ordinary validation and commit/rollback rules accept it.

The profile is also distinct from DMKB-1. DMKB-1 remains the bounded binary lowering/transport format for the QSOL 3D graphics codec; this module is a sparse 3D byte-volume AE/AD reference VM.

## Logical geometry

```text
axis span            1,000,000,000 byte coordinates
logical positions    10^27
active tile edge     64 bytes
active tile volume   64^3 = 262,144 bytes
latent tile edge     8
latent tile volume   8^3 = 512 bytes
tiles per axis       15,625,000
```

The required binary hierarchy depth to span one logical axis is

\[
\lceil \log_2(10^9) \rceil = 30.
\]

## Execution pipeline

```text
VIRTUAL 3D BYTE FIELD
        |
        v
SPARSE 64^3 ACTIVE TILE
        |
        v
ENCODE: 64^3 -> 8^3 QUANTIZED LATENT
        |
        v
INWARD RESIDUAL-GUIDED LATENT REFINEMENT
        |
        v
DECODE: 8^3 -> 64^3 RECONSTRUCTION
        |
        v
SIGNED RESIDUAL R = X - X_hat
        |
        v
CORRECTION X_corr = X_hat + R
        |
        v
STRICT BYTE-EQUALITY VERIFICATION
```

The current reference encoder uses block averages with quantization. The residual path preserves the information not represented by that lossy latent. Therefore exact recovery in the strict demonstration is produced by reconstruction plus the explicit signed residual, not by claiming that the 512-byte latent alone losslessly represents arbitrary 262,144-byte tiles.

## ROM container

The ROM begins with a fixed 64-byte little-endian header followed by fixed-width 32-byte instructions.

Header fields:

```text
magic          8 bytes   "DM3DROM1"
version        u32       1
axis_bytes     u64       1,000,000,000
tile_edge      u32       64
latent_edge    u32       8
instr_count    u32
code_offset    u64       64
flags          u64
reserved       16 bytes
```

Instruction fields:

```text
opcode         u8
flags          u8
reserved       u16
x,y,z          u32 x 3
p0,p1,p2,p3    u32 x 4
```

Current opcode surface:

```text
0x00 NOP
0x01 FILL_TILE
0x10 ENCODE
0x11 REFINE
0x12 DECODE
0x13 RESIDUAL
0x14 CORRECT
0x15 VERIFY
0x16 DROP_TILE
0x20 JUMP
0xFF HALT
```

## Residual refinement

For an active source tile \(X\), initial latent \(Z_0\), and decoder \(D\):

\[
\hat X_j = D(Z_j),
\qquad
R_j = X - \hat X_j.
\]

Residuals are block-pooled back into latent space and applied as bounded corrections:

\[
Z_{j+1} = \operatorname{clip}_{[0,255]}
\left(Z_j + \alpha P(R_j)\right).
\]

The reference implementation bounds the refinement iteration count and never interprets unbounded recursion as literal infinite execution.

## Exact correction path

After decoding,

\[
R = X - \hat X.
\]

The strict reference correction is

\[
\boxed{X_{corr}=\hat X+R=X.}
\]

This is deliberately explicit: arbitrary input bytes are only byte-exact when the residual information needed for exact recovery is retained.

## Verification

`VERIFY` compares the candidate reconstruction with the authoritative active tile. Strict verification (`flags & 0x01`) raises on mismatch.

The focused test suite checks:

1. the 1 GB-per-axis virtual geometry;
2. fixed ROM header/instruction sizes;
3. sparse addressing at a far-edge tile coordinate;
4. encode/refine/decode/residual/correct execution;
5. strict byte-exact verification;
6. rejection of corrupt ROM magic.

Run:

```bash
pytest -q tests/test_dm3d_rom.py
```

Build and execute the demonstration ROM:

```bash
python -m jarvisx.dm3d_rom demo dr_moagi_3d_1gb3.rom
```

Inspect it:

```bash
python -m jarvisx.dm3d_rom inspect dr_moagi_3d_1gb3.rom
```

## Capability boundary

The profile demonstrates sparse virtual addressing, bounded 3D block autoencoding, residual-guided refinement, explicit exact residual correction, deterministic ROM serialization, and verification.

It does not claim that a physically resident \(10^{27}\)-byte memory exists, that arbitrary data can be losslessly compressed into the 8³ latent alone, or that recursive refinement guarantees unlimited capability growth.
