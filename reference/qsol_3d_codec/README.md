# QSOL 3D Auto-Encoding/Decoding Bytecode Reference

This directory contains the executable research reference for ADR-010.

## Scope

The VM implements a deterministic four-byte instruction profile:

```text
[ opcode:8 ][ dest:8 ][ src/coord:8 ][ modifier:8 ]
```

It models a 256-register SIMD-style toroidal lattice using signed saturating Q16.16 arithmetic. It is a bounded software reference only.

`ACTUATE_E` writes to an in-memory simulated actuation register. It does not perform device, RF, power-electronics, electron-transport, or other external hardware control.

## Canonical program

```text
01 10 00 10
01 11 01 33
02 20 10 06
03 30 20 10
04 40 30 40
07 50 40 00
05 60 40 9A
06 70 60 00
08 10 70 01
00 00 00 00
```

Compact form:

```text
01100010 01110133 02201006 03302010 04403040
07504000 0560409A 06706000 08107001 00000000
```

## Constant selectors

The fourth byte selects a Q16.16 constant for coefficient-bearing opcodes:

| Selector | Q16.16 | Decimal |
|---|---:|---:|
| `0x10` | `0x00010000` | `1.0` |
| `0x33` | `0x00003333` | `0.1999969482421875` |
| `0x40` | `0x00004000` | `0.25` |
| `0x9A` | `0x0000D99A` | `0.850006103515625` |

This expansion is required because a single 8-bit modifier cannot directly encode the wider Q16.16 values.

## Register-byte correction

The submitted draft is retained exactly in `PROGRAM_SUBMITTED_DRAFT`, but the executable canonical stream corrects four source bytes:

```text
R48  decimal 48  = 0x30, not 0x48
R64  decimal 64  = 0x40, not 0x64
R96  decimal 96  = 0x60, not 0x96
R112 decimal 112 = 0x70
```

Accordingly:

```text
ENCODE_3D    source 0x48 -> 0x30
HAMILTON_CHK source 0x64 -> 0x40
DECODE_3D    source 0x64 -> 0x40
ACTUATE_E    source 0x96 -> 0x60
```

## Fixed-point fixture

The canonical initialization sets `R16 = 1.0` uniformly over the torus. The normalized six-neighbor Laplacian is therefore zero:

```text
mean(neighbors(R16)) - R16 = 0
```

The encode/decode residual path remains zero. `SYNC_COMMIT` modifier `0x01` is defined as atomic residual-add:

```text
R16 <- sat_Q16(R16 + R112)
```

so the uniform fixture produces `max_commit_drift_q16=0`.

This verifies a discrete software fixed point only; it does not establish zero physical latency, zero energy dissipation, zero ringing, or literal electron-level actuation.

## Run

```bash
python reference/qsol_3d_codec/reference_vm.py
```

Expected tail:

```text
[CYCLE 010] PC 0x0024 HALT_LOOP     -> end-of-epoch fence

commit_generation=1
max_commit_drift_q16=0
fixed_point=True
```

## Test

```bash
cd reference/qsol_3d_codec
python -m unittest -v test_reference_vm.py
```

The dedicated GitHub Actions workflow runs the reference on Python 3.10 and 3.13 and verifies the fixed-point receipt.
