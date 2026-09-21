# ADR-010: QSOL 3D Auto-Encoding/Decoding Bytecode Profile

- **Status:** Proposed
- **Date:** 2026-08-20
- **Decision scope:** bounded Q16.16 research interpreter for the DM–vΩΞ⁺ engine

## Context

ADR-008 established QSOL as a bounded, non-authoritative research surface. ADR-009 established a separate 32-bit DM–vΩΞ⁺ swarm ISA with a 16-register encoding. This decision defines a new QSOL codec profile for a 256-register namespace and a four-byte instruction word:

```text
[ opcode:8 ][ dest:8 ][ src/coord:8 ][ modifier:8 ]
```

The profile expresses a deterministic 3D toroidal stencil, local Q16.16 transform encoding/decoding, a bounded Hamiltonian/error proxy, an internal actuation register, and an atomic residual commit.

This profile is **not** binary-compatible with ADR-009. It is a sibling research ISA profile.

## Decision

### 1. Fixed-width instruction word

Every instruction is exactly four bytes, big-endian by field position:

```text
byte 0: opcode
byte 1: destination register ID, 0..255
byte 2: source or coordinate-register ID, 0..255
byte 3: opcode-specific modifier
```

The fourth byte is deliberately opcode-specific. It is not claimed to directly contain an arbitrary Q16.16 immediate.

### 2. Q16.16 constant-selector table

The supplied program needs constants wider than eight bits. The modifier therefore selects a profile-local constant table for opcodes that require scalar coefficients:

```text
selector  Q16.16 value  decimal
0x10      0x00010000    1.0
0x33      0x00003333    0.1999969482421875
0x40      0x00004000    0.25
0x9A      0x0000D99A    0.850006103515625
```

This resolves the width contradiction between an 8-bit modifier and values such as `0x00010000`, `0x4000`, and `0xD99A`.

### 3. Opcode table

| Opcode | Mnemonic | v1 reference semantics |
|---|---|---|
| `0x00` | `HALT_LOOP` | End-of-epoch fence. The host may start another epoch explicitly. |
| `0x01` | `LDF_PSI` | Load a Q16.16 scalar selected by byte 3 into the destination register across the current lattice. Byte 2 is a source tag (`0x00=MEM_NULL`, `0x01=MEM_OFF`). |
| `0x02` | `FETCH_NB` | Fetch six orthogonal toroidal neighbors of source register byte 2 into a typed six-value vector at destination. Byte 3 must be `0x06`. |
| `0x03` | `LAPLACE_3D` | Compute normalized six-point Laplacian `mean(neighbors) - center`. Byte 2 names the neighbor-vector register and byte 3 names the center scalar register. |
| `0x04` | `ENCODE_3D` | Q16.16 local transform `dest = src * K_encode`, with `K_encode` selected by byte 3. |
| `0x05` | `DECODE_3D` | Q16.16 local reconstruction transform `dest = src * K_decode`, with `K_decode` selected by byte 3. |
| `0x06` | `ACTUATE_E` | Copy the decoded value into an **internal simulated actuation register**. No device, electron, power, RF, or other external hardware I/O is authorized by this opcode. |
| `0x07` | `HAMILTON_CHK` | Evaluate the bounded reference energy proxy `H_eff = 0.5 * Xi^2` and store it in destination. Byte 3 selects the reference profile; `0x00` is v1. |
| `0x08` | `SYNC_COMMIT` | Atomic lattice commit. Modifier `0x01` means residual-add: `dest <- sat_Q16(dest + src)` using one pre-commit snapshot. Modifier `0x00` is replace mode. |

The discrete operator implemented by `LAPLACE_3D` is

```text
L_norm(Psi)[i,j,k] =
    (Psi[i+1,j,k] + Psi[i-1,j,k]
   + Psi[i,j+1,k] + Psi[i,j-1,k]
   + Psi[i,j,k+1] + Psi[i,j,k-1]) / 6
   - Psi[i,j,k]
```

with all indices wrapped modulo lattice extent. This is a normalized form of the standard unit-spacing six-point Laplacian; the factor of six is absorbed into the transform scale.

### 4. Canonical corrected byte stream

The submitted draft described registers by decimal labels such as `R48`, `R64`, and `R96`, but several source bytes encoded those decimal digits as hexadecimal values. The canonical stream uses the actual byte IDs:

```text
01 10 00 10  # LDF_PSI      R16,  MEM_NULL, CONST[0x10] = 1.0
01 11 01 33  # LDF_PSI      R17,  MEM_OFF,  CONST[0x33] = 0x00003333
02 20 10 06  # FETCH_NB     R32,  R16,      six orthogonal neighbors
03 30 20 10  # LAPLACE_3D   R48,  R32,      center=R16
04 40 30 40  # ENCODE_3D    R64,  R48,      gain=0.25
07 50 40 00  # HAMILTON_CHK R80,  R64,      profile=0
05 60 40 9A  # DECODE_3D    R96,  R64,      gain=0.8500061035
06 70 60 00  # ACTUATE_E    R112, R96,      internal sink
08 10 70 01  # SYNC_COMMIT  R16,  R112,     residual-add
00 00 00 00  # HALT_LOOP                         epoch fence
```

Raw hexadecimal stream:

```text
01100010 01110133 02201006 03302010 04403040
07504000 0560409A 06706000 08107001 00000000
```

### 5. Corrections relative to the submitted draft

The profile makes the following arithmetic/encoding corrections explicit:

1. An 8-bit modifier cannot directly hold a 16-bit or 32-bit Q16.16 constant. The v1 constant-selector table supplies those values deterministically.
2. `LAPLACE_3D` uses byte 3 as a second register ID (`R16`) rather than as a scalar modifier.
3. `R48` is register ID `48 decimal = 0x30`; therefore `ENCODE_3D` must read source byte `0x30`, not `0x48`.
4. `R64` is `0x40`; therefore `HAMILTON_CHK` and `DECODE_3D` read `0x40`, not `0x64`.
5. `R96` is `0x60`; therefore `ACTUATE_E` reads `0x60`, not `0x96`.
6. `0x9A` is a selector for Q16.16 `0x0000D99A`; it is not itself the value `0.85`.
7. The two initial `LDF_PSI` words are separate instructions. `R16=0x00010000` and `R17=0x00003333`; they do not arithmetically combine into `R16=0x00013333` without an explicit add instruction.
8. The supplied `R17` offset is initialized but not consumed by the v1 dataflow. It is retained for compatibility and exposed as a dead-value diagnostic rather than given a hidden side effect.

### 6. Correct execution trace

The single-issue reference trace is ten four-byte instructions:

```text
[CYCLE 001] PC 0x0000 LDF_PSI      -> R16 = 0x00010000
[CYCLE 002] PC 0x0004 LDF_PSI      -> R17 = 0x00003333
[CYCLE 003] PC 0x0008 FETCH_NB     -> R32 = six wrapped R16 neighbors
[CYCLE 004] PC 0x000C LAPLACE_3D   -> R48 = mean(R32) - R16
[CYCLE 005] PC 0x0010 ENCODE_3D    -> R64 = R48 * 0.25
[CYCLE 006] PC 0x0014 HAMILTON_CHK -> R80 = 0.5 * R64^2
[CYCLE 007] PC 0x0018 DECODE_3D    -> R96 = R64 * 0.8500061035
[CYCLE 008] PC 0x001C ACTUATE_E    -> R112/internal actuation register updated
[CYCLE 009] PC 0x0020 SYNC_COMMIT  -> R16 += R112 atomically over the lattice
[CYCLE 010] PC 0x0024 HALT_LOOP    -> end-of-epoch fence
```

For the supplied uniform fixture, `R16=1.0` at every toroidal coordinate. Therefore

```text
L_norm(Psi) = 0
R48 = R64 = R80 = R96 = R112 = 0
SYNC_COMMIT residual = 0
max commit drift = 0
```

so the fixture is a valid discrete fixed point. This is an arithmetic property of the reference lattice; it is not evidence of zero physical latency, zero thermal dissipation, zero inductive ringing, or literal electron-level actuation.

### 7. Auto-encoding interpretation

The v1 reference instruction `ENCODE_3D` is a deterministic local transform over a toroidal field. By itself, multiplication by `0.25` does **not** reduce tensor cardinality or establish a compression ratio. A future profile may add explicit latent subsampling, quantization, entropy coding, learned transform tables, or sparse index selection.

Likewise, `DECODE_3D` in v1 is a deterministic reconstruction transform, not a claim of trained autoencoder fidelity.

### 8. Fixed-point semantics

`SYNC_COMMIT ... 0x01` is a residual update:

```text
Psi_(t+1) = sat_Q16(Psi_t + DeltaPsi_t)
```

A fixed point is detected by measured residual/drift criteria, for example

```text
max_abs(DeltaPsi_t) <= epsilon_commit
```

The phrase `I AM = I DESCRIBE` remains a mnemonic for this self-consistent state and does not alter the numerical acceptance test.

## Trust boundary

This profile inherits ADR-008 and ADR-009 boundaries:

- QSOL research state is not authoritative Jarvis-X task state.
- `ACTUATE_E` writes only an internal interpreter register in the reference implementation.
- `SYNC_COMMIT` commits only the simulated lattice/register state owned by this VM.
- No network, device, RF, power-electronics, filesystem, shell, market, medical, infrastructure, or privileged external adapter is implied by these opcodes.
- Physical latency, energy, thermal behavior, ringing, and electron transport require independent hardware measurement and are not inferred from bytecode semantics.

## Validation requirements

The reference implementation must test:

1. exact four-byte instruction decoding;
2. constant-selector expansion;
3. corrected register IDs through the full dataflow;
4. six-neighbor toroidal wraparound;
5. normalized Laplacian arithmetic;
6. saturating Q16.16 multiplication and commit;
7. deterministic Hamiltonian proxy;
8. internal-only actuation semantics;
9. atomic residual-add commit;
10. zero-drift behavior of the uniform fixed-point fixture;
11. exact program-counter trace `0x0000..0x0024`;
12. preservation of the submitted draft stream as a non-canonical compatibility fixture so encoding corrections remain auditable.

## Consequences

### Positive

- the proposed four-byte bytecode becomes arithmetically representable;
- register addressing is unambiguous across the 256-register namespace;
- the toroidal stencil has executable boundary semantics;
- the fixed-point claim is reduced to a measurable residual invariant;
- hardware/thermal claims remain outside the simulator trust boundary;
- the original draft can be preserved while a corrected executable stream becomes canonical for this profile.

### Trade-offs

- byte 3 is opcode-specific rather than a uniform immediate field;
- the v1 encoder/decoder are transform primitives, not yet a rate-distortion codec;
- `R17` is currently an unused compatibility value;
- this profile intentionally differs from the 16-register encoding accepted in ADR-009.

## Status

**Proposed.** Promotion to Accepted requires a reviewable reference implementation, passing conformance tests, and repository CI.