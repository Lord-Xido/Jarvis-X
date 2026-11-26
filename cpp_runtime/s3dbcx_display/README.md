# S3D-BCx — bounded bytewise display ANN adapter

Status: **CPU research reference, not hardware display driver or canonical authority VM**.

This subsystem supplements the existing Dr Moagi geometric-bytecode ANN and DMKB-1 profile. It does not replace the Jarvis-X outer candidate-first verification/authority boundary. The local CPU `ECHO` commits only within the demonstration process.

## Build / run

```sh
g++ -std=c++17 -O2 -Wall -Wextra -Werror cpp_runtime/s3dbcx_display/s3dbcx.cpp -o /tmp/s3dbcx
/tmp/s3dbcx
```

Expected output: `S3D-BCx PASS: pack, ROM, 3D transport, two epochs, 14427360 framebuffer bytes, XOR residual`.

## Binary format

Touch packet is 64 bits: OP[63:60], X[59:48], Y[47:36], PRESS[35:24], TIME[23:12], FLAGS[11:0]. 12-bit X/Y are mapped to pixel indices using `(coordinate * dimension) >> 12`, which is in range [0, dimension-1]. Example: `0x2800800400001000` = MOVE, X=2048, Y=2048, P=1024, T=1, F=0. Serialized little-endian bytes: `00 10 00 40 00 80 00 28`.

Each of the 8 ROM instructions is 8 bytes in little-endian order: opcode, mode, dst, src, imm32. ROM is 64 bytes: ENC, GEO, RICCI, REC, TRANS, DEC, SCAN, ECHO. Instructions are supervisory dispatch entries, **not** proof that the workload takes eight hardware instructions or eight cycles.

128-byte signed-int8 lattice: 8 x 4 x 4 indexed by `x + 8*y + 32*z`. Every X row maps to a 64-bit word. The recurrence uses 6 toroidal neighbors and a fixed-point signed accumulator with defined flooring shift, followed by saturation. `TRANS` uses the bijection `(x,y,z) -> ((x+y)&7,y,z)`. `RICCI` is currently a bounded six-neighbor smoothing surrogate; mathematical Ricci curvature is NOT computed. `ENC` currently injects packet bytes into an otherwise zero latent input; it is not a trained encoder.

`DEC` produces a deterministic procedural BGRA8 frame of 2796 x 1290 = 14,427,360 bytes. `SCAN` only simulates presentation. **No iOS, OLED, Metal, Apple hardware, ANN training, or 120 Hz throughput is claimed.** The frame requires physical generation/memory bandwidth despite the small latent.

## Verification

Local C++17 test covers a canonical input word, little-endian serialization, ROM layout, 128 unique geometric permutation destinations, two executed epochs, framebuffer allocation and alpha coverage, and exact XOR residual reconstruction for modified target bytes. The residual can grow to the size of the frame: a 128-byte latent is not guaranteed to losslessly encode arbitrary 14.4 MB frames.

The module does not prove authorization to actuate displays. For integration, run the output through the existing outer Jarvis-X candidate -> verify -> commit/rollback path. CI integration, GPU implementation, real Ricci geometry, learned decoder, throughput and iPhone device tests are follow-up work.

## Cycle

```text
touch 8 B -> ENC -> GEO -> RICCI surrogate -> REC 128 B
       -> TRANS -> DEC BGRA8 14,427,360 B
       -> simulated SCAN -> local ECHO -> next epoch
```
