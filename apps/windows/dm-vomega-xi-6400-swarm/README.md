# DM–vΩΞ⁺ 6400³ Virtual VRAM Swarm ISA

This Windows research artifact implements the 32-bit fixed-width swarm ISA for the DM–vΩΞ⁺ 3D auto-encoding/decoding model.

## Scope

The logical address fabric spans **6400 GiB per axis**. It is a sparse 3D virtual coordinate space, not a literal `6400^3 GiB` allocation. The reference executable materializes only 2,048 resident 64 KiB page slots and runs 64 virtual agents.

The machine word is:

```text
31         24 23     20 19     16 15     12 11                      0
+---------------+-------+---------+---------+------------------------+
| Opcode (8-bit)| Rd/Ra | Rs1/Rv  | Rs2     | Immediate / SubOp (12) |
+---------------+-------+---------+---------+------------------------+
```

Arithmetic instructions interpret their operands as signed saturating Q16.16. `R10` is a raw resident-page handle, `R13` is a raw mode value, and `R14` is a typed scalar/synchronization register. The storage width remains 32 bits for all registers.

## Canonical stream and executable stream

`PROGRAM_CANONICAL_V1` preserves the supplied 13-word stream bit-for-bit, including:

```text
0x50A12300  HASH_ADDR R10,R1,R2,R3
0x10FA0000  VREAD3D R15,R10
0x30CF0000  EVAL_FGRAD R12,R15
0x31710000  EVAL_FLATENT R7,R1
0x32810000  EVAL_FREPEL R8,R1
0x21778000  Q16ADD R7,R7,R8
0x2177C000  Q16ADD R7,R7,R12
0x21447000  Q16ADD R4,R4,R7
0x21114000  Q16ADD R1,R1,R4
0x40BF0000  ENCODE_STEP R11,R15
0x11AB0000  VWRITE3D R10,R11
0x0F000000  SYNC_SWARM
0xFF000000  HALT
```

The comment in the supplied kernel states Euler `dt = 0.01`, but the two raw `Q16ADD` words above are mathematically a unit step. `PROGRAM_ENCODE_DT001` is therefore the executable v1.1 stream: it inserts `Q16MUL` operations using `R14 = 655`, the nearest Q16.16 encoding of 0.01, before the velocity and position additions.

`HASH_ADDR` uses `imm[11:8]` as the third register selector, which is why `0x50A12300` encodes `R3`. `VWRITE3D` uses store-specific field semantics: `[23:20]` is the address register and `[19:16]` is the value register, making `0x11AB0000` mean `VRAM[R10] <- R11`.

## Auto-execution

Each virtual agent alternates between an encode epoch and a decode epoch:

```text
position -> HASH_ADDR -> VREAD3D
        -> error/latent/repulsion forces
        -> Q16.16 kinetic integration
        -> ENCODE_STEP -> VWRITE3D
        -> SYNC_SWARM -> HALT
        -> DECODE epoch -> repeat
```

`SYNC_SWARM` is a real global barrier in the reference scheduler: an agent cannot advance beyond it until every active agent is waiting or halted.

## Build

From Linux with LLVM installed:

```bash
./build-windows.sh
```

The build generates a deterministic Windows x86-64 PE32+ console executable without a C runtime dependency.

## Trust boundary

This is a bounded research/runtime surface. Its sparse VRAM writes are **not authoritative Jarvis-X commits**. It has no network, filesystem, market, medical, infrastructure, or device authority. Authoritative Jarvis-X state still flows through `jarvisx.system_runtime` and its verification/audit/commit boundary.
