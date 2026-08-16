# ADR-009: DM–vΩΞ⁺ 6400³ Virtual VRAM Swarm ISA

- **Status:** Proposed
- **Date:** 2026-08-16
- **Decision scope:** native Windows research/runtime surface

## Context

ADR-008 established QSOL kinetic 3D visualization as a non-authoritative research surface. The next layer is a native fixed-width instruction engine that can express the same spatial recurrence with deterministic Q16.16 arithmetic and a bounded sparse 3D virtual address fabric.

The architecture specifies a 32-bit word containing an 8-bit opcode, three 4-bit register fields, and a 12-bit immediate/sub-op field. It also specifies a 16-entry register file, sparse 3D VRAM access, kinetic force operators, encode/decode operations, a swarm barrier, and halt semantics.

## Decision

Add `apps/windows/dm-vomega-xi-6400-swarm/` as the reference native ISA implementation.

The reference engine:

1. preserves the supplied 13-word encode stream bit-for-bit as `PROGRAM_CANONICAL_V1`;
2. defines missing opcode `0x50` as `HASH_ADDR`;
3. defines `HASH_ADDR` third-source encoding as `imm[11:8]`;
4. defines `VWRITE3D` store fields as address `[23:20]` and value `[19:16]`;
5. uses saturating signed Q16.16 arithmetic with 64-bit multiply intermediates;
6. treats `R10`, `R13`, and control uses of `R14` as typed 32-bit values rather than arithmetic Q16.16 values;
7. treats `R10` as a sparse resident-page handle, while the page table retains the full 3D page tuple;
8. adds an executable v1.1 stream that inserts Q16 multiplications for the stated Euler `dt=0.01`;
9. runs 64 bounded virtual agents over 2,048 resident 64 KiB sparse pages;
10. alternates encode and decode epochs behind a real global barrier.

## Why the dt-corrected stream exists

The supplied V1 sequence describes `dt=0.01`, but

```text
VEL_X <- VEL_X + FRC_X
POS_X <- POS_X + VEL_X
```

is a unit-step update under the declared `Q16ADD` semantics. The reference therefore retains those original words for compatibility and executes a V1.1 sequence containing explicit `Q16MUL` steps by the Q16.16 scalar `655`, which is the nearest representable value to `0.01`.

## Virtual-memory interpretation

`6400 GiB × 6400 GiB × 6400 GiB` names the coordinate extent of the virtual 3D fabric. It is not physically allocated. Each axis is mapped to 64 KiB page coordinates, and only a bounded resident working set is materialized.

A 32-bit `VOX_ADDR` cannot uniquely contain the full 3D key. It is therefore an opaque resident-page handle returned by `HASH_ADDR`; the page table stores the collision-resolving `(page_x, page_y, page_z)` tuple.

## Trust boundary

This native swarm engine remains outside the authoritative Jarvis-X task-state path. `VWRITE3D` mutates only the engine's research VRAM state. It is not equivalent to a `jarvisx.system_runtime` commit and cannot bypass capability projection, verification, audit, or authoritative commit.

## Validation

Repository validation must check:

- exact preservation of the supplied 13 canonical machine words;
- presence of the dt-corrected execution stream;
- sparse-address and typed-register invariants;
- deterministic cross-build of a Windows x86-64 PE32+ artifact;
- expected `KERNEL32.dll`-only import surface;
- no external side-effect adapters.

## Promotion

Promote this ADR to **Accepted** only after the implementation PR and Windows artifact workflow pass repository validation.
