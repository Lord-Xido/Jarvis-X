# ADR-025: AVX-512 blockwise volumetric fold kernel

- **Status:** Proposed
- **Date:** 2026-09-21
- **Decision scope:** native research/runtime acceleration surface

## Context

A proposed `vOmegaXi+` kernel described an `8192^3` bit-space, three external
state buffers, a forward/reverse 512-bit traversal, a ternary Boolean transform,
and an entropy/delta writeback.

The proposal contained four implementation mismatches that must be resolved
before it can be treated as executable repository evidence:

1. `8192^3` bits is 64 GiB and therefore `0x200000000` `uint64_t` words, not
   `0x40000000` words.
2. The function interface exposed `total_words` as an argument but the body
   overwrote it with a fixed constant.
3. The zero-source sparse bypass is not semantics-preserving for
   `(src & mirror) XOR (~src)` because a zero source maps to all ones.
4. The reverse index performs a 512-bit block reflection, not an exact per-bit
   3D central reflection.

The hand-written hexadecimal stream is non-authoritative because branch
displacements, immediates, register allocation, and ABI changes alter machine
bytes.

## Decision

Add `native/avx512-volumetric-fold/` as a bounded reference module.

The module:

- takes `total_words` from the caller and requires a multiple of eight;
- provides a scalar reference implementation;
- provides preprocessed AVX-512 assembly for System V AMD64 and Windows x64;
- dispatches at runtime to AVX-512F+DQ when available;
- uses unaligned vector loads/stores for safe alignment semantics;
- implements `(~src) | mirror == (src & mirror) XOR (~src)`;
- writes `entropy = next XOR src`;
- removes the non-equivalent sparse-zero shortcut from the exact path;
- treats the 8192^3 extent as a logical, caller-managed extent compatible with
  sparse or mapped backing rather than mandatory 192 GiB triple residency;
- generates machine code from assembly rather than preserving hand-authored raw
  bytes.

## Geometry semantics

Let `B = total_words / 8` be the number of 512-bit blocks. For block `j`:

```text
mirror_block(j) = B - 1 - j
```

Lane `l` is paired with the same lane `l` in that mirrored block.

This is a 1D blockwise reflection of the flattened representation. It is not
promoted to a coordinate-exact 3D reflection without an explicit bit-to-voxel
layout and the required intra-block reversal or mapping.

## Memory and performance boundary

The full logical lattice has 64 GiB per bit-plane. Three fully materialized
planes require 192 GiB of payload. This ADR makes no throughput, latency,
convergence, or residency claim beyond measurements from a named benchmark on
a named machine.

The zero-block optimization may be reintroduced only as a separately named
alternative operator, or after the fold algebra is changed so zero is a proven
fixed point. It must not be described as a semantics-preserving optimization of
this operator.

## Trust boundary

This kernel is a pure local transform over explicit buffers. It remains outside
the authoritative Jarvis-X commit path unless a higher-level validated runtime
explicitly incorporates its outputs.

## Validation

The reference tests verify:

- scalar and dispatched outputs against the Boolean formula;
- entropy/delta equality;
- zero-input behavior;
- argument and multiple-of-eight validation;
- native assembly compilation with AVX-512F+DQ;
- Windows x64 COFF assembly generation with Clang.

Promotion from Proposed should require repository CI and benchmark evidence for
any performance claims attached to the kernel.
