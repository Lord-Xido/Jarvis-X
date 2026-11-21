# Dr Moagi Inward1000: bounded 1000³ sparse quine and Merkle octree

**Status:** Tested reference implementation. **Source basis:** React Artifact(10) and PR #436. **Date:** 2026-10-30.
**Constraints:** ADR-015 provenance; ADR-016 candidate-first gate; ADR-017 operator framework; ADR-031 self-hosting boundary; ADR-034 bitwise XOR/popcount closure.

## Logical geometry, not 1,000,000,000 physical operations

The logical memory extent is `1000 × 1000 × 1000 = 1,000,000,000` voxel positions. No dense cube is allocated. Only nonzero input bytes and their occupied 3D ancestor nodes are materialized.

Let `d_0 = 1000`, `d_(l+1) = ceil(d_l/2)` until `d_10 = 1`. The dimensional sequence is:

`1000 → 500 → 250 → 125 → 63 → 32 → 16 → 8 → 4 → 2 → 1`.

The position of byte index `i` is `(i % 1000, floor(i / 1000) % 1000, floor(i / 10^6))`. This is reversible using `i = x + 1000(y + 1000z)` for valid integer coordinates.

The inward projection of a child voxel is `p_l(x,y,z)=(floor(x/2),floor(y/2),floor(z/2))`. The position and ordered slots of the up to eight children are included in a SHA-256 Merkle parent digest. Nonexistent children use a distinguished empty digest. Only occupied paths are stored. The root digest binds the original input byte length as well as the hierarchy, so missing trailing zero bytes cannot silently change length.

**Critical:** Merkle folding is *not* a reversible compression transform. The source bytes remain in the sparse leaf payload map, from which the decoder recovers the original buffer. The Merkle tree is an integrity/index summary of that data.

## Exact source fixed point

A source byte sequence `B` is encoded and independently reconstructed as `B_hat`. Exact binary closure requires `popcount(B XOR B_hat) = 0` with equal lengths and SHA-256 integrity checked. The separate classical 161-byte quine also reproduces its own source exactly 1,000 times; the full HTML has not been shown to be a quine.

The inward loop uses shadow per-patch overrides and recomputes hashes on just affected ancestor paths, at most one parent per level for a single edited byte. Each source-change candidate is rejected because this *particular contract* permits only bit-identical source states. This is deliberate: accepting beneficial code changes requires a different authenticated functional-correctness gate.

An unchanged candidate needs no rehash because its previously validated root remains unchanged. That is cached verification, **not** 1,000 real rescans. The `deepAudit()` function independently rebuilds the entire hierarchy, recomputes source SHA-256, and compares each stored digest; this is essential if corruption outside the controlled patch API may occur.

## Exact implementation and test commands

```
node --test apps/quine-bitwise3d/quine_bitwise3d.test.js apps/quine-bitwise3d/inward1000.test.js
node apps/quine-bitwise3d/inward1000.js
node apps/quine-bitwise3d/benchmark_inward1000.js
```

The reference is Node.js 18+ and requires no external dependencies. Default byte ceiling remains 1 MiB, and shadow candidate edits are bounded to 1,024 per cycle. Large *virtual* coordinates do not imply giant allocations, training, parallel CPU execution, GPU activity or changed physical clock speed.

## Observed local verification (Node.js 22, this session)

- Original `React Artifact(10).html`: **913,296 bytes**, SHA-256 `89b7a07fda22aca3f5b3d38f317530ad32dc2f600357b194b928d4f00c4df348`.
- Full-source 1000³ geometric archive: 913,296 nonzero leaf payloads; logical 10 inward folds; independently rebuilt root SHA-256 `0e8129e7de63603182daf3faca12828ac30071479f6ebe7137ca91d131666cef`; exact **0 differing bits**.
- Initial build approximately **3,914 ms**, deep independent audit approximately **4,129 ms** on the available container CPU; numbers are not portable.
- Single candidate one-bit difference updates **11** Merkle nodes, fails the fixed-point gate, and leaves the original source unmodified.
- The 1,000-cycle driver, without periodic full audits, accepted **999** unchanged source receipts and rolled back **1** mutation candidate.
- **23/23** combined baseline + inward tests passed locally.
- Benchmark of a different, precisely scoped task on a **16,384-byte source**: median full modified-tree rebuild **61.1 ms** vs median one-byte shadow-root update **0.070 ms**, approximately **878× faster** for *computing a modified root*. The incremental path does not recheck all bytes, so it is not a complete substitute for full verification.

## Non-claims

`1000³` is virtual *capacity*, not measured speedup, concurrent operations, CPU instructions per second, or 1,000,000,000x acceleration. A one-billion-fold speedup is unverified. Results depend on CPU, workload, input distribution, benchmark protocol and acceptance criteria; cached no-op iterations are not independent reconstruction proofs.

This reference does **not** imply trained ANN behavior, universal fixed-point convergence, autonomous self-improvement, neural learning, physical trillion-scale compute, consciousness or hardware speed beyond the measured task-specific benchmark.
