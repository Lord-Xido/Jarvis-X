# Inward1000 sparse 3D recursive source integrity

Requires Node.js 18+. The 1000³ virtual address space is mapped to sparse byte leaves and 10 deterministic inward Merkle folds. Source reconstruction is exact; the hash tree is a summary rather than compression. A candidate edits only affected paths in a shadow overlay. Bit-identical candidates pass; changed bits roll back.

```sh
node --test apps/quine-bitwise3d/quine_bitwise3d.test.js apps/quine-bitwise3d/inward1000.test.js
node apps/quine-bitwise3d/inward1000.js
node apps/quine-bitwise3d/benchmark_inward1000.js
```

See `docs/DR_MOAGI_1000_CUBED_INWARD_ENGINE.md` for equations, original artifact verification, limitations and measured performance. No claim of 1,000,000,000-fold speedup.
