# Dr Moagi React Artifact (10): Bitwise Quine and Reversible 3D Bridge

**Status:** Proposed bounded research reference, 2026-10-30.  
**Provenance:** Matladi Maxwell Moagi (Lord-Xido), per ADR-015.  
**Architectural constraints:** ADR-016, ADR-017, ADR-031 and ADR-034.  
**Implementation:** `apps/quine-bitwise3d/` (Node.js >=18; no external dependencies).

## Supplied source baseline

Uploaded `React Artifact(10).html`: **913296 bytes**, SHA-256:

`89b7a07fda22aca3f5b3d38f317530ad32dc2f600357b194b928d4f00c4df348`

The React/Three.js demonstration displays a `Float32Array(512)` encoder adding `charCodeAt(i)*0.0007` into index `i % 512`, but its decoder is the closure `const decode = (latent) => src;`. Two distinct 513-character strings have **identical** Float32 latent bytes; therefore this displayed transform is not an arbitrary-input lossless decoder. Displayed fidelity is a simulated metric and visual helix generations do not establish evaluated self-editing programs. `SELF_AWARE` is a UI label, not a consciousness claim. The embedded **161-byte** source-template quine does reproduce its own source. The entire HTML is **not** established to be a quine.

The original HTML is **not committed** to the public repository: only its checksum, measured size, and audit method are published. Re-audit offline with:

```sh
node apps/quine-bitwise3d/audit_react_artifact10.js "/path/to/React Artifact(10).html"
```

The auditing script reads bytes, checks known source substrings and generates a concrete latent collision; it does not execute the uploaded module and is version-specific.

## Exact reversible geometric mapping

Each byte address `i` maps to a unique voxel `(x,y,z)` under bounded width `W` and height `H`:

\[
x=i\bmod W,\quad y=\lfloor i/W\rfloor\bmod H,\quad z=\lfloor i/(WH)\rfloor,
\quad i=x+W(y+Hz).
\]

Nonzero source bytes are stored as `[x,y,z,byte]`; source length plus implicitly zero-filled cells restore zeros. The archive includes schema, shape, length and source SHA-256. Decoder rejects invalid, repeated, out-of-range or corrupted coordinates and digest mismatches. SHA-256 checks integrity, not authenticity against an attacker who can replace a whole archive. The default input ceiling is 1 MiB; the sparse voxel table is a *logical* 3D arrangement, **not** claimed data compression, physical memory, or neural representation.

For equal-sized byte arrays, the exact mismatch is `sum(popcount(A[i] XOR B[i]))`. Unequal sizes fail. ADR-034 requires **zero** mismatching bits. The candidate-first loop either verifies source-identical reconstruction and commits, or rejects changes and retains the original byte state. It supports no self-authorizing source mutation, ANN training, or performance speedup. A separate bundled classical quine passes 1,000 repetitions; Node `vm` runs **only** that trusted constant and must not be used as a sandbox for untrusted code.

## Local evidence and regression

The uploaded **913296-byte** HTML completed a real 3D encode/decode roundtrip with **zero bit differences** and matching SHA-256 (913296 nonzero stored voxel records). This proves representation recovery, not autonomous source generation. The test suite covers reversibility, exact XOR residuals, corruption rejection, malformed addresses, commit/rollback, bounded iteration, 1,000 classical quine reproductions, and the Float32 collision.

```sh
node --test apps/quine-bitwise3d/quine_bitwise3d.test.js
node apps/quine-bitwise3d/quine_bitwise3d.js
```

CI runs path-scoped tests under Node 18 and 22. The bridge supplements open PRs #434 and #435. Review and CI are required before merging; no existing authoritative Jarvis-X runtime, fixed-point equation, or production control plane is altered.
