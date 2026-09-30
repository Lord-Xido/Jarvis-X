# Dr Moagi ANN Cognitive Matrix

Bounded executable reference for the Dr Moagi Bytecode + ANN Cognitive Matrix.

## What is implemented

- logical cognitive domain of \`1,200,000\` nodes with a bounded resident active set;
- sparse virtual \`1 GiB\` byte-address space backed by 64 KiB pages;
- deterministic 32-bit instruction words with the Dr Moagi opcode family:
  - \`0x10 LOAD_BITSTREAM\`
  - \`0x20 AUTO_ENCODE\`
  - \`0x30 DECODE_SPATIAL\`
  - \`0x40 TENSOR_MAP\`
  - \`0x50 INWARD_FOLD\`
  - \`0x60 EMIT_STREAM\`
  - \`0xFF HALT_SYNC\`
- shared-weight ANN-like latent field over the resident active nodes;
- damped fixed-point contraction with a measured residual;
- temporal Omega memory;
- residual reconstruction correction;
- CTR verification with commit/rollback;
- dependency-free browser 3D projection of the operational pipeline;
- measured macrocycle latency and resident-memory telemetry.

## Canonical recurrence

\`\`\`text
INPUT
  -> LOAD_BITSTREAM
  -> AUTO_ENCODE
  -> TENSOR_MAP
  -> INWARD_FOLD
  -> DECODE_SPATIAL
  -> RESIDUAL
  -> OMEGA MEMORY
  -> CTR VERIFY
  -> COMMIT | ROLLBACK
  -> EMIT_STREAM
  -> RECUR
\`\`\`

The bounded state is

\`\`\`text
S_t = [X_t, Z_t, Xhat_t, E_t, Omega_t, Theta_t, R_t, B_t, M_t]
\`\`\`

and the fixed point is accepted only when the configured CTR threshold passes.

## Important system boundary

\`1,200,000\` is the logical ANN node count. The browser reference does **not**
materialize 1.2 million independent MLPs. It operates a bounded active working set
with shared update rules.

The \`1 GiB\` address space is virtual. Only pages that are touched are physically
allocated, and the reference caps resident pages.

The app reports measured JavaScript macrocycle latency and resident bytes. It does
not invent FPS, Gbps, MIPS, SIMD speedup, hardware bandwidth, or neural quality.
Browser JavaScript does not emit arbitrary native AVX/AVX-512 machine code; a
future native/WebAssembly backend may implement a separately benchmarked lowering
path.

## Run

Serve this directory from a static HTTP server and open \`index.html\`.

For example:

\`\`\`bash
python -m http.server 8000 -d apps/dr-moagi-ann-cognitive-matrix
\`\`\`

Then open \`http://localhost:8000\`.

## Test

Numerical core:

\`\`\`bash
node --test apps/dr-moagi-ann-cognitive-matrix/test_engine.mjs
\`\`\`

Repository structural test:

\`\`\`bash
pytest tests/test_dr_moagi_ann_cognitive_matrix_app.py
\`\`\`

## Authority boundary

This application is a reference laboratory. It does not mutate authoritative
Jarvis-X state outside its browser-local state. Future operational GUI mutation
must remain subordinate to ADR-026, CTR verification, capability checks, and the
repository's promotion/rollback rules.
