# ADR-029: Bounded Dr Moagi ANN Cognitive Matrix permeation

- **Status:** Proposed
- **Date:** 2026-09-21
- **Decision scope:** Dr Moagi ANN/VM reference runtime, sparse memory, fixed-point verification and 3D observation plane
- **Extends:** ADR-016, ADR-017, ADR-020, ADR-026, ADR-028

## Context

The Dr Moagi architecture now has separate repository implementations for sparse
spatial state, bytecode, cognitive/fixed-point surfaces, volumetric ANN references,
CTR/evidence boundaries and 3D control planes. The ANN Cognitive Matrix requires a
single bounded reference that demonstrates how those concepts compose without
turning logical scale or visualization telemetry into physical-performance claims.

The requested logical ANN population is \`1,200,000\` nodes and the byte-address
surface is a virtual 1 GiB ring. Neither value implies full physical residency.

## Decision

Add \`apps/dr-moagi-ann-cognitive-matrix/\` as a reference laboratory with:

1. a logical node count of \`1,200,000\`;
2. a configurable bounded resident active working set;
3. shared ANN-like update weights rather than a separate parameterized MLP per
   logical node;
4. sparse 64 KiB pages over a virtual 1 GiB address space;
5. deterministic 32-bit instruction words;
6. the opcode family \`LOAD_BITSTREAM\`, \`AUTO_ENCODE\`,
   \`DECODE_SPATIAL\`, \`TENSOR_MAP\`, \`INWARD_FOLD\`, \`EMIT_STREAM\`,
   and \`HALT_SYNC\`;
7. damped inward recurrence with an explicit measured fixed-point residual;
8. Omega residual memory;
9. CTR verification before candidate promotion;
10. atomic local commit or rollback of the candidate latent state;
11. measured macrocycle latency and resident-memory telemetry;
12. a dependency-free browser 3D projection that observes but does not enlarge
    the runtime's authority.

The reference state is

\`\`\`text
S_t = [X_t, Z_t, Xhat_t, E_t, Omega_t, Theta_t, R_t, B_t, M_t]
\`\`\`

and the local recurrence is

\`\`\`text
Z^(k+1) = (1-gamma) Z^k + gamma Phi_in(Z^k)
Omega_(t+1) = rho Omega_t + (1-rho) |E_t|
V_CTR = E_t^2 + residual_fixed_point^2 + constraint_penalty
\`\`\`

The candidate becomes authoritative only when the configured CTR threshold passes.

## GUI authority

ADR-026 remains authoritative for any operational GUI command surface. The browser
reference introduced here has browser-local authority only. It does not mutate
canonical Jarvis-X runtime state, invoke a shell, access a filesystem, control
devices, or bypass capability/verification gates.

## JIT / SIMD boundary

The browser reference does not claim arbitrary self-modifying native code or direct
AVX/AVX-512 emission. Browser JavaScript execution is controlled by the browser
engine. A future WebAssembly/native backend may lower verified hot paths into SIMD,
but the implementation and benchmark evidence must be reviewed separately.

## Evidence boundary

The repository must continue to distinguish:

\`\`\`text
logical capacity != physically materialized state != measured performance
\`\`\`

Therefore:

- \`1,200,000\` describes the logical ANN population;
- virtual 1 GiB describes address semantics;
- resident pages and active nodes describe physical JavaScript state;
- macrocycle latency is measured by the local runtime;
- FPS, Gbps, MIPS, SIMD speedup, compression quality and neural accuracy are not
  reported unless a named benchmark measures them.

## Consequences

Positive:

- ANN, VM, sparse-memory, fixed-point and CTR concepts have one executable bounded
  composition;
- scale claims remain falsifiable and auditable;
- the 3D surface shows authoritative local metrics rather than invented telemetry;
- rollback preserves the last committed latent state after verification failure.

Costs:

- the reference is intentionally smaller than the logical matrix;
- the shared-weight update is an ANN-like numerical reference, not evidence of
  trained model quality;
- production GPU/WebGPU/native execution requires separate adapters and benchmarks.
