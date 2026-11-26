# 3D Volumetric ISA64 — experimental reference profile

Source: 3D Volumetric Bytecode ISA Specification, supplied 2026-11-26, pages 1–3. This implementation is additive under ADR-027 and ADR-031 and does not replace JX3DVM1 or grant commit authority.

## Execution
Run `node experimental/volumetric_isa64/isa64.mjs` with a BigInt-capable Node.js runtime. The standalone script runs deterministic tests and prints a JSON receipt.

The logical domain is 10000^3. Physical execution is an origin brick with side 1..32, never the full trillion cells. Each sweep uses read-only previous state and a private candidate map; the caller receives the map only after coverage and exact token/residual checks. This is a local check, not the canonical CTR transaction system.

## Binary audit
The fixed layout is OP8 MODE4 RD8 RS1_8 RS2_8 IMM28, MSB first. Corrected PDF listing words:
| PC | word |
|---|---|
| 00 | 0104000100000000 |
| 02 | 110005000000000B |
| 03 | 0203000100002710 |
| 05 | 2016280100000001 |
| 06 | 2016380200000002 |
| 09 | 301644060000003F |
| 11 | 2103140600000001 |

At PC 02 the signed PC-relative displacement is +11, not +10; it targets PC 13 rather than the stale-state store at PC 12. MORT3D and LINIDX use implicit Z0 in this profile. Q_FMA is scaled multiplication as in the listing, not fused multiply-add.

## Explicit design choices (not specified by PDF)
- X-fast traversal; global stride 10000; bounded brick-periodic neighbourhood (not global torus).
- Morton interleave X/Y/Z in low-to-high bit order; local octree prefixes at min(3, ceil(log2(side))). A nonempty prefix evaluates every cell in that prefix. The current loop visits even pruned cells.
- Nonnegative Q16.16 values and factors in [0,1]; arbitrary-width BigInt intermediates, floor rounding, no machine overflow in supported domain.
- Six-face arithmetic mean from previous state, missing cells zero. Axis factor product times the mean. Codebook [0, 0.25, 0.5, 0.75, 1], nearest value, lowest index on ties. Exact signed residual stored separately.
- Only the opcodes in the fixed 15-word kernel are dispatched. UNMORT is a helper; MOD_WRP and general assembly/program loading are not implemented. No trainable ANN, no hardware acceleration, no full-domain execution.

## Test receipt
Executed in a JavaScript runtime before commit: 15 instruction word roundtrips; 1000 Morton and 1000 residual roundtrips; 1000 primary sweeps and 1000 matching replays, on a 2^3 fixture; 113000 primary instruction dispatches. First cell Q16 raw 9216 (0.140625). Sparse path 43 instructions, empty path 33. The fixture reaches zero; general stability or useful learning is not established. GitHub Actions and Node CLI execution are separate verification steps.

Instruction count = 4N + 10A + 1 for N visited cells, A evaluated cells. No measured throughput claim.
