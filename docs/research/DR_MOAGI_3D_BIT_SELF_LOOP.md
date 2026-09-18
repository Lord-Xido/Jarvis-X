# Dr Moagi 3D Bit-Level Inward Self-Loop

**Status:** Canonical research specialization  
**Parent architecture:** ADR-002, ADR-016, ADR-017  
**Attribution:** Dr Moagi family; see `docs/attribution/DR_MOAGI_EPHEMERAL_NOTION_FRAMEWORK.md`

## 1. Purpose

This specification defines the bit-level operational substrate for the Jarvis-X inward 3D codec/runtime. It translates the existing sparse 3D, residual-preserving, Omega/Theta/Pi architecture into finite binary state transitions without changing the repository's evidence boundary.

The central invariant is:

```text
read bits
 -> spatially contract
 -> preserve XOR residual shells
 -> refine the inner state
 -> reconstruct
 -> compare
 -> verify
 -> stage candidate control/model changes
 -> commit or rollback
 -> re-encode the verified machine state
 -> recur
```

The architecture is self-referential only in the bounded software sense: a finite representation of the runtime's state and policy becomes part of the next finite input state. It does not imply infinite physical computation.

## 2. Binary machine state

Let the complete finite digital state at cycle `t` be

```text
B_t = B_X || B_Z || B_R || B_Omega || B_Theta || B_Pi || B_C
B_t in {0,1}^N
```

with:

- `B_X` — observed/input state;
- `B_Z` — compressed/refined latent state;
- `B_R` — residual shells;
- `B_Omega` — persistent temporal/error memory;
- `B_Theta` — model/adaptation parameters;
- `B_Pi` — bounded runtime policy;
- `B_C` — CTR/admissibility evidence.

The irreducible transition is

```text
B_(t+1) = T_b^3D(B_t, I_t)
```

where `T_b^3D` is finite, versioned and resource-bounded.

## 3. Spatial addressing versus voxel state

The existing volumetric ROM uses a **60-bit spatial address**:

```text
address = x_20 || y_20 || z_20
```

which provides a logical `2^60` voxel address extent.

That address is deliberately separate from the **64-bit voxel-state word** defined here. A logical cell therefore has the conceptual form

```text
(address_60, state_64)
```

and sparse execution materializes only bounded active entries.

The address extent is virtual. It is not a claim that `2^60` state words are physically resident.

## 4. Canonical 64-bit voxel-state word

The canonical reference layout is

```text
63                                                        0
+--------+------------+----------+--------+------------+----------------+
| opcode | residual   | activate | memory | feature    | payload        |
|   8    |    12      |    8     |   8    |    12      |      16        |
+--------+------------+----------+--------+------------+----------------+
```

or

```text
W = O_8 || R_12 || A_8 || M_8 || F_12 || P_16
```

with:

- `O` — operation/state tag;
- `R` — local residual/error magnitude or residual metadata;
- `A` — activation/state value;
- `M` — Omega-local temporal memory;
- `F` — latent/feature value;
- `P` — payload/index/value.

Reference bit positions:

```text
payload    [15:0]
feature    [27:16]
memory     [35:28]
activation [43:36]
residual   [55:44]
opcode     [63:56]
```

The layout is a compact reference contract, not a claim that every higher-level tensor must be represented by one word.

## 5. 3D inward contraction

For each local `2 x 2 x 2` neighborhood,

```text
W_0, W_1, ..., W_7
```

the reference binary contraction is a per-bit majority operator:

```text
Z_j = 1  iff  sum_i bit_j(W_i) >= 4
```

for `j in [0,63]`.

Thus

```text
2^3 state words -> 1 latent state word
```

and recursive application gives the geometric hierarchy

```text
32^3 -> 16^3 -> 8^3 -> 4^3 -> 2^3 -> 1^3
```

matching the existing volumetric ROM pyramid.

The majority operator is intentionally simple and deterministic. Learned or quantized contraction kernels may replace it only behind the same tested interface and evidence boundary.

## 6. Residual-preserving shells

The reference bit residual for child `i` is

```text
R_i = W_i XOR Z
```

and exact reconstruction is

```text
W_i = Z XOR R_i.
```

Therefore the reference binary shell satisfies the invariant

```text
Decode(Encode(W_0...W_7)) == W_0...W_7
```

when all XOR residuals are retained.

This is a **lossless reference transform**. A production codec may quantize, prune or entropy-code residuals; any such path must declare its distortion semantics explicitly.

## 7. Inner refinement

For binary states `A` and `B`, define Hamming distance

```text
d_H(A,B) = popcount(A XOR B).
```

The reference fixed-point/refinement loop uses a bounded monotone correction:

```text
Z_(k+1) = Refine(Z_k, Z_target)
d_H(Z_(k+1), Z_target) <= d_H(Z_k, Z_target)
```

and stops when

```text
d_H(Z_(k+1), Z_k) <= tau_H
```

or the configured iteration budget is exhausted.

The reference implementation flips a bounded number of differing bits per step. It exists to make monotone self-correction testable; it is not presented as a learned intelligence mechanism.

## 8. 3D error field and compute allocation

For each active voxel,

```text
E_b(x,y,z) = W(x,y,z) XOR W_hat(x,y,z)
epsilon(x,y,z) = popcount(E_b(x,y,z)).
```

A runtime may use

```text
q(x,y,z) =
    q_0
  + alpha * epsilon(x,y,z)
  + beta  * (1 - confidence(x,y,z))
```

to allocate:

- recursion depth;
- precision;
- active-tile residency;
- verification work;
- memory bandwidth budget.

Canonical systems principle:

> **Compute density follows verified information difficulty.**

Stable regions may remain sparse or compressed. High-error regions may be refined more deeply, subject to `Pi_runtime` and `Pi_Lambda`.

## 9. Omega, Theta and Pi at bit level

The high-level states remain typed concepts even when serialized to bits.

```text
B_Omega <- MemoryUpdate(B_Omega, B_E)
B_Pi'   <- Schedule(B_Pi, B_E, B_C)
B_Theta'<- Adapt(B_Theta, B_Z, B_E)
```

A binary representation does not justify arbitrary bit flipping in live model state. Candidate updates remain transactional.

## 10. Verification-gated self-modification

For baseline configuration `M_t = (Theta_t, Pi_t)` and candidate `M'`:

```text
J_old = J(M_t; replay_suite)
J_new = J(M';  replay_suite)
```

Promotion requires:

```text
finite(candidate)
AND invariant_ok(candidate)
AND J_new <= J_old - epsilon
AND Pi_Lambda(candidate) == ACCEPT
```

Then

```text
M_(t+1) = M'   if verified
M_(t+1) = M_t  otherwise
```

At the binary selection boundary this is a multiplexer:

```text
B_(t+1) = MUX(V_t, B_candidate, B_baseline)
```

where `V_t` is the verified promotion decision.

This preserves the repository invariant:

```text
PROVISIONAL != AUTHORITATIVE
```

## 11. Complete inward self-loop

The full finite loop is

```text
B_t
 -> sparse 3D materialization
 -> 2x2x2 bit contraction
 -> residual shell capture
 -> latent nucleus
 -> bounded Hamming refinement
 -> outward reconstruction
 -> XOR error field
 -> CTR / Pi_Lambda verification
 -> Omega update
 -> Pi candidate
 -> Theta candidate
 -> isolated replay benchmark
 -> commit OR rollback
 -> pack B_(t+1)
 -> re-encode the verified state
 -> recur
```

Compactly:

```text
B_(t+1) = R_b^3D(B_t)
```

with authoritative promotion constrained by

```text
J_(t+1) <= J_t
```

for the declared replay objective of every accepted revision. This inequality is an acceptance rule, not a claim that all future tasks or external benchmarks improve monotonically.

## 12. Relationship to the existing runtime

This specialization maps directly onto the existing volumetric ROM stages:

```text
RESOLVE          -> 60-bit spatial address
FETCH_ALLOC      -> sparse active tile
ENCODE_PYRAMID   -> spatial hierarchy
CONTRACT         -> inward state contraction
FIXPOINT         -> bounded latent refinement
DECODE           -> outward reconstruction
COMPARE          -> XOR/Hamming or numeric residual
UPDATE_OMEGA     -> temporal state
UPDATE_THETA     -> staged candidate adaptation
OPTIMIZE_RUNTIME -> staged Pi candidate
STORE            -> persistence/writeback boundary
RECUR            -> next finite cycle
```

The current floating-point reference runtime remains valid. The bit-level layer is a lower-level serialization and reference-transform specialization, not a replacement for all numerical kernels.

## 13. Required invariants

1. `pack(unpack(W)) == W` for every 64-bit word.
2. `unpack(pack(fields)) == fields` for every valid field tuple.
3. `W_i == Z XOR (W_i XOR Z)` for every residual-preserving child.
4. Hamming refinement never increases declared target distance.
5. Candidate promotion is impossible when verification fails.
6. Candidate promotion is impossible without the configured objective improvement.
7. Spatial address width and voxel-state width remain distinct.
8. Iteration, memory and active-tile budgets are explicit.
9. Virtual geometry is never reported as physically resident memory.
10. Internal improvement is never reported as external SOTA evidence without matched benchmarking.

## 14. Reference implementation

The executable reference primitives live in:

```text
cpp_runtime/include/jarvisx/bit_self_loop3d.hpp
cpp_runtime/tests/bit_self_loop3d_tests.cpp
```

CTest exercises packing, contraction, exact residual reconstruction, monotone Hamming refinement and commit/rollback gating.

## 15. Evidence boundary

This specification defines operational mechanics. It does **not** establish:

- external state-of-the-art performance;
- autonomous unbounded self-improvement;
- physically infinite recursion;
- physical materialization of the virtual address space;
- learned intelligence from the majority/Hamming reference operators alone.

Those remain empirical claims requiring independent measurements.
