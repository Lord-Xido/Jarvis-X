# ADR-039: Exact Arithmetic Verification for Admitted 3D Operations

- **Status:** Proposed / executable reference
- **Date:** 2026-10-27
- **Extends:** ADR-016, ADR-017, ADR-034, ADR-038
- **Implementation:** `src/jarvisx/gmral_verified_3d.py`
- **Tests:** `tests/test_gmral_verified_3d.py`

## Governing policy

Every admitted GMRAL operation is an explicit partial mathematical function
(f : D_f \subseteq \mathbb F_2^{L^3} \times A_f \to \mathbb F_2^{L^3} \times R_f)
where (L \in \{2,4,8,16\}) and (A_f) is the operation-specific typed argument domain.

An operation produces a **candidate** and is accepted only if an independent
arithmetic reference evaluator agrees *at every voxel*, geometric addresses
round-trip, and all non-state outputs (carry, period, residue) match exactly.

```text
typed operation
  -> validate input domains
  -> propose candidate with optimized arithmetic kernel
  -> calculate independent reference via voxel equation(s)
  -> validate each source/destination 3D coordinate
  -> verify output + carry/period/residue
  -> issue reproducible receipt + per-voxel witness
  -> return immutable committed state, or raise before acceptance
```

The initial implementation **does not retrofit or guard every existing Jarvis-X
subsystem**. The contract is binding only within this new, explicitly admitted
entrypoint. Migration of additional workloads requires their own mathematical
semantics, reference evaluators, and integration tests.

## Exact 3D address contract

Let `E_L(x,y,z)` interleave coordinate bits in the canonical X:Y:Z
order and `D_L(i)` perform the inverse deinterleaving. Require:

```text
D_L(E_L(x,y,z)) = (x,y,z)
E_L(D_L(i)) = i
0 <= x,y,z < L
0 <= i < L**3
```

Each accepted transition includes exactly (L^3) voxel witnesses.
A voxel witness exposes its 3D output coordinate, input bit(s),
arithmetic carry where applicable, and source 3D coordinate where
a permutation or cycle transformation is involved.

## Admitted arithmetic operations

| Operation | Arithmetic reference |
| --- | --- |
| XOR | `(a_i+b_i) mod 2` |
| AND | `a_i*b_i` |
| OR | `min(1,a_i+b_i)` |
| NOT | `1-a_i` |
| ADD | `a_i+b_i+c_i=s_i+2*c_(i+1)`, cross-checked against integer addition |
| AFFINE | `y_d=x_(P^-1(d)) XOR M_d` |
| EVOLVE | `F^N(x)=F^(N mod T)(x)`, verified with an independent permutation-cycle parity formula |

All operations are bounded by the declared lattice side and typed arguments.
The integer arithmetic implementation and 3D evidence model are exact for
their specified bit widths; overflow from ADD is reported as a separate carry bit.

### Exponential reduction witness

For an affine bit transformation (F(X)=P(X)\oplus M),
let (r=\operatorname{ord}(P)). Then (F^r) is XOR with a fixed mask.
The exact universal period is either (T=r) or (T=2r).

The optimized candidate is computed by the existing GMRAL-Ω operator
exponentiation. A **different** verifier independently decomposes the
geometric permutation into cycles and computes the cumulative XOR parity
using doubled prefix tables; it does not execute the giant iteration count.

For the 4×4×4 example, `T=24`, so:

```text
(1_000_000 ** 1_000_000) mod 24 == 16
F ** (1_000_000 ** 1_000_000) == F ** 16
```

The expression represents a mathematically exact final-state shortcut,
not a physical processor frequency or a universal time complexity bound.

## Transaction and evidence requirements

- Unknown operation types, invalid coordinates, mismatched lattice shapes,
  malformed typed arguments, wrong output bits, wrong carry and false period
  or exponent residues must raise an exception before receipt issuance.
- State objects and verification receipts are immutable.
- An atomic chain returns the final state and receipts only after all
  requested operations pass; no intermediate state is externally committed.
- The SHA-256 witness digest is deterministic **integrity metadata**, not
  an independent mathematical proof or a security authentication mechanism.
- The verifier itself is ordinary software. A passing runtime check, even
  backed by an independent oracle, is not a proof about hardware faults,
  real-world correctness, or arbitrary unimplemented operations.

## Reproduction

```bash
python -m jarvisx.gmral_verified_3d
pytest -q -o addopts='' tests/test_gmral_verified_3d.py
```

Example: on a `4×4×4` lattice, the verified ADD operation
`5+3=8` returns 64 voxel equations and a reproducible receipt.

## Follow-on acceptance boundary

Integration into the *canonical* Jarvis-X runtime must remain a separate
step: enumerate all state-mutating paths, define mathematical specifications,
add independent oracles and property-based tests for each path, enforce
the verification gate in the runtime, then require successful CI evidence
and explicit review before changing production authority.
