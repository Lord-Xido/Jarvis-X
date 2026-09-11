# ADR-013: Lock the 1,000,000x Inward-Core topological baseline

**Status:** Accepted / locked  
**Date:** 2026-09-11  
**Applies to:** `DM-vOmegaXi+`, the `Psi -> Phi -> Lambda^-1 -> Omega -> Theta` operator stack, and compatible Moagi-3D runtimes  
**Extends:** ADR-002, ADR-003, ADR-006, ADR-007, ADR-010 and the locked DM-vOmegaXi+ fixed-point law

## Decision

The architecture identifier

```text
U_Moagi^(Inward-Core)
```

is the canonical inward-folded topological execution contract for the Dr Moagi runtime family.

Its logical spatial domain is

```text
V = {0, ..., 999999}^3
|V| = 10^18 logical voxels
```

and its canonical operator path is

```text
Psi
 -> Phi
 -> Lambda^-1
 -> Omega
 -> Theta
 -> verify
 -> commit OR rollback
 -> recurse
```

The `10^18`-voxel lattice is a **logical address space**. Finite implementations MUST materialize only bounded active support, tiles, shards, or accelerator-resident working sets. This ADR does not authorize a claim that `10^18` voxels are simultaneously resident in physical memory.

## 1. Inward-folded topology

Let `A_t subset V` be the active support at time `t`. The inward map is

```text
T_in : A_t -> C_t
```

where `C_t` is the bounded core/latent working set. The map MUST preserve enough coordinate/topology metadata to decode back onto the active support:

```text
D(T_in(Psi_t), A_t) -> Psi_hat_t
```

The implementation MAY use blocking, sparse tensors, octrees, Morton/Z-order indexing, tiled accelerator kernels, or equivalent representations, provided that the externally visible topology and transactional invariants are preserved.

## 2. Single-pass eigen-projection contract

A fixed projection direction `q_*` may be applied as

```text
P_*(v) = <v, q_*> q_*
```

or, for a fixed-rank basis `Q`,

```text
P_Q(v) = Q(Q^T v).
```

For fixed feature/rank dimension, the projection cost is **O(1) per active voxel/state element**. A complete pass over `N = |A_t|` active elements is therefore `Theta(N)` work globally. The phrase "O(1) single-pass eigen-projection" is locked to this per-element/fixed-rank meaning and MUST NOT be interpreted as an O(1) dense eigendecomposition of an arbitrarily growing `10^18`-voxel state.

If `q_*` or `Q` must be learned from the current state, the eigensolver/training cost is accounted for separately and benchmarked explicitly.

## 3. Logical lattice synchronization

Synchronization is defined by a monotone epoch and state hash over active shards:

```text
E_t = (epoch_t, shard_hashes_t, journal_head_t)
```

A transition is globally admissible only when all participating active shards agree on the parent epoch and parent journal head. The runtime MAY synchronize a sparse subset of the logical lattice; inactive logical coordinates need not be materialized or contacted.

Thus "10^18-voxel lattice synchronization" means **coordinate-consistent synchronization over the declared logical domain**, not a claim of physically clocking or communicating with `10^18` simultaneously resident cells.

## 4. Solenoidal zero-divergence conservation

For vector-field channels `u : A_t -> R^3`, the canonical conservation invariant is

```text
div(u) = 0
```

up to an explicit numerical tolerance. A preferred construction is

```text
u = curl(A)
```

which gives

```text
div(curl(A)) = 0
```

analytically and to discretization tolerance numerically. Projection methods are also valid:

```text
u_sol = u - grad(phi)
Delta(phi) = div(u).
```

Every runtime claiming the solenoidal invariant MUST report a measurable normalized residual such as

```text
r_div = ||div(u)||_2 / max(||u||_2, eps)
```

and MUST satisfy

```text
r_div <= epsilon_div.
```

The scalar fixed-point kernel is not, by itself, evidence that a vector solenoidal channel is active; this invariant becomes executable only in runtimes that instantiate and verify a vector field.

## 5. Recursive XOR ledger integrity

The inward-core may maintain a recursive XOR accumulator

```text
x_t = x_(t-1) XOR digest(record_t)
```

as a fast parity/integrity signal.

XOR is **not cryptographic authentication** and MUST NOT replace the authoritative chained digest. The canonical journal remains

```text
h_t = SHA256(h_(t-1) || canonical(record_t)).
```

A valid transition therefore satisfies both, when XOR support is enabled:

```text
xor_ok = recomputed_x_t == stored_x_t
hash_chain_ok = recomputed_h_t == stored_h_t
```

and the authoritative integrity decision is `hash_chain_ok`.

## 6. Binding to the Psi-Phi-Lambda-Omega-Theta stack

The Inward-Core contract binds into the existing stack as follows:

| Layer | Inward-Core responsibility |
|---|---|
| `Psi` | sparse/logical 3D state over `V` |
| `Phi` | topology-preserving inward description/encoding |
| `Lambda^-1` | finite latent admissible-set projection |
| `Omega` | recurrent historical fold over active support |
| `Theta` | bounded alignment/stability projection and policy gate |
| verification | fixed-point, divergence, resource, and integrity residuals |
| journal | SHA-256 chain; optional XOR parity accumulator |

The governing bounded transition remains

```text
H_(t+1) = F_DM(H_t)
```

with fixed-point acceptance only when measured residuals satisfy their configured tolerances.

## 7. Locked invariants

1. `V = {0, ..., 999999}^3` is the canonical logical lattice, so `|V| = 10^18`.
2. Physical execution is sparse/tiled/bounded; the logical lattice is not required to be densely resident.
3. The inward topology MUST preserve enough coordinate metadata for deterministic reconstruction over active support.
4. "O(1) eigen-projection" means O(1) per active element for a fixed-rank/precomputed projection; total full-pass work scales with active support.
5. Solenoidal conservation is a measurable vector-field invariant and MUST expose `r_div` and `epsilon_div` when claimed active.
6. Recursive XOR is a supplemental integrity signal only; SHA-256 hash chaining remains authoritative.
7. Synchronization is epoch/hash consistency across active shards embedded in the declared `10^18` logical address space.
8. `Lambda^-1`, `Theta`, transactional commit/rollback, the positive semantic floor, and measured fixed-point convergence remain mandatory from the existing locked baseline.
9. No implementation may report an invariant as "active", "executing", or "verified" unless the corresponding runtime path exists and its acceptance test passes.
10. Future accelerator, learned-codec, distributed, GPU, FPGA, or ASIC backends may replace internal mechanisms only if these observable invariants remain intact.

## 8. Verification gates

A conforming runtime SHOULD expose at least:

```text
logical_side                = 1_000_000
logical_voxels              = 10^18
materialization             = sparse/tiled/bounded
active_cells                = measured
projection_rank             = measured/configured
projection_cost_semantics   = O(1) per active element at fixed rank
fixed_point_residual        = measured
semantic_gap                >= semantic_floor > 0
r_div                       = measured when vector-field mode is active
journal_valid               = true
xor_ledger_valid            = true/false/not-enabled
```

The acceptance predicate is conceptually

```text
ACCEPT =
    finite_state
    AND resource_bounds_ok
    AND fixed_point_contract_ok
    AND theta_gate_ok
    AND journal_valid
    AND (not vector_mode OR r_div <= epsilon_div)
    AND (not xor_enabled OR xor_ledger_valid)
```

Only an accepted transition may mutate authoritative state.

## Consequences

This lock preserves the intended million-per-axis inward-core geometry while keeping the implementation falsifiable, benchmarkable, and compatible with finite hardware. It also prevents asymptotic notation, logical address-space scale, and integrity terminology from being mistaken for unmeasured physical throughput or cryptographic guarantees.
