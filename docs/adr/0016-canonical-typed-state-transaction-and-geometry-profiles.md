# ADR-016: Canonical typed state, transaction contract and geometry profiles

**Status:** Accepted  
**Date:** 2026-09-13  
**Applies to:** Jarvis-X core VM, Dr Moagi OS, DM-DD, DM-vOmegaXi+, sparse field runtimes, volumetric ROM ANN, multimodal loops, meta-optimizers and future CPU/GPU/distributed/hardware backends  
**Extends:** ADR-001, ADR-003, ADR-007, ADR-013, ADR-014 and ADR-015

## Context

Jarvis-X now contains several compatible but independently evolved execution surfaces:

- the deterministic 64-bit CodexVM;
- sparse 3D field runtimes;
- the Dr Moagi OS transaction pipeline;
- Deep Distiller adaptive state;
- DM-vOmegaXi+ fixed-point refinement;
- multimodal recursive loops;
- the `1000^3` inward-permeation profile;
- the `10^18` Inward-Core logical lattice;
- the `2^60` volumetric ROM ANN address space;
- virtual `10^24` temporal/state accounting;
- runtime and architecture meta-optimization.

These surfaces share the same systems thesis but currently use overlapping state definitions, symbol meanings and geometry scales. The next architectural priority is therefore structural closure rather than another larger virtual extent.

## Decision

Jarvis-X SHALL expose one canonical typed system contract and one transactional promotion law. Geometry size, representation and execution backend become profiles/adapters of that contract rather than competing definitions of the architecture.

The invariant is:

```text
one typed state
-> one candidate transaction
-> one verification contract
-> atomic commit OR rollback
-> many interchangeable geometry/backend profiles
```

## 1. Canonical typed system state

The system state is conceptually

```text
SystemState = (
    world,          # X: authoritative observed/domain state
    spatial,        # B/Gamma: sparse materialized geometry and topology
    latent,         # Z: inward/encoded working state
    prediction,     # X_hat: reconstructed or predicted state
    residual,       # E: X - X_hat and localized error evidence
    memory,         # Omega_mem: adaptive residual/temporal memory
    model,          # Theta_model: trainable model parameters
    runtime_policy, # Pi_runtime: bounded execution policy
    arch_policy,    # A_arch: bounded orchestration policy
    evidence,       # R: verification/reckoning state
    audit            # L: journal, lineage and integrity state
)
```

Symbols remain valid mathematical shorthand, but executable APIs SHOULD use explicit typed names. In particular:

- `Omega_mem` means adaptive memory; journal state is `audit`, not Omega.
- `Theta_model` means trainable/model parameters; runtime policy is separate.
- `Pi_runtime` means runtime policy; `Pi_Lambda` is reserved for admissibility projection.
- `A_arch` means architecture/orchestration policy.
- `Gamma_hierarchy` means spatial/hierarchical decomposition; damping remains a separate operator.

No backend may silently alias two authoritative state domains that have different commit or rollback semantics.

## 2. Authority namespaces

Authoritative mutation is partitioned into explicit domains:

```text
S_authority =
    S_vm
  x S_spatial
  x S_adaptive
  x S_control
  x S_audit
```

where:

- `S_vm`: bytecode registers, memory, instruction/lifecycle state;
- `S_spatial`: sparse domain state, topology and materialization metadata;
- `S_adaptive`: `Omega_mem`, `Theta_model` and bound model version;
- `S_control`: `Pi_runtime`, `A_arch` and constraint/version state;
- `S_audit`: journal head, transaction lineage, checkpoint identity.

A transaction that touches more than one domain MUST commit all affected domains atomically or restore all of them.

## 3. Canonical candidate transaction

For authoritative state `S_t` and external/input event `U_t`, a backend proposes

```text
S_candidate = K(S_t, U_t, C_t)
```

where `C_t` is the active bounded configuration.

The verification gate is

```text
V_t = Verify(S_t, S_candidate, U_t, receipts_t) in {0,1}
```

and the only authoritative promotion law is

```text
S_(t+1) = V_t * S_candidate + (1 - V_t) * S_t
```

interpreted structurally rather than as numeric vector addition.

The system-wide invariant remains:

```text
PROVISIONAL != AUTHORITATIVE
```

until every required gate for the transaction has passed.

## 4. Verification contract

Verification is separated into independent gates:

```text
V =
    V_finite
  AND V_resource
  AND V_type
  AND V_topology
  AND V_reconstruction
  AND V_fixed_point
  AND V_internal
  AND V_external
  AND V_policy
  AND V_transport
  AND V_integrity
```

Only gates applicable to a candidate must execute, but an implementation may not report an omitted capability as verified.

### Internal verification

Internal verification measures self-consistency, including:

- finite state;
- dimensional/type compatibility;
- reconstruction and cycle error;
- fixed-point residual;
- transform precision;
- topology/invariant residuals;
- resource ceilings;
- deterministic replay where required.

### External verification

External verification measures correspondence to independent evidence when a runtime makes claims about the world rather than only about self-reconstruction.

Internal convergence alone is insufficient:

```text
||Z - Phi(Z)|| -> 0
```

does not imply:

```text
d(X_world, X_hat) -> 0.
```

The Dr Moagi reality-coupled criterion is therefore retained as a distinct executable target:

```text
internal_coherence AND external_correspondence.
```

Runtimes without an external evidence source MUST label that gate `not-applicable` or `not-implemented`, not `passed`.

## 5. Geometry profiles

Logical scale is a configuration of the canonical contract, not proof of physical residency or throughput.

The following named profiles are recognized:

| Profile | Logical geometry | Meaning |
|---|---:|---|
| `BillionField1000` | `1000^3 = 10^9` coordinates | sparse billion-address field and global inward-permeation profile |
| `InwardCore1M` | `1_000_000^3 = 10^18` coordinates | ADR-013 canonical inward-core topology |
| `VolumetricROM20` | `(2^20)^3 = 2^60` coordinates | packed 20-bit-per-axis volumetric ROM ANN universe |
| `VirtualTemporal24` | `10^24` virtual states/indexes | virtual accounting/temporal indexing; not resident geometry or measured ops/s |

A geometry profile declares at minimum:

```text
profile_id
coordinate_domain
address_mapping
boundary_semantics
materialization_unit
active_support_limit
neighborhood/topology
contraction hierarchy
serialization order
resource accounting
```

Multiple profiles may implement the same higher-level operator contract.

## 6. Sparse execution invariant

For logical domain `V` and materialized active support `A_t`:

```text
A_t subseteq V
```

and finite implementations SHALL publish the measured/limited size of `A_t`.

Runtime work claims must be expressed in terms of resident/active work unless a dense algorithm is actually executed.

The preferred optimization law remains:

```text
recurse only where information, error or instability remains.
```

Active refinement SHOULD therefore be driven by explicit criteria such as residual magnitude, latent change, uncertainty, dependency demand or policy priority.

## 7. Canonical processing interface

Numerical/spatial backends SHOULD converge on the following narrow interface:

```text
ingest(state/event) -> TypedInput
encode(TypedInput) -> LatentReceipt
contract(latent, profile) -> ContractedReceipt
refine(contracted, memory, model, policy) -> FixedPointReceipt
decode(refined, residuals/support) -> ReconstructionReceipt
contrast(reference, reconstruction) -> EvidenceReceipt
adapt(evidence, staged_state) -> AdaptationReceipt
verify(candidate, receipts) -> VerificationReceipt
commit(candidate) | rollback()
```

A receipt records the backend/version, input identity, output identity, metrics, tolerances, resource cost and deterministic/replay information required by the enclosing transaction.

## 8. Control plane and data plane

The long-term architecture SHALL treat the deterministic VM/control ISA and volumetric numerical engines as complementary rather than competing authorities.

Preferred lowering:

```text
CodexVM / control plane
    -> LOAD_REGION
    -> ENCODE
    -> CONTRACT
    -> FIXPOINT
    -> DECODE
    -> COMPARE
    -> VERIFY
    -> COMMIT / ROLLBACK
    -> CHECKPOINT

Backend data plane
    -> Python reference
    -> C++ kernels
    -> GPU/CUDA
    -> distributed sparse workers
    -> FPGA/ASIC adapters
```

A backend may accelerate an operator but may not bypass its state, resource, evidence, policy or provenance contract.

## 9. Multiscale adaptation

The four adaptation scales remain distinct:

```text
t : authoritative state evolution
u : adaptive memory/model evolution
n : runtime-configuration evolution
k : architecture-policy evolution
```

with conceptual ordering

```text
state cycles >> model updates >> meta epochs >> architecture epochs.
```

Slower layers evaluate isolated candidates against immutable source snapshots and may not remove mandatory transaction/verification stages.

## 10. Convergence semantics

Jarvis-X distinguishes local inference convergence from global system equilibrium.

An inner fixed point may satisfy:

```text
Z* = Phi(Z*)
```

within a declared tolerance.

A continuously driven outer system need not approach a static state. Its stronger operational target is bounded tracking/dynamic equilibrium, in which stable regions consume little work and changed/high-error regions receive refinement.

No subsystem may infer global convergence solely from a local latent fixed point.

## 11. Provenance envelope

Cross-layer journals SHOULD converge on a shared transaction identity:

```text
TransactionEnvelope = (
    transaction_id,
    parent_id,
    epoch,
    input_provenance,
    state_hash_before,
    state_hash_candidate,
    model_hash,
    runtime_policy_hash,
    architecture_policy_hash,
    geometry_profile_id,
    receipts,
    decision,
    journal_parent
)
```

Separate VM, OS, meta and architecture journals may remain physically independent, but they SHOULD reference the same transaction lineage when they describe one causal promotion.

## 12. Migration rules

This ADR does not require immediate replacement of existing implementations.

Instead:

1. existing engines become adapters/profiles of this contract;
2. new engines MUST declare their canonical state domains and geometry profile;
3. duplicated meanings of Omega/Theta/Pi/Gamma should be removed from implementation-facing APIs as touched;
4. cross-domain mutations must become rollback-complete;
5. consolidated empirical validation should progressively consume standard receipts;
6. geometry/profile conversion must be explicit rather than inferred from naming.

## 13. Consequences

### Positive

- eliminates competition between `1000^3`, `10^18`, `2^60` and `10^24` abstractions;
- creates a stable target for Python, C++, GPU, distributed and hardware implementations;
- preserves the strongest Jarvis-X invariant: candidate-first promotion;
- makes external/reality verification independent from self-consistency;
- reduces symbol/state ambiguity;
- enables cross-backend benchmarking and replay using shared receipts;
- clarifies the future role of CodexVM as control plane and volumetric engines as bounded data planes.

### Costs

- requires adapter work across existing research runtimes;
- introduces explicit state/version schemas where some modules currently use local dataclasses;
- requires migration of metrics into shared receipt forms;
- exposes capability gaps that were previously hidden by compatible prose-level abstractions.

## 14. Acceptance and future implementation

ADR-016 is an architectural decision, not evidence that all adapters already conform.

Implementation is complete only when the repository can demonstrate, through tests and machine-readable evidence, that representative VM, sparse Python and C++ volumetric paths:

1. declare a geometry profile;
2. produce typed stage receipts;
3. stage rather than directly promote adaptive changes;
4. use the common verification decision model;
5. rollback all affected authoritative domains on rejection;
6. bind the resulting commit to common transaction lineage.

Until then, existing subsystem-specific contracts remain valid where they are stricter, and ADR-016 governs convergence of future integration work.
