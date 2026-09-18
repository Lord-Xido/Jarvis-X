# ADR-019: Portable continuity envelope and self-carried learned substrate

**Status:** Accepted  
**Date:** 2026-09-13  
**Applies to:** Jarvis-X model packaging, persistent memory, adaptive runtimes, checkpoint transport, resumable execution, CodexVM control-plane work and future CPU/GPU/distributed backends  
**Extends:** ADR-016, ADR-017 and ADR-018

## Context

ADR-016 established one typed state and one candidate-first transaction boundary. ADR-017 established the canonical encode -> compact -> fixed-point -> decode -> evidence -> staged-adaptation law. ADR-018 added sparse conditional computation, adaptive inward depth and bounded speculative optimization.

Those decisions still leave one systems-level continuity problem: a runtime may preserve memory or journals while remaining dependent on an external host to reconstruct the learned transformation substrate, runtime policy and verification state that gave those records meaning.

Jarvis-X therefore treats continuity as a portable systems property rather than a property of one process, session or machine.

The research principle is:

```text
carry state
 -> carry memory
 -> carry runtime policy
 -> carry verification lineage
 -> carry the learned substrate when the checkpoint is owned/licensed for transport
 -> restore under compatibility and verification gates
 -> resume as a candidate
 -> commit OR reject
```

The phrase **carry your own weights** is an engineering shorthand for portable checkpoint ownership and reproducible restoration. It does not imply access to, extraction of, or transport of proprietary third-party model weights.

## Decision

Jarvis-X adopts a **Portable Continuity Envelope (PCE)** as the canonical transport and resume contract for a self-contained or partially self-contained runtime state.

The logical continuity state is

\[
\mathfrak C_t
=
(\Theta_t,\Omega_t,\Pi_t,V_t,A_t,S_t,M_t),
\]

where:

- \(\Theta_t\) is the learned model substrate: checkpoint weights, adapters or an immutable external model reference;
- \(\Omega_t\) is persistent memory and residual/temporal state;
- \(\Pi_t\) is bounded runtime policy;
- \(V_t\) is verification, provenance and audit lineage;
- \(A_t\) is architecture/schema/backend compatibility state;
- \(S_t\) is optional resumable execution state;
- \(M_t\) is the signed or hashed manifest binding the components and their versions.

No component becomes authoritative merely because it is present in an envelope. Restore remains subordinate to the ADR-016 authority law.

## 1. Learned-substrate modes

The continuity envelope SHALL distinguish at least three model-substrate modes.

### 1.1 Embedded owned checkpoint

A checkpoint controlled by the deployment may be stored directly in the envelope or in a content-addressed artifact store referenced by the envelope.

The manifest SHALL bind at least:

```text
model_format
model_family
architecture_id
checkpoint_digest
checkpoint_size
precision
adapter_stack
training_or_origin_record
license_or_usage_boundary
```

### 1.2 External immutable model reference

When the model weights are not distributable or not locally owned, the envelope SHALL store an immutable model identity/reference rather than pretending to contain the weights.

A restore is conformant only if the resolved model satisfies the declared compatibility contract.

### 1.3 Adapter-only continuity

A deployment MAY carry portable adapters, deltas or task-specific learned state while binding them to a fixed base-model digest or immutable model identifier.

A mismatched base model SHALL fail compatibility validation rather than silently loading the adapter.

## 2. Canonical continuity invariant

A runtime identity for engineering purposes is the bound tuple

\[
\mathfrak I_t
=
\operatorname{Bind}(\Theta_t,\Omega_t,\Pi_t,V_t,A_t),
\]

not any one component in isolation.

This is an operational continuity identity, not a claim about consciousness, personhood or subjective identity.

The invariant is:

\[
\operatorname{digest}(\mathfrak C_t)
=
H(M_t\|\Theta_t\|\Omega_t\|\Pi_t\|V_t\|A_t\|S_t)
\]

for a declared cryptographic hash construction or equivalent content-addressed binding.

A restore SHALL reject missing, mutated or incompatible authoritative components unless the manifest explicitly marks them as optional/reconstructable.

## 3. Serialize -> transport -> restore -> verify -> resume

The canonical lifecycle is:

```text
freeze authoritative source state
 -> flush journals and staged updates
 -> serialize typed namespaces
 -> hash / sign manifest
 -> export checkpoint or immutable checkpoint reference
 -> transport
 -> validate schema and component digests
 -> validate model/backend compatibility
 -> restore into isolated candidate namespace
 -> execute deterministic restore checks
 -> Verify + Pi_Lambda
 -> COMMIT restored state OR reject/rollback
 -> resume execution
```

Formally, serialization is

\[
B_t=\operatorname{Serialize}(\mathfrak C_t),
\]

transport is

\[
B_t \xrightarrow{\mathcal T} B'_t,
\]

and reconstruction is provisional:

\[
\mathfrak C_t^{\rm cand}
=
\operatorname{Deserialize}(B'_t).
\]

Authority is granted only after verification:

\[
\mathfrak C_t^{\rm auth}
=
\begin{cases}
\Pi_\Lambda(\mathfrak C_t^{\rm cand}), & V_t=1,\\
\mathfrak C_{\rm prior}, & V_t=0.
\end{cases}
\]

## 4. Resumable reasoning and runtime state

A PCE MAY contain an execution snapshot, but it SHALL distinguish durable state from ephemeral process state.

Durable state includes, when applicable:

- model/checkpoint identity;
- memory/journal state;
- typed geometry/profile state;
- runtime-policy state;
- verification and transaction lineage;
- architecture/configuration version;
- deterministic RNG state when required for replay;
- external artifact identifiers required to reconstruct a candidate.

Ephemeral state such as raw process handles, device pointers, sockets or accelerator allocations SHALL NOT be treated as portable identity. A backend adapter must reconstruct those resources on the destination host.

## 5. Weight adaptation remains candidate-first

Owning a checkpoint does not authorize unrestricted parameter mutation.

A proposed learned-state update is

\[
\Theta_{t+1}^{\rm cand}
=
\mathcal U_\Theta(\Theta_t,D_t,\Omega_t),
\]

but promotion requires an immutable evaluation anchor and the existing transaction boundary:

\[
\Theta_{t+1}
=
\begin{cases}
\Theta_{t+1}^{\rm cand}, & V_{\Theta}=1,\\
\Theta_t, & V_{\Theta}=0.
\end{cases}
\]

Verification SHOULD include declared task quality, regression checks, stability/resource bounds, checkpoint integrity and compatibility with the envelope schema.

The current Jarvis-X architecture therefore distinguishes:

```text
runtime recursion       -> may change S without changing Theta
memory consolidation    -> may change Omega without changing Theta
runtime optimization    -> may change Pi under bounded verification
model adaptation        -> may change Theta only through candidate/verify/commit
architecture evolution  -> may change A only through a versioned migration contract
```

## 6. Cross-host and cross-version migration

A conforming envelope SHALL declare enough version information to determine whether a destination can restore it.

Compatibility decisions SHALL be explicit:

```text
exact-compatible
migratable-with-versioned-transform
external-dependency-missing
model-incompatible
schema-incompatible
verification-failed
```

Migration transforms are themselves versioned code and SHALL preserve source lineage. A migration may not silently erase provenance in order to make a newer runtime accept an older envelope.

## 7. Security and integrity

A portable envelope is executable-adjacent state and SHALL be treated as untrusted input unless its provenance is already trusted.

Implementations SHOULD provide:

- cryptographic digests for all authoritative artifacts;
- optional manifest signatures;
- size/resource limits before materialization;
- safe parsing and schema validation;
- explicit deserialization allow-lists;
- no arbitrary code execution during checkpoint load;
- rollback on partial restore failure;
- audit records for migration and promotion decisions.

Model formats that permit arbitrary code execution during load SHALL NOT be accepted by the canonical portable profile without an isolated and explicitly trusted adapter.

## 8. Evidence boundary

A successful PCE round trip proves only the declared continuity properties.

It does not by itself prove:

- model intelligence or benchmark superiority;
- semantic equivalence across different base models;
- bit-identical floating-point execution across heterogeneous hardware;
- safe autonomous self-modification;
- legal portability of third-party checkpoints;
- continuity of consciousness or subjective identity.

The empirical claim to test is narrower:

> Given a declared portable profile, can Jarvis-X serialize the authoritative learned/runtime/memory state, restore it on a compatible host, verify integrity and behavior, and resume without untracked state loss?

## 9. Minimum implementation profile

The first executable PCE implementation SHOULD provide:

1. a versioned JSON manifest;
2. content digests for every authoritative component;
3. Hugging Face/safetensors-compatible owned-checkpoint support where available;
4. immutable external-model-reference mode;
5. Omega journal snapshot/export;
6. Pi_runtime and architecture/configuration serialization;
7. verification/audit lineage export;
8. isolated restore into a candidate namespace;
9. deterministic round-trip tests;
10. corruption, mismatch and missing-component rejection tests;
11. commit/rollback integration with ADR-016;
12. a machine-readable continuity receipt.

A reference receipt should minimally report:

```text
continuity_schema_version
source_transaction_id
envelope_digest
model_mode
model_digest_or_reference
memory_digest
runtime_policy_digest
architecture_digest
verification_lineage_digest
restore_backend
compatibility_status
restore_verification_status
commit_status
```

## Consequences

### Positive

- operational continuity becomes portable and testable;
- memory is bound to the model/runtime context that interprets it;
- owned checkpoints can move with the system rather than being reconstructed informally;
- external/proprietary models remain usable without falsely claiming ownership of their weights;
- model, memory, runtime and verification evolution become independently versionable;
- restore and migration inherit the same candidate/verify/commit discipline as ordinary execution.

### Costs

- envelope schemas and migrations become part of the compatibility surface;
- checkpoint transport can be large and expensive;
- security requirements increase because restore paths consume complex persisted state;
- deterministic replay may be limited by hardware, kernels and third-party dependencies;
- model licenses and provenance become first-class deployment constraints.

## Architectural invariant

The portable continuity law is:

\[
\boxed{
\text{carry }\Theta
+\text{ carry }\Omega
+\text{ carry }\Pi
+\text{ carry }V
+\text{ carry }A
\xrightarrow{\text{verify}}
\text{resumable Jarvis-X state}
}
\]

or, when the weights cannot legally or technically be carried:

\[
\boxed{
\text{carry an immutable model reference}
+\Omega+\Pi+V+A
\xrightarrow{\text{compatibility + verification}}
\text{resumable state}
}
\]

The host may change. The authoritative continuity contract must remain explicit, versioned, auditable and evidence-gated.
