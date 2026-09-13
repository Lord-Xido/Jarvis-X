# Dr Moagi Four-Scale System Auto-Evolution

## Status

The Jarvis-X Dr Moagi stack now has four explicit adaptation time scales:

```text
t  authoritative sparse/world state X
u  adaptive model memory/parameters (Omega_mem, Theta_model)
n  runtime configuration / Pi_runtime
k  architecture orchestration policy A_arch
```

The architecture follows ADR-016 typed-state closure and ADR-017's canonical operational auto-encoding/decoding equation. Geometry profiles and execution backends are adapters of one state/transaction contract rather than competing definitions of the system.

## Canonical operational auto-encoding/decoding law

Candidate generation is governed by

\[
S^{\rm cand}_{t+1}
=
\left[
\mathcal U_{\Omega,\Theta,\Pi_{\rm run}}
\circ
\mathcal R_{\rm CTR}
\circ
\mathcal D_{\mathcal R}
\circ
\operatorname{Fix}_{F_\Theta}
\circ
\Phi_{\rm fusion}
\circ
\mathcal C_{\exp}
\circ
\mathcal E
\right](S_t,U_{t+1}).
\]

The operational sequence is

```text
typed input/event
 -> encode
 -> sparse exponential/multiresolution compaction
 -> preserve residual hierarchy
 -> shared/modal fusion
 -> bounded fixed-point refinement
 -> selective residual-aware decode
 -> compare / CTR evidence
 -> stage Omega_mem
 -> stage Theta_model
 -> stage Pi_runtime
 -> build candidate state
 -> Pi_Lambda / verification
 -> atomic commit OR rollback
 -> audit / recur
```

The candidate is promoted only under the ADR-016 transaction law

\[
S_{t+1}=V_t\,\Pi_\Lambda(S^{\rm cand}_{t+1})+(1-V_t)S_t,
\]

interpreted structurally. `Pi_runtime` and `Pi_Lambda` are intentionally distinct: the first optimizes bounded execution policy; the second is the admissibility/projection boundary for authoritative promotion.

See `docs/DR_MOAGI_OPERATIONAL_AUTOENCODING_EQUATION.md` and ADR-017 for the complete operator, residual hierarchy, fixed-point, evidence, adaptation and capability contracts.

The evolution hierarchy is:

```text
external sparse/world state
      |
      v
+----------------------------------+
| Inner transactional data plane   |
| encode -> compact -> residuals   |
| -> fusion -> fixed point         |
| -> decode -> CTR/evidence        |
| -> stage Omega/Theta/Pi_runtime  |
| -> Pi_Lambda -> verify -> commit |
+----------------+-----------------+
                 |
                 v
+----------------------------------+
| Runtime meta optimizer           |
| compression / adaptation /       |
| spatial / fixed-point policy     |
+----------------+-----------------+
                 |
                 v
+----------------------------------+
| Architecture optimizer A_arch    |
| cadence / search budget /        |
| promotion resilience             |
+----------------+-----------------+
                 |
                 v
          next autonomic epoch
```

The architecture layer does **not** rewrite source code or remove required safety or verification stages. It searches bounded orchestration policies around the incumbent system.

## Constitutional execution topology

The following semantic stages are immutable in the architecture controller even when a backend fuses them operationally:

```text
typed_ingest
 -> encode
 -> inward_compact
 -> residual_preserve
 -> fixed_point
 -> decode
 -> contrast / evidence
 -> stage_adaptation
 -> pi_lambda
 -> dmos2_verify
 -> atomic_commit_or_rollback
```

Existing lower-level paths such as bitplane conversion, AutoExec and Deep Distiller remain valid backend stages beneath this semantic contract.

Architecture evolution changes how the nested loops are scheduled and evaluated, not whether transactional validation exists.

## Four recurrences

### 1. State recurrence

Conceptually:

```text
S_candidate = K(S_t, U_t, Pi_runtime,t)
S_(t+1) = Verify(S_t, S_candidate) ? Pi_Lambda(S_candidate) : S_t
```

A rejected cycle leaves all touched authoritative state domains unchanged.

### 2. Model recurrence

```text
E_t = X_t - Xhat_t
Omega_candidate = U_Omega(Omega_t, E_t, residual_hierarchy, Z*)
Theta_candidate = Theta_t - eta * grad_Theta L_t
```

These staged values are committed only with the state transaction. Their exact update law is backend-specific but must produce the ADR-016 receipts needed by the enclosing verification gate.

### 3. Runtime-configuration recurrence

The historical configuration state `C_n` maps to the canonical `Pi_runtime` namespace:

```text
Pi_runtime,n+1_candidate
    = argmin_{Pi in N3(Pi_runtime,n)} J_runtime(Pi)
```

The 3D runtime lattice axes may include compression geometry, adaptive dynamics, and spatial/fixed-point dynamics. A candidate runtime policy is still provisional until the corresponding meta-level gate accepts it.

### 4. Architecture-policy recurrence

```text
A_arch,k+1_candidate
    = argmin_{A in N3(A_arch,k)} J_arch(A)
```

The architecture axes are:

- X: state-to-meta cadence;
- Y: meta-search budget/depth;
- Z: promotion resilience.

Every architecture candidate is executed in isolated kernels over the same bounded source state. The production state is not used as scratch space.

## Autonomic scheduler

`SelfEvolving3DArchitecture.run_autonomic(cycles)` closes all four loops:

1. execute authoritative state transactions;
2. stage and verify model-memory updates with the state cycle;
3. after `state_cycles_per_meta`, run one inward runtime-policy epoch;
4. after `meta_epochs_per_architecture_review`, run one architecture epoch;
5. promote only candidates that pass their corresponding gates;
6. stop immediately on a rejected authoritative state transaction where the active contract requires it.

This creates a hierarchy of increasingly slower adaptation:

```text
state cycles >> model updates >> runtime meta epochs >> architecture epochs
```

## Architecture objective

Architecture evaluation measures the complete nested runtime using applicable receipts such as:

- reconstruction MSE;
- DM-DD / residual RMS;
- fixed-point residual;
- residual-hierarchy or active-support cost;
- exact transport bytes per source cell;
- active/latent compute proxy;
- resident memory;
- phase velocity where defined;
- internal meta-optimization improvement;
- evaluation cost;
- rejection penalties.

The architecture gate additionally limits regression of reconstruction and residual quality.

An internal architecture improvement is **not** an external SOTA claim. Status therefore keeps `external_sota_verified=false` until matched external benchmarks exist.

## Architecture policy

`ArchitecturePolicy` controls:

```text
state_cycles_per_meta
meta_epochs_per_architecture_review
max_architecture_candidates
max_architecture_eval_cells
max_eval_state_cycles
min_architecture_improvement
max_architecture_metric_regression
rejection_penalty
meta_search
```

Candidate policies form a bounded 27-node neighborhood around the incumbent.

## Audit layers

There are now three audit chains:

```text
os-journal.jsonl            state execution history
meta-journal.jsonl          runtime-configuration evolution history
architecture-journal.jsonl  orchestration-policy evolution history
```

ADR-016 migration progressively binds these to common transaction lineage and typed stage receipts.

Together they answer:

```text
What state changed?
What evidence admitted it?
Why did the runtime policy change?
Why did the architecture policy change?
```

## CLI

Run a bounded autonomic demo:

```bash
jarvisx-dr-moagi-system demo \
  --side 16 \
  --cycles 8 \
  --state-cycles-per-meta 4 \
  --meta-epochs-per-architecture 2 \
  --meta-candidates 5 \
  --architecture-candidates 3 \
  --max-eval-cells 128 \
  --pretty
```

Serve the top-level system control plane:

```bash
jarvisx-dr-moagi-system serve --host 0.0.0.0 --port 10001
```

## System API

| Endpoint | Purpose |
|---|---|
| `GET /healthz` | State/meta/architecture journal health |
| `GET /v1/system/capabilities` | Four-scale capability contract |
| `GET /v1/system/status` | Unified nested-system status |
| `POST /v1/system/boot` | Boot lower OS kernel |
| `POST /v1/system/demo` | Load deterministic sparse demo state |
| `POST /v1/system/load` | Load sparse 3D state |
| `POST /v1/system/step` | One authoritative OS transaction |
| `POST /v1/system/run` | Bounded state cycles only |
| `POST /v1/system/meta/optimize` | One runtime-configuration epoch |
| `GET /v1/system/architecture/lattice` | Inspect 27-node architecture neighborhood |
| `POST /v1/system/architecture/evolve` | One architecture-policy epoch |
| `POST /v1/system/autonomic/run` | Close all four loops automatically |

These current APIs predate full ADR-016/017 receipt migration; their existence does not imply every stage already exposes the canonical typed interface.

## Operational boundary

This is bounded self-evolution of computational state, adaptive parameters, runtime configuration, and orchestration policy. It deliberately does not:

- execute arbitrary host commands;
- rewrite arbitrary source code;
- remove the transactional validation pipeline;
- allocate logical sparse space densely;
- infer lossless information preservation from a small latent without residual/side information;
- infer global correctness from local fixed-point convergence;
- claim external state-of-the-art performance without matched benchmarks.

The system-level invariant remains:

```text
PROVISIONAL != AUTHORITATIVE
```

and applies at state, model, runtime-policy, and architecture-policy promotion boundaries.