# Dr Moagi Four-Scale System Auto-Evolution

## Status

The Jarvis-X Dr Moagi stack now has four explicit adaptation time scales:

```text
t  authoritative sparse state X
u  adaptive model memory/parameters (Omega, Theta)
n  runtime configuration C
k  architecture orchestration policy A
```

The evolution hierarchy is:

```text
external sparse state
      |
      v
+-----------------------------+
| Inner transactional OS      |
| X -> bitplane -> fold       |
| -> AutoExec -> DM-DD        |
| -> fixed point -> PiLambda  |
| -> DMOS2 -> atomic commit   |
+-------------+---------------+
              |
              v
+-----------------------------+
| Runtime meta optimizer C_n  |
| compression / adaptation /  |
| spatial dynamics            |
+-------------+---------------+
              |
              v
+-----------------------------+
| Architecture optimizer A_k  |
| cadence / search budget /   |
| promotion resilience        |
+-------------+---------------+
              |
              v
       next autonomic epoch
```

The architecture layer does **not** rewrite source code or remove required safety
stages. It searches bounded orchestration policies around the incumbent system.

## Constitutional execution topology

The following stages are immutable in the architecture controller:

```text
sparse_state
 -> uint64_bitplane
 -> inward_fold
 -> autoexec
 -> deep_distiller
 -> fixed_point
 -> pi_lambda
 -> dmos2_verify
 -> atomic_commit
```

Architecture evolution changes how the nested loops are scheduled and evaluated,
not whether transactional validation exists.

## Four recurrences

### 1. State recurrence

```text
(X, Omega, Theta)_{t+1} = Pi_Lambda[F_C(X, Omega, Theta)_t]
```

A rejected cycle leaves authoritative state and adaptive model state unchanged.

### 2. Model recurrence

```text
E_t = X_t - Xhat_t
Omega_{t+1} = rho Omega_t + (1-rho) E_t
Theta'_{t+1} = Theta_t - eta grad_Theta ||E_t||^2
```

These staged values are committed only with the state transaction.

### 3. Runtime-configuration recurrence

```text
C_{n+1} = Pi_meta[argmin_{C in N3(C_n)} J_runtime(C)]
```

The 3D runtime lattice axes are compression geometry, adaptive dynamics, and
spatial/fixed-point dynamics.

### 4. Architecture-policy recurrence

```text
A_{k+1} = Pi_arch[argmin_{A in N3(A_k)} J_arch(A)]
```

The architecture axes are:

- X: state-to-meta cadence;
- Y: meta-search budget/depth;
- Z: promotion resilience.

Every architecture candidate is executed in isolated kernels over the same bounded
source state. The production state is not used as scratch space.

## Autonomic scheduler

`SelfEvolving3DArchitecture.run_autonomic(cycles)` closes all four loops:

1. execute authoritative OS state cycles;
2. after `state_cycles_per_meta`, run one inward runtime meta epoch;
3. after `meta_epochs_per_architecture_review`, run one architecture epoch;
4. promote only candidates that pass their corresponding gates;
5. stop immediately on a rejected authoritative state transaction.

This creates a hierarchy of increasingly slower adaptation:

```text
state cycles >> model updates >> meta epochs >> architecture epochs
```

## Architecture objective

Architecture evaluation measures the complete nested runtime using:

- reconstruction MSE;
- DM-DD residual RMS;
- fixed-point residual;
- exact transport bytes per source cell;
- active/latent compute proxy;
- phase velocity;
- internal meta-optimization improvement;
- evaluation cost;
- rejection penalties.

The architecture gate additionally limits regression of reconstruction and DM-DD
residual quality.

An internal architecture improvement is **not** an external SOTA claim. Status
therefore keeps `external_sota_verified=false` until matched external benchmarks
exist.

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

Together they answer:

```text
What state changed?
Why did the runtime configuration change?
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

## Operational boundary

This is bounded self-evolution of computational state, adaptive parameters, runtime
configuration, and orchestration policy. It deliberately does not:

- execute arbitrary host commands;
- rewrite arbitrary source code;
- remove the transactional validation pipeline;
- allocate logical sparse space densely;
- claim external state-of-the-art performance without matched benchmarks.

The system-level invariant remains:

```text
PROVISIONAL != AUTHORITATIVE
```

and now applies at state, configuration, and architecture promotion boundaries.
