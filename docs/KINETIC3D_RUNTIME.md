# Jarvis-X Kinetic 3D Adaptive Runtime

## Status

This module operationalizes the next step beyond the static 3D Bit Code codec as a deterministic CPU reference implementation of a kinetic state-transition runtime.

The runtime does **not** claim frontier GPU performance. It makes the execution semantics concrete and testable so they can later be lowered to Triton/CUDA/MLIR without changing the observable contract.

## Canonical kinetic loop

```text
OBSERVE
  -> PREDICT
  -> RESIDUAL
  -> ACTIVE_SET
  -> ENCODE_COARSE
  -> LATENT_WRITE
  -> REFINE
  -> DECODE
  -> VERIFY
  -> COMMIT
  -> TELEMETRY
  -> EMIT
  -> HALT
```

Its 3D execution depth forms an inward/outward wave:

```text
z=5                         LATENT_WRITE
                            /         \
z=4              ENCODE_COARSE       REFINE
                   /                    \
z=3          ACTIVE_SET               DECODE
               /                           \
z=2      RESIDUAL                       VERIFY
           /                                  \
z=1   PREDICT                           COMMIT -> TELEMETRY
       /                                         \
z=0 OBSERVE                                      EMIT -> HALT
```

The coordinates mean:

- `x`: pipeline position;
- `y`: logical execution tick;
- `z`: execution depth.

## State law

The operational state is:

\[
\mathbb J_t=(W_t,\hat W_t,R_t,A_t,Z_t,G_t,\Lambda_t,E_t)
\]

where:

- \(W_t\) is the observed 3D world;
- \(\hat W_t\) is the persistence prediction;
- \(R_t=W_t-\hat W_t\) is the residual field;
- \(A_t\) is the active set of cells whose residual magnitude exceeds a threshold;
- \(Z_t\) is the hierarchical residual latent;
- \(G_t\) is the per-block execution path schedule;
- \(\Lambda_t\) is the verification tolerance;
- \(E_t\) is reconstruction error.

The committed transition is:

\[
W_{t+1}^{commit}=
\begin{cases}
\tilde W_t, & \max |W_t-\tilde W_t|\le\Lambda_t\\
W_t^{commit}, & \text{otherwise}
\end{cases}
\]

A failed verification therefore does not mutate authoritative runtime state.

## Active-volume execution

Only cells satisfying

\[
|R_i|>\tau_{active}
\]

enter the active set. Inactive cells remain at their predicted state.

This means compute tracks change rather than the full represented world whenever the residual field is sparse.

## Hierarchical residual codec

Active residuals are grouped into 3D blocks of edge length `coarse_factor`.

For block \(b\):

\[
z_b=\frac{1}{|A_b|}\sum_{i\in A_b}R_i
\]

The coarse latent stores \(z_b\). Each active cell receives this coarse correction.

A fine correction is stored only when:

\[
|R_i-z_b|>\tau_{refine}.
\]

Therefore uniform regional changes may be represented by one coarse latent value, while irregular changes selectively allocate fine residuals.

With `refine_threshold=0`, reconstruction is exact for every active cell. Raising the threshold trades latent size for error under the explicit verification budget.

## Persistent prediction

If `previous` is supplied, it is used as the predictor. Otherwise the runtime uses its last committed state when the shape matches. If there is no committed state, the predictor is zero.

A successful result becomes the next committed world and increments the epoch. A failed result leaves the epoch and committed world unchanged.

## Path scheduling

Each active coarse block becomes an independent path assignment:

```text
PathAssignment {
  path_id
  block[x,y,z]
  active_cells
  resource
  estimated_cost
}
```

The current backend assigns every path to `cpu-reference:0`. This is an explicit backend boundary. Future scheduling can map paths to CPU/GPU/remote accelerators while preserving deterministic state semantics.

## Telemetry

Every execution reports:

- total and active cells;
- active fraction;
- coarse latent count;
- fine correction count;
- latent value count;
- approximate value compression ratio;
- estimated bytes moved;
- elapsed reference-runtime time;
- backend identifier;
- MSE, max error, tolerance, and SHA-256 checksum.

The value-compression ratio counts latent scalar values only; it is an algorithmic indicator and does not include index/metadata overhead.

## API

Run locally:

```bash
docker compose -f deploy/kinetic3d/docker-compose.yml up --build
```

The API is exposed on port `8081` and Prometheus on `9091`.

Example exact transition:

```bash
curl -sS \
  -H 'content-type: application/json' \
  -d '{
    "session_id": "demo",
    "shape": [2,2,2],
    "values": [0,1,2,3,4,5,6,7],
    "active_threshold": 0,
    "coarse_factor": 2,
    "refine_threshold": 0,
    "tolerance": 0
  }' \
  http://127.0.0.1:8081/v1/kinetic3d/execute
```

The next request with the same `session_id` automatically uses the committed reconstruction as its predictor unless `previous` is supplied explicitly.

State inspection:

```bash
curl -sS http://127.0.0.1:8081/v1/kinetic3d/state/demo
```

Reset:

```bash
curl -sS -X DELETE http://127.0.0.1:8081/v1/kinetic3d/state/demo
```

Metrics:

```bash
curl -sS http://127.0.0.1:8081/metrics
```

## What is operational now

- typed spatial kinetic IR;
- persistent 3D world state and epochs;
- prediction from prior committed state;
- residual field computation;
- thresholded active-volume selection;
- hierarchical coarse residual encoding;
- selective fine refinement;
- block-level path scheduling;
- sparse reconstruction;
- verify-before-commit transactional state;
- rollback by non-commit on failed verification;
- per-session network API;
- Prometheus-compatible telemetry;
- Docker packaging and smoke-testable deployment.

## Next backend milestone

The next performance step is not to change the state law. It is to lower the same path operations to hardware:

```text
Kinetic Spatial IR
  -> typed tensor/path IR
  -> Triton or MLIR lowering
  -> CUDA/HIP kernels
  -> hardware-aware path scheduler
```

The required benchmark is verified useful work per second, byte moved, and joule against CPU, PyTorch, Triton, and accelerator-native baselines.
