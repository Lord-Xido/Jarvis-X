# Dr. Moagi Cloud Control Plane

**Status:** Operational reference layer  
**Applies to:** Jarvis-X sparse 3D runtimes, 1K³ cloud-organ architecture, streamed block engines, future CPU/GPU/accelerator workers.

This layer turns the canonical cloud-organ scheduling model into an executable
control protocol. It does not claim that a logical `1024³` body is physically
resident or that the Python reference runtime is connected to a production
cloud provider.

## Operational invariant

```text
state
  -> versioned tile task
  -> deterministic route
  -> candidate execution
  -> CTR verification
  -> commit | reject
  -> immutable receipt
  -> next state
```

The implementation is `src/jarvisx/cloud_control_plane.py`.

## State model

A task is

```text
T = (task_id, tile_id, operation, parent_version,
     bytes_estimate, preferred_kind, priority)
```

and a worker observation is

```text
W = (worker_id, region, kind, capacity, free,
     queue_depth, latency, resident_tiles, healthy)
```

Authoritative tile state is versioned:

```text
S_b = (tile_id, version, value, value_hash)
```

A worker never writes `S_b` directly. It receives an immutable snapshot and
returns a candidate value.

## Routing

For worker `n`, the reference scheduler minimizes

```text
J(n) =
    w_L * latency_cost
  + w_Q * queue_cost
  + w_M * memory_cost
  + w_T * transfer_cost
  + w_K * kind_cost
```

with:

```text
latency_cost = latency_ms / 100
queue_cost   = queue_depth / capacity_units
memory_cost  = 1 - free_units / capacity_units
transfer_cost =
    0                              if tile is resident
    bytes_estimate / 1e9           otherwise
kind_cost =
    0                              if worker kind matches
    1                              otherwise
```

Unhealthy workers and workers with no free capacity are excluded. Equal scores
are broken by worker ID, making placement deterministic for a fixed worker
snapshot.

## Candidate-first execution

The dispatch protocol is:

```text
1. resolve task idempotency
2. choose worker
3. read authoritative tile snapshot
4. reject stale parent version
5. execute candidate in shadow
6. run CTR verifier
7. re-check authoritative version/hash
8. commit candidate or reject
9. emit receipt
```

The corresponding state transition is:

```text
S_b^(v)
  -> candidate C_b
  -> V_CTR(S_b^(v), C_b)
  -> S_b^(v+1)      if verified and still current
  -> S_b^(v)        otherwise
```

This is deliberately compatible with the canonical Jarvis-X kinetic sequence:

```text
snapshot -> observe -> encode -> propose -> shadow -> verify
         -> commit | rollback -> receipt -> re-enter
```

## Concurrency and stale-state rejection

Each task names the parent version it observed. If

```text
task.parent_version != authoritative.version
```

the task is rejected before execution.

After candidate execution, the control plane checks the state again. A changed
version or value hash produces `concurrent-state-change` rather than silently
overwriting newer state.

This implements optimistic concurrency control over the distributed tile field.

## Retry semantics

A successfully committed `task_id` is idempotent. Replaying it returns the
original route/receipt and does not execute the worker again or advance the tile
version.

Rejected tasks are not cached as successful commits; callers may repair the
cause, obtain a current parent version, and submit a new task identity.

## Receipt

Every attempted transition yields a deterministic receipt containing:

- task and tile identity;
- selected worker;
- operation;
- parent and resulting versions;
- commit/reject decision;
- rejection/verification reason;
- parent, candidate, and resulting hashes;
- route score;
- receipt hash.

The receipt is evidence about the control-plane transaction. It is not a claim
that an external cloud service, GPU, or remote network action occurred unless a
real backend is explicitly connected.

## Relationship to the 1K³ organism

For cloud organ `B[i,j,k]`, the tile identifier can map directly to its spatial
coordinate. The control plane supplies the missing operational bridge between:

```text
32³ cloud organs
  -> adaptive worker allocation Pi_t / W_t
  -> local candidate compute
  -> verified state reconciliation
  -> global/local recurrence
```

Compute can therefore follow information difficulty while the authoritative
state remains deterministic and auditable.

## Tests

`tests/test_cloud_control_plane.py` verifies:

- data-local routing under high transfer cost;
- unhealthy-worker exclusion;
- verified candidate commit;
- idempotent retry;
- CTR rejection without state mutation;
- stale-parent rejection before execution;
- fail-closed behavior when no worker is schedulable.

## Capability boundary

This is a backend-neutral control plane. A production cloud adapter still needs
transport, authentication, worker discovery, encryption, persistence, rate
limits, observability, and provider-specific failure handling. Those concerns
must not bypass the candidate-first verification and versioned commit boundary.
