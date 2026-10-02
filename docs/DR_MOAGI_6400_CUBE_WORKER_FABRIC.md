# Dr Moagi 6400³ Logical Worker Fabric

**Status:** executable sparse reference  
**IDE API:** `/v1/fabric`, `/v1/fabric/step`  
**Implementation:** `src/jarvisx/dr_moagi_worker_fabric.py`

## Geometry

The worker address space is exactly:

```text
Λ_W = {0, ..., 6399}³
|Λ_W| = 6400³ = 262,144,000,000
```

A logical worker is an addressable work-domain coordinate. It is **not** a
native operating-system thread and it is not assumed to be physically resident.

The lattice is partitioned into 64³-worker bricks:

```text
6400 / 64 = 100 bricks per axis
100³ = 1,000,000 logical bricks
64³ = 262,144 logical workers per brick
```

## Execution model

The reference scheduler separates four quantities:

1. **logical workers** — the full 262.144-billion coordinate domain;
2. **active workers** — sparse workers selected for the current measured step;
3. **resident bricks** — the bounded working set admitted at once;
4. **physical workers** — bounded host threads executing the active work.

The default reference profile is:

| Quantity | Default |
| --- | ---: |
| Logical dimensions | 6400 × 6400 × 6400 |
| Logical workers | 262,144,000,000 |
| Brick side | 64 |
| Logical bricks | 1,000,000 |
| Workers per brick | 262,144 |
| Active workers / step | 4,096 |
| Resident brick limit | 128 |
| Physical worker threads | host CPU count, capped at 64 |
| Maximum active workers / request | 100,000 |

The execution fraction for one step is measured as:

```text
f_exec = N_active / 262,144,000,000
```

No part of the implementation labels that fraction as simultaneous physical
parallelism.

## Addressing

The canonical row-major worker ID is:

```text
id(x, y, z) = x + 6400*y + 6400²*z
```

The inverse mapping is exact and tested at domain boundaries. Brick ownership is:

```text
b(x, y, z) = (floor(x/64), floor(y/64), floor(z/64))
```

This gives a stable bridge between 3D geometry, sparse scheduling, telemetry and
future distributed placement.

## Reference step

A fabric step deterministically samples an active work domain from the full
logical lattice, partitions the active range over a bounded `ThreadPoolExecutor`,
executes a small encode/fold/decode residual kernel, reduces the results and
returns a receipt containing:

- active workers actually executed;
- active bricks touched;
- resident bricks admitted;
- physical worker count;
- execution fraction;
- residual RMS;
- checksum;
- measured wall time;
- a bounded coordinate/state sample for visualization.

The deterministic kernel is a control-plane/reference workload. It is not a
claim that the reference Python implementation performs production ANN inference
for all 262.144 billion workers.

## IDE surface

`GET /v1/fabric` reports the topology and current scheduler state.

`POST /v1/fabric/step` accepts:

```json
{
  "active_workers": 4096,
  "sample_size": 256
}
```

The ANN IDE visualizes returned sample coordinates as a projection of the
6400³ logical lattice and displays logical, active, resident and physical counts
separately.

## Invariant

The fabric preserves the repository-wide systems rule:

```text
logical address space != active work != resident state != physical parallelism
```

Future C++/CUDA/WebGPU/distributed backends may replace the reference scheduler
without changing this accounting contract.
