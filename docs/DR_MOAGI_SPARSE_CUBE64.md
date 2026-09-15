# Dr Moagi Sparse Cube64 Cloud Layer

**Status:** integration specification + dependency-free reference planner  
**Tracking issue:** #272  
**Integration PR:** #88

## Purpose

This layer gives the Dr Moagi Cloud OS a precise interpretation of a
`1000 x 1000 x 1000` logical 3D machine with 64-bit voxel descriptors while
preserving the repository's existing numerical contracts.

The central rule is:

```text
virtual address space != resident working set
```

A `1000^3` lattice contains:

```text
N = 1,000,000,000 logical voxels
```

At exactly 64 bits / 8 bytes per voxel, a dense descriptor cube requires:

```text
8,000,000,000 bytes
```

Therefore the dense cube is not a 1 MiB object. The 1 MiB constraint is a
bounded machine-description / active-payload policy. The runtime materializes
only selected sparse records and tiles.

## Two-layer contract

The new 64-bit layer does **not** replace the existing QVector field engine.

```text
Sparse Cube64 control/state plane
        |
        | selects active addresses / tiles
        v
Q16.16 x 3 QVector numerical plane
        |
        v
field operators / encoder / decoder / verification
```

The roles are deliberately separate:

| Layer | Width | Role |
|---|---:|---|
| Cube64 voxel descriptor | 64 bits | activation, residual, opcode, compact state |
| QVector numerical cell | 96 bits | three signed Q16.16 field components |
| Field bytecode instruction | 128 bits | deterministic field-operation instruction |

This keeps the bit contracts composable instead of forcing unrelated state into
one representation.

## 64-bit voxel word

`jarvisx.sparse_cube64.VoxelWord64` uses:

```text
bits 63..48  payload[16]
bits 47..36  feature[12]
bits 35..28  memory[8]
bits 27..20  activation[8]
bits 19..8   residual[12]
bits 7..0    opcode[8]
```

Total:

```text
16 + 12 + 8 + 8 + 12 + 8 = 64 bits
```

The word is a control/state descriptor. Numerical field values remain in the
QVector plane.

## Sparse global addressing

Each axis of a 1000-cell cube fits in 10 bits because:

```text
0 <= coordinate <= 999 < 1024 = 2^10
```

The reference codec packs `(x, y, z)` into 30 bits held inside one uint32.
A globally addressable sparse record is therefore:

```text
uint32 packed_xyz + uint64 voxel_state = 12 bytes
```

The top two coordinate bits are reserved and rejected on decode.

## 1 MiB working-set arithmetic

For a 1 MiB payload budget:

```text
B = 1,048,576 bytes
record = 12 bytes
max_active_records = floor(B / 12) = 87,381
```

For `10 x 10 x 10` tiles:

```text
1 tile = 1,000 voxels
max_full_tiles = floor(87,381 / 1,000) = 87
```

The active global sparse fraction is therefore below `0.01%` of the billion-cell
virtual lattice.

This number describes the packed reference payload only. A Python process,
dictionaries, queues, HTTP service, allocator metadata, model weights, and
operating-system overhead are outside that 1 MiB payload figure.

If a tile already supplies spatial locality, implementations may omit per-voxel
global coordinates inside that tile and approach the 8-byte state limit. That
is an optimization, not the canonical global sparse-record format.

## Cloud execution loop

The control loop is:

```text
Generate
  -> Partition
  -> Select active tiles
  -> Dispatch to resident workers
  -> Encode / evolve / decode
  -> Aggregate changed summaries
  -> Verify reconstruction
  -> Measure residual + runtime telemetry
  -> Refine / freeze / migrate / evict
  -> Recur
```

Let the system state be:

```text
S_t = [X_t, Z_t, Omega_t, Xhat_t, E_t, Pi_t, C_t]
```

where `Pi_t` is the execution policy and `C_t` is cloud topology/runtime state.
The cloud transition is abstracted as:

```text
S_(t+1) = M_(Theta, Pi_t)(S_t, t, dt)
```

The feedback vector may contain:

```text
F_t = [loss_t, latency_t, bandwidth_t, memory_t, queue_t, stability_t]
```

and the runtime policy evolves as:

```text
Pi_(t+1) = G(Pi_t, F_t)
```

with policy components such as:

```text
tile size
active-set threshold
precision
worker/device placement
batch size
recursion depth
```

## Error determines computation

The intended scheduling invariant is:

```text
high residual / novelty
    -> activate or retain region
    -> finer tiles / more compute / stronger verification

low residual / stable region
    -> freeze or compress region
    -> retain compact state / evict numerical materialization
```

One generic activity law is:

```text
A_t(v) = 1[ residual_t(v) > epsilon OR importance_t(v) > tau ]
```

Only addresses with `A_t(v) = 1` need enter the active numerical working set.

This is the practical meaning of the inward loop: compute contracts toward the
regions where unresolved information remains.

## Data-movement rule

For the full 64-bit descriptor cube, one dense pass is 8 GB of payload before
replication, metadata, numerical tensors, or results. Repeatedly moving the full
cube through a network would dominate the runtime.

The cloud rule is therefore:

```text
move computation to resident state;
do not move the complete state to computation every cycle
```

Workers should retain tiles and exchange bounded information such as:

- changed sparse records;
- halo/boundary data;
- latent summaries;
- residual summaries;
- policy/control messages;
- hashes and verification records.

## Hierarchical memory and scheduling

A scalable implementation should avoid global synchronization on every inner
step:

```text
Pi_global -> Pi_node -> Pi_tile
Omega_global + Omega_regional + Omega_local
```

with:

```text
f_local >> f_regional >> f_global
```

Fast numerical refinement stays close to resident data. Slower global cycles
reconcile summaries, policy, and authoritative state.

## Reference implementation

`src/jarvisx/sparse_cube64.py` currently provides:

- exact 64-bit word packing/unpacking;
- exact 30-bit 3D coordinate packing;
- exact 12-byte sparse-record serialization;
- a zero-allocation virtual-cube working-set planner;
- explicit dense-vs-sparse memory arithmetic.

`tests/test_sparse_cube64.py` locks the boundary values and the canonical
`1000^3`, 1 MiB calculations.

The planner intentionally does **not** allocate a billion-cell cube.

## Implemented boundary

This integration establishes a deterministic software contract and planner. It
does not claim that PR #88 already supplies:

- authenticated multi-host worker transport;
- distributed consensus for Omega state;
- provider autoscaling;
- GPU/HBM sparse residency management;
- zero-copy RDMA;
- production failure recovery across machines;
- a complete executable whose whole process image consumes only 1 MiB.

Those are later implementation layers built on top of this contract.

```text
Working -> Robust -> Portable -> Elegant -> Advanced
```
