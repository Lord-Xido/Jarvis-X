# JARVIS-X Ω³ — 3D Fractal Cloud Execution Runtime v2.0

**Status:** operational, deterministic **local CPU-sharded reference** with a portable offline 3D replay viewer and a single-pod cloud deployment handoff. This is a separate extension; the locked Dr Moagi Fractal Algebra v1.0 reference remains unchanged.

## What runs end to end

1. Recursively build the Menger-sponge 3D lattice (20 retained child positions per level).
2. Encode each 3D position to a five-bit-per-level packed fractal address.
3. Build six-face (±X, ±Y, ±Z) neighbour topology with precomputed neighbour indices and a six-bit occupancy mask.
4. Partition the cell range among local CPU worker threads. Produce explicit per-partition halo lists and copy boundary state into read-only ghost buffers before each iteration.
5. Execute the original deterministic Q16 consensus equation in parallel. Each worker writes its own slice of the next-state buffer.
6. Independently recompute candidate arithmetic and check invariants before a single synchronous global commit; a fault can trigger rollback.
7. Emit the canonical v1.0-compatible FNV-1a chain and full or sampled per-cell event logs with spatial addresses, bitwise deltas, neighbour sums, codec residuals and commit identity.
8. Encode the final lattice into a real **lossy 8-bit state-payload file** and a lossless Q16 checkpoint, along with a portable 3D replay artifact.
9. Resume a checkpoint on a different number of CPU workers and continue the exact hash chain from its stored parent digest.
10. Use the included verifier to independently test graph topology, checkpoint encoding, quantisation, event correctness, replay, rollback, restart and tamper rejection.
11. Build a Docker image and submit the supplied Kubernetes **single-pod Job** after providing a registry and suitable cluster/PVC.

## Quick start

Linux x86-64 with C++17 compiler, CMake and Python 3.8+:

```bash
./run_local.sh
python3 tests/verify_runtime.py --binary build/jx3d_cloud
```

Alternatively, from the package root:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -pthread src/jx3d_cloud.cpp -o jx3d_cloud
./jx3d_cloud --depth 2 --steps 24 --workers 4 --out output --events full
python3 tools/embed_viewer.py --frames output/frames.json --output output/viewer.html
# Open output/viewer.html locally; no server or internet connection is needed.
```

For a larger test:

```bash
./jx3d_cloud --depth 4 --steps 24 --workers auto --events none --no-frames --out deep-output
```

For deterministic checkpoint continuation (absolute `--steps` target):

```bash
./jx3d_cloud --depth 2 --steps 1 --workers 1 --out partial
./jx3d_cloud --depth 2 --steps 24 --workers 8 --resume partial/checkpoint.bin --out continued
# continued/checkpoint.bin is byte-identical to an uninterrupted 24-step run.
```

For the rollback test:

```bash
./jx3d_cloud --depth 2 --steps 10 --workers 4 --inject-fault-step 2 --out fault-output
# Exit code 2 means rejected transaction; checkpoint stores the last committed step (1).
```

## Results reproduced at depth 2

| Criterion | Measured result |
|---|---:|
| Cells | 400 |
| Undirected face links | 672 |
| Executed iterations | 24 |
| Local CPU partitions | 4 |
| Unique copied halo values/iteration | 192 |
| Final canonical v1 hash | `098794e9b04115c7` |
| Canonical `trace.csv` | **Byte-for-byte identical** |
| Cross-worker tests | 1/2/4/8 threads, matching v1 trace |
| Rollback test | Fault at iteration 2 rejected, prior checkpoint preserved |
| Restart test | Resume after step 1 on 8 workers produces same step-24 checkpoint |
| Depth-4 throughput | 160,000 cells × 24 steps; ~1.08 s (4 workers, observed local run) |

Depth 3 is tested for 24 iterations on 1/4 workers; depth 4 for a single iteration on 4 workers. The complete verifier records all checks in `outputs/verification.json`. GCC/Clang equivalence and a short AddressSanitizer+UBSan run were also confirmed. On this one CPU environment, depth-4 24-step runs took ~1.48 s with one worker and ~1.08 s with four workers (single observations, not an established speedup).

## Core mathematics

Fractal occupancy and degree:

```text
N(L) = 20^L                         active cells
side(L) = 3^L                       coordinate resolution
edges(L) = 2*(20^L - 8^L)           undirected face-neighbour edges
path = sum_k(symbol_k << (5*k))     packed child path
```

Canonical integer Q16 update, using *old-state* neighbours:

```text
q_next[i] = floor((3*d*q[i] + sum(q[nb]) + 2*d) / (4*d)),  d > 0
q_next[i] = q[i], d = 0
```

The old graph state is immutable throughout the local shard computation. No partition may read a partially committed update. Aggregation and hash creation iterate cells in the canonical sequential order.

### Integrity and fidelity

- `trace.csv` matches the locked v1.0 execution chain exactly.
- `quality.csv` has the **true nonrounded RMS change** and exact changed-cell/changed-bit counts, in addition to the v1-compatible rounded diagnostic in `trace.csv`.
- `events.csv` records the input state, neighbour sum, packed address, six-direction mask, XOR/Hamming change, Q8 code, decoded Q16 and parent commit hash. `--events full` records all committed cell transitions; `sample` records an illustrative subset.
- `checkpoint.bin`: 8-byte ASCII magic `JX3DCHK2`, LE `uint32 depth`, LE `uint32 last_step`, LE `uint64 node_count`, LE `uint64 digest`, LE `uint64 parent_digest`, and LE `uint16 q16[node_count]`. A restart validates the complete digest against its parent before permitting further computation. Coordinates are deterministically regenerated from the depth and canonical child ordering.
- `codec_q8.bin`: 8-byte ASCII magic `JX3DQ801`, LE `uint32 depth`, LE `uint32 last_step`, LE `uint64 node_count`, then `uint8 q8[node_count]`, with scalar `q8 = floor((q16*255+32767)/65535)` and `decoded = floor((q8*65535+127)/255)`. Maximum per-cell quantisation error is 128 Q16 units. This is **lossy codec payload**, not yet an optimised entropy-coded format or learned autoencoder.
- FNV-1a snapshot hashes support reproducibility, **not cryptographic tamper resistance**. The checkpoint verifies its embedded FNV parent/snapshot relationship, which detects ordinary accidental corruption but is not a secure anti-forgery mechanism. SHA-256 file manifests improve accidental-corruption detection but do not provide authentication or a trusted signing identity.

## Output files

| File | Purpose |
|---|---|
| `trace.csv` | Canonical reference-compatible time-series/hash trace |
| `quality.csv` | Exact change metrics, edge-difference energy, halo-copy counts |
| `events.csv` | Full or sampled cell-level operational trace |
| `state.csv` | Final 3D coordinates, packed address, owner, face mask, Q16/Q8 |
| `checkpoint.bin` | Lossless validated deterministic restart encoding |
| `rejections.csv` | Populated on rejected iterations, including failing candidate and gate |
| `codec_q8.bin` | Operational lossy Q8 encoding of final state |
| `frames.json` | Sampled 3D geometry and per-iteration state values |
| `viewer.html` | Portable HTML5 offline 3D replay UI, created by embed_viewer.py |
| `run.json` | Worker counts, topology, elapsed time, final hash, status |
| `verification.json` | Automated test report |

## 3D viewer

The included viewer rotates a rendered volumetric fractal using a 3D camera transform. At <= 1600 displayed cells it draws voxel faces; for larger data it switches to a 3D point-cloud view for interactive performance. Drag to rotate, scroll to zoom, play the time-series, inspect the last journal hash, and toggle partition ownership, Q16 colour, bit differences, or decoded Q8. At higher recursion levels `frames.json` samples at most 8,000 display cells, even when all cells have been computed and checked.

## Cloud readiness and exclusions

**Implemented and tested:** local CPU-threaded parallel computation, ghost/halo buffering, exact numerical equivalence, full spatial coordinates, a real binary codec, lossless checkpoints, rollback and offline interactive visualization.

**Handoff provided but not remotely exercised:** Docker multi-stage container image, Kubernetes Job and persistent volume template. Build/push/deploy needs your cloud registry, cluster and credentials. No Docker daemon or remote cloud execution was used in this verification environment.

**Not yet implemented:** true multi-node networking (MPI/NCCL/RDMA), GPU acceleration, automatic remote node provisioning, authenticated distributed journal/consensus, trainable ANN encoder-decoder, fully adaptive fractal occupancy, or demonstrated 1000× speedup. The canonical v1.0 loop is geometric consensus, not trained neural intelligence.

## Design invariants

- No writes to locked original reference files.
- Fixed child symbol ordering matches canonical v1.0 recursion (z/y/x nested ordering).
- Q16 values stay in `[0,65535]`, and candidate arithmetic is recomputed independently before commit.
- Each cell is owned by one local shard; remote neighbour reads use the halo snapshot.
- Per-cell journal emission and snapshot hashes use a globally stable canonical order.
- Fault injection rejects a candidate and checkpoints the prior committed state; the fault reason is written to `rejections.csv`. Checkpoint continuation on a different worker count preserves the canonical output hash.
- Compressed Q8 reconstruction is explicitly lossy; it is not silently fed back into the original exact loop.

For production use, split the control plane and execution plane, implement real networked transport/barriers, add journal signing and crash recovery, run load/chaos tests, and benchmark against a fixed hardware and numerical-fidelity target.
