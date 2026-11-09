# JARVIS-X Omega3 Fractal Auto-Engineering Deployment

This is an additive, versioned research runtime and **not** the locked v1.0 source.
A path-scoped GitHub Actions workflow builds this C++17 program on a hosted Ubuntu
runner with GCC and Clang. Its Python engineering controller verifies:

- Byte-for-byte golden v1 trace at depth 2 (400 cells, 672 links, 24 steps).
- Identical trace and checkpoint output for 1, 2, 4 and 8 CPU threads.
- Checkpoint continuation on a different thread count.
- Rejected fault injection with prior checkpoint byte-identical.
- Rejection of deliberately corrupted checkpoints.
- Depth-3 8,000-cell graph counts and cross-worker determinism.
- Depth-4 160,000-cell graph counts and cross-worker determinism.
- Exploratory timings and advisory worker selection **without modifying source**.

Runs are triggered by pushes and pull requests touching the isolated runtime,
and can also be dispatched manually after the workflow is present on the
repository's default branch. Each compiler job uploads engineering artifacts.

## Reproduce locally

```bash
cmake -S experimental/fractal_cloud_v2 -B /tmp/jx3d-build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/jx3d-build --parallel 2
python3 experimental/fractal_cloud_v2/tools/engineer.py \
  --binary /tmp/jx3d-build/jx3d_cloud \
  --golden experimental/fractal_cloud_v2/tests/golden_v1_depth2_steps24.csv \
  --output /tmp/jx3d-engineering --depth 4 --steps 24
```

## Security and cloud boundary

Automated GitHub runs use only read permissions and are bounded to depth <=4.
No cloud credentials, secrets, billing, GPU, remote nodes, container registry,
or Kubernetes cluster are provisioned. The Dockerfile and Kubernetes Job are
**deployment handoff files** only. The multithreaded partitions and halo
exchange operate **within one process on one machine**. Do not conflate them
with multi-node RDMA/MPI/NCCL. No trained ANN has been installed. Do not
enable automated merges or broad repository write permissions without review.
