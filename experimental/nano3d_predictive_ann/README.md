# JARVIS X Nano3D: One-shot verified computation reference

**One command performs an actual million-step geometric trajectory, trains and
precomputes a synthetic 3D classifier, predicts C++ function templates,
independently checks every generated source line, compiles and executes a bounded
behavioral test harness, and transactionally commits validated source plus evidence.**

**The two loops are distinct:** 1,000,000 optimizer steps are performed on an
explicit synthetic S² objective, not one million program-generation iterations.
The 32³ ANN table classifies octants, not real C++ semantics. Output semantics
are established separately by a typed independent oracle and four runtime
inputs per function. This is **not** a demonstration of million verified LOC/s.

## One-command execution (Linux, Python 3, GCC/Clang)

```sh
python3 experimental/nano3d_predictive_ann/one_shot.py \
   --root /tmp/jarvisx-nano3d-one-shot --release first --steps 1000000 \
   --files 4 --lines 128 --task-version demo-v1

# Verify cached deterministic reuse on unchanged context:
python3 experimental/nano3d_predictive_ann/one_shot.py \
   --root /tmp/jarvisx-nano3d-one-shot --release second --steps 1000000 \
   --files 4 --lines 128 --task-version demo-v1
```

Each command invokes a local C++17 compiler, runs the geometric optimizer and
checks every emitted line against a separately coded Python oracle. It performs
C++ per-file syntax checks and compiles/executes every function on four
unsigned-32-bit test inputs. It hashes source and writes a receipt, then atomically
publishes the release directory and updates the latest pointer. The trusted
workdir uses a Linux process lock to avoid two simultaneous writers.

Cache reuse requires matching source/optimizer hashes, compiler version, task
revision, workload and test-policy version. A manifest with matching SHA-256
source hashes is checked before reuse; **all syntax and behavioral tests still
run** even when the source is retrieved. A changed task-version or tampered cache
forces regeneration. Cache identities are content hashes, not signatures or
protection against hostile writers. Local storage is trusted. Only visibility
atomicity is claimed; no power-failure fsync guarantees.

### Independent tests

```sh
cmake -S experimental/nano3d_predictive_ann -B build/nano3d -DCMAKE_BUILD_TYPE=Release
cmake --build build/nano3d --parallel 2
ctest --test-dir build/nano3d --output-on-failure
```

The dedicated GitHub Actions workflow tests both GCC and Clang.

## Mathematics

The adaptive trajectory is on the *unit sphere* S², with objective
`f(z)=1-dot(z,t)` where `t=normalize(0.31,0.41,0.86)`. The gradient is
`grad f(z) = -t + dot(z,t)*z`. It updates a tangent momentum, a *scalar* second
moment from squared Riemannian gradient norm, a sphere exponential map, and
closed-form shortest-arc parallel transport of momentum. This is Riemannian
momentum with adaptive scalar preconditioning; it is not ordinary coordinate-
wise Adam on an arbitrary manifold. The end-state and maximum norm drift are
verified; the loss is synthetic and does not measure C++ code quality.

The *separate* predictor is the original 3-input/8-output octant classifier,
precomputed into a 65,536-byte Morton cache. It emits deterministic C++
uint32 unsigned arithmetic templates rather than solving arbitrary prompts.

Cache decision uses **illustrative, user-overridable cost assumptions**:

`gain_ms = expected_reuses*(p_valid*(fresh_ms-lookup_ms) - (1-p_valid)*invalidation_ms) - precompute_ms`

Cache is used only if an existing matching entry is found and `gain_ms > 0`.
Cost scenario defaults are not hardware measurements, nor evidence of
speedup. In the local tests, a cached run was *slightly slower* end-to-end
than an uncached run because compilation and full re-verification dominate.

## Limits and authority

- One-shot demo limits: <=40 files, <=850 LOC/file, <=8,000 LOC total per invocation.
- Geometry: 1..1,000,000 exact iterations, constant-memory.
- 32³ prediction table: 64 KiB; no neural natural-language reasoning.
- Behavioral checks: four fixed inputs per function (not exhaustive C++ proof).
- All tests run only against known, generated templates. Never execute untrusted
  generated code on the host without a sandbox, or treat this harness as one.
- No privileged deployment or merge is performed by the program.
- No generalized 1M LOC/s throughput or nanosecond request latency claim.
