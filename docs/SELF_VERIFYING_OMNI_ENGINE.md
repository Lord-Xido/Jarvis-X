# Dr Moagi self-verifying inward engine

Classification: **numerical reference / integration candidate**. This C++17 engine
implements the supplied million-pathway inward recurrence as a bounded, seeded
numerical workload with independently recomputed verification. It is a separate
native reference surface, not a canonical VM or worker-fabric adapter.

## Build and run

```bash
cmake -S cpp_runtime -B build/cpp-runtime -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp-runtime --parallel 2 \
  --target jarvisx-self-verifying-omni jarvisx-self-verifying-omni-tests
ctest --test-dir build/cpp-runtime --output-on-failure -R self-verifying-omni
./build/cpp-runtime/jarvisx-self-verifying-omni --threads 2 --json
./build/cpp-runtime/jarvisx-self-verifying-omni \
  --pathways 257 --depth 1024 --threads 2 --require-convergence --json
```

On multi-configuration generators, add `--config Release` for build and `-C Release`
for CTest; the executable lives in the `Release` subdirectory.

Direct GCC build, with optional OpenMP:

```bash
g++ -std=c++17 -O3 -fopenmp -Wall -Wextra -Wpedantic \
  -Wconversion -Wshadow -Icpp_runtime/include \
  cpp_runtime/src/self_verifying_omni_main.cpp -o jarvisx-self-verifying-omni
./jarvisx-self-verifying-omni --threads 2 --json
```

Omit `-fopenmp` for a serial direct build. CMake detects OpenMP without requiring
it; `-DJARVISX_OMNI_ENABLE_OPENMP=OFF` explicitly exercises the serial fallback.
There are no x86-specific intrinsics or target-specific alignment attributes.

| Option | Default | Accepted values |
|---|---:|---|
| `--pathways` | 1,000,000 | 1 through 10,000,000 |
| `--depth` | 16 | 1 through 4,096 |
| `--threads` | 64 requested | 1 through 1,024; actual team is resource-capped |
| `--seed` | 0 | unsigned 64-bit integer |
| `--json` | off | emit one schema-version-1 JSON receipt |
| `--require-convergence` | off | require every pathway to meet both tolerances |

Exit status is `0` for successful verification, `1` for failed mechanics, `2` for
incomplete convergence when explicitly required, and `64` for invalid arguments
or an allocation failure. A verified bounded run may still contain unconverged
pathways. JSON states this separately as `verification_passed` and `all_converged`.

## Numerical state and recurrence

Each pathway has a position `p` in the unit cube, eight scalar channels `s_c`, a
double-precision complex token `v`, a residual MSE, a budget `b` and an active mask.
The immutable `16³` grid samples `f(p) = exp(-4 ||p - (0.5,0.5,0.5)||)` at grid
nodes. Lookup uses the lower grid node after clamping finite coordinates; it is
piecewise constant, not trilinear interpolation or a continuous field solver.

For each active pathway:

\[
h=0.05 f(p)b,\qquad p'=p+h(c-p),\qquad c=(0.5,0.5,0.5),
\]
\[
s'_k=0.9s_k+0.1f(p),\qquad
r'=\frac18\sum_{k=0}^{7}(s'_k-f(p'))^2,
\]
\[
v'=\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}v,
\qquad \theta=\pi f(p)b.
\]

The token uses standard double-precision sine/cosine and fused multiply-add.
It is not renormalized to conceal drift. A candidate is staged in local storage;
non-finite values, invalid bounds, an outward radius change or a failed unit-norm
gate reject the complete pathway update. Rejections remain a sticky failure in
the engine's verification receipts. This is a local numerical gate, not ADR-016
rollback across canonical system namespaces.

A pathway is converged only when `r' <= 0.001` and `||p'-c|| <= 0.001`. The C++
`Config` exposes both tolerances and rejects non-finite or out-of-range values.
Budget becomes `0.8b` while both gates hold and `0.8b + 0.2` otherwise. Pruning
requires convergence plus budget below `0.01`; pruned budget becomes exactly zero.
Small per-step changes alone cannot establish convergence. Eight channels share
the same initial zero state and scalar target in this reference and therefore
remain identical; they are independent storage lanes, not learned modalities.

`0.8^16 ≈ 0.02815`, so the default 16-step run cannot reach the pruning cutoff,
even if a pathway is already converged. The earliest possible cutoff is step 21.
The longer convergence fixture verifies actual pruning and skipped updates.

## Independent audit

The verifier reads every authoritative array and independently recomputes:

- per-pathway and aggregate squared token norms, including a mean absolute error
  that cannot hide opposite-signed errors;
- finite values, channel/position/budget bounds and exact mask membership in `{0,1}`;
- actual residual MSE at the final grid coordinate versus the recorded residual;
- final radius versus its initialized radius;
- both convergence conditions for every inactive pathway;
- receipt bounds and absence of rejected candidates.

Maximum per-pathway norm error must be at most `1e-10`; mean absolute norm error
and relative aggregate norm-sum error must be at most `1e-11`. These are declared
floating-point tolerances, not claims of exact arithmetic or topological proof.
Thread/partition counters are not a proof of memory safety or race freedom.

The submitted fourth/fifth-order sine/cosine polynomials give
`sin_approx(pi)^2 + cos_approx(pi)^2 ≈ 0.28998`; that is an algebraic approximation
error, not evidence of SIMD misalignment. Regression tests reproduce the failure,
then exercise corrected rotation across the full angle interval. Further tests
cover opposite norm errors, NaN/Infinity, mask/budget/shape corruption, cached
residual tampering, premature pruning, seeded replay across thread counts,
convergence and frozen inactive state.

## Resources, determinism and boundary

Aligned heap vectors store positions and channel-major state. Array and grid
payload is `80*N + 16,384` bytes: **80,016,384 bytes** at the default million
pathways. This excludes vector metadata, allocator overhead and thread stacks;
it is not measured process RSS. Construction rejects the resident bound before
allocating. Execution is bounded by `N*depth` attempted updates and `O(N)` resident
storage. Pruning skips pathway updates but still scans the active masks each step.

The requested team is capped by the OpenMP runtime, reported processor count and
pathway count; the receipt records the actual team. A million pathways are data
items scheduled through that team, not a million simultaneous hardware workers.
No assumption is made that the host has 64 physical cores.

A counter-based integer mixer seeds each coordinate without shared `rand()` state.
Pathways write only their own array entries and read an immutable grid. The final
audit sums in fixed index order. Thread-count replay is tested within one build;
bit-exact results across compilers, architectures or math libraries are not claimed.
Elapsed time uses `steady_clock` and does not influence acceptance or pruning.

The reference implements inward geometric contraction, scalar state relaxation,
unit-norm verification and residual-aware scheduling. It does not implement a
trained encoder/decoder, model-weight learning, distributed consensus or native
self-recompilation. No `1000x` speedup is asserted without a measured baseline.
There are no persistent formats, input files, network calls or external mutations.
Recreate an engine from its configuration for a seeded restart; the JSON receipt
is telemetry, not a checkpoint or canonical state promotion record.
