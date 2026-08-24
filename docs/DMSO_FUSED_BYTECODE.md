# DMSO Fused Bytecode Execution

## Status

Executable reference backend for PR #84. This layer gives a promoted level-1 DMSO operator an actual execution meaning beyond description compression.

Two reference implementations are provided:

- `src/jarvisx/dmso_bytecode.py`: Python primitive interpreter and fused kernel with benchmark telemetry;
- `cpp_runtime/include/jarvisx/dmso_fused.hpp`: dependency-free C++17 primitive and native fused kernels.

## Canonical primitive trace

The current cell update is represented as seven primitive operations:

```text
LOAD_SELF
AGGREGATE_26
DECODE_FRONT
LOAD_INPUT
AFFINE
TANH
RELAX
```

The primitive interpreter dispatches those operations one by one.

After the runtime observes and verifies the complete sequence, it can promote the trace to one `OperatorDefinition`. The fused compiler accepts only a verified operator whose expansion exactly matches the canonical trace and replaces the seven interpreter dispatches with one direct kernel call.

## Semantic contract

For execution context

```text
(current, neighbour_mean, projected, stimulus, alpha)
```

and parameters

```text
(self_gain, neighbour_gain, projection_gain, input_gain, bias)
```

the fused kernel computes, per channel,

```math
z = w_s s + w_n n + w_p p + w_u u + b
```

```math
m = \tanh(z)
```

```math
s' = s + \alpha(m-s)
```

Before benchmarking, the backend executes both the primitive program and the fused operator and requires exact reference equality. If the values differ, benchmarking aborts.

## What is accelerated

The demonstrated optimization is **interpreter dispatch fusion**:

```text
primitive dispatches per cell update: 7
fused dispatches per cell update:     1
structural dispatch reduction:        7x
```

The benchmark additionally records wall-clock nanoseconds per call and a measured speedup for the specific host and run. Timing is telemetry only; it is never used to prove correctness, select authoritative state, or claim a portable speedup.

### Python reference

Run:

```bash
python examples/dmso_fused_benchmark.py
```

A local reference run on the development host measured approximately `2.46x` fused-versus-primitive speedup with zero output error. That number is host-specific telemetry, not a guaranteed result.

### C++17 native reference

Build through the existing processor laboratory:

```bash
cmake -S cpp_runtime -B build/cpp-runtime -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp-runtime --config Release --parallel
ctest --test-dir build/cpp-runtime -C Release --output-on-failure
```

Run the benchmark:

```bash
./build/cpp-runtime/jarvisx-dmso-fused-benchmark --repetitions 500000
```

The native benchmark reports JSON containing:

- primitive and fused dispatch counts;
- structural dispatch-reduction ratio;
- primitive and fused nanoseconds per call;
- measured host-specific speedup;
- maximum absolute semantic error;
- a checksum that keeps the benchmark result live.

A strict local C++17 build using `-Wall -Wextra -Wpedantic -Wconversion -Wshadow` passed. A 100,000-iteration development run measured approximately `2.59x` on that host with zero semantic error. Repository CI remains authoritative.

## Verification boundary

The C++ regression target `jarvisx-dmso-fused-tests` verifies:

- primitive execution dispatches exactly seven canonical operations;
- fused execution dispatches once;
- the primitive and fused reference outputs are exactly equal;
- non-finite execution inputs are rejected.

No wall-clock threshold appears in CTest because machine timing is not deterministic enough to be a correctness invariant.

## Current boundary

This PR now provides both Python-level and C++17 native direct fusion for the canonical level-1 operator. It does **not** yet provide:

- LLVM or runtime JIT code generation;
- GPU kernel fusion;
- recursive fusion of arbitrary higher-order operators;
- dynamic native code mutation;
- cross-platform performance guarantees.

Subsequent backends should preserve the same rule: compile only verified operator expansions, compare against the primitive reference on bounded fixtures, and treat performance timing as measured telemetry rather than an authority signal.
