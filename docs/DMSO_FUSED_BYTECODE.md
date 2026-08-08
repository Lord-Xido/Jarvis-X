# DMSO Fused Bytecode Execution

## Status

Executable reference backend for PR #84. This layer gives a promoted level-1 DMSO operator an actual execution meaning beyond description compression.

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

Before benchmarking, the backend executes both the primitive program and the fused operator and requires exact floating-point equality for the reference formula. If the values differ, the benchmark aborts.

## What is accelerated

The demonstrated optimization is **interpreter dispatch fusion**:

```text
primitive dispatches per cell update: 7
fused dispatches per cell update:     1
structural dispatch reduction:        7x
```

The benchmark additionally records wall-clock nanoseconds per call and a measured speedup for the specific host and run. Timing is telemetry only; it is never used to prove correctness, select authoritative state, or claim a portable speedup.

Run:

```bash
python examples/dmso_fused_benchmark.py
```

The JSON result reports:

- primitive and fused dispatch counts;
- dispatch-reduction ratio;
- median primitive and fused nanoseconds per call;
- measured host-specific speedup;
- maximum absolute semantic error.

## Current boundary

This backend is still Python. It proves that a promoted operator can move from a symbolic macro to a directly executable fused operation with semantic verification and measurable dispatch reduction.

It does **not** yet provide:

- native machine-code generation;
- LLVM or JIT compilation;
- GPU kernel fusion;
- recursive fusion of arbitrary higher-order operators;
- cross-platform performance guarantees.

Those are subsequent implementation layers. A native backend should preserve the same rule: compile only verified operator expansions, compare against the primitive reference on bounded fixtures, and treat timing as measured telemetry rather than an authority signal.
