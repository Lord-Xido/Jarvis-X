# Moagi OmegaFold — Bounded Meta-Optimizer Specification

**Status:** experimental specification  
**Tracking:** #62  
**Canonical authority:** none until review and merge

## 1. Purpose

Moagi OmegaFold translates the symbolic idea of "collapsing many iterations into an axiomatic result" into a finite software contract. The subsystem may replace an iterative computation with a closed form, fixed point, memoized result or reduced representation only when the replacement is independently verified against an explicit residual.

The implementation does not claim literal `O(0)` computation, zero elapsed time, tachyonic hardware, causal reversal, perfect loss or `10^3000x` acceleration. Those expressions remain narrative shorthand for eliminating avoidable work.

## 2. Operational equation

For problem state `x`, transition `T`, residual `r`, admissibility set `Λ`, tolerance `ε` and iteration bound `N`:

```text
x_0 = Π_Λ(initial_state)
x_(k+1) = Π_Λ(T(x_k))
stop when r(x_k) <= ε, a prior state repeats, or k = N
```

A closed-form candidate `C(x_0)` may be evaluated before iteration, but it is accepted only when:

```text
shape(C(x_0)) = shape(x_0)
all_finite(C(x_0))
r(C(x_0)) <= ε
```

The result is therefore a state plus evidence, not an unverified assertion:

```text
OmegaFold(problem, config) -> (terminal_state, residual_trace, certificate)
```

## 3. Symbolic-to-engineering map

| Symbolic term | Implemented interpretation |
|---|---|
| `CollapseTime` | eliminate iterations through a verified closed form or fixed point |
| `HilbertFold` | future spectral, low-rank or Krylov reduction with error bounds |
| `AxiomaticResolve` | return a candidate state plus independently measured residual |
| `Eternal Now` | immutable result keyed by canonical input and configuration hashes |
| topological pruning | future graph equivalence, dead-code and common-subexpression elimination |
| `PerfectLoss` | `residual <= tolerance`; never an unconditional zero |
| hash seal | canonical JSON serialization and SHA-256 digest |

## 4. Data contracts

### `FoldConfig`

- `max_iterations >= 0`;
- `tolerance >= 0`;
- `quantization_digits >= 0`.

### `FoldProblem`

- non-empty name;
- non-empty finite initial state;
- deterministic transition for canonical inputs;
- non-negative finite residual;
- optional closed form.

### `FoldCertificate`

The certificate binds:

- problem name;
- resolution method;
- iteration count;
- convergence flag;
- measured residual;
- terminal reason;
- terminal-state digest;
- configuration digest.

## 5. Termination states

- `residual_satisfied` — measured residual is within tolerance;
- `cycle_detected` — a canonical state repeated;
- `iteration_limit` — the configured finite bound was exhausted;
- validation exception — malformed dimensions or non-finite values fail closed.

## 6. Determinism boundary

Determinism requires deterministic user-supplied transition, residual and closed-form functions. OmegaFold canonicalizes floating values by decimal rounding before comparison and hashing. This creates reproducible reference behavior but does not establish bit-identical floating-point execution across every hardware platform.

## 7. Complexity boundary

For state width `d`, iteration bound `N`, transition cost `C_T` and residual cost `C_r`, the reference path is bounded by:

```text
O(N * (C_T + C_r + d)) time
O(N * d) worst-case cycle-memory space
```

A verified closed form may reduce the iteration count to one, but invocation, validation, hashing and output remain finite operations. No `O(0)` class is claimed.

## 8. Benchmark contract

Every performance statement must report:

- exact workload and input dimensions;
- baseline algorithm and implementation;
- tolerance and numeric representation;
- warm-up and repeat counts;
- hardware, operating system and runtime versions;
- median and tail latency;
- verification result;
- raw data or reproducible script.

A speedup is valid only for the measured workload and environment.

## 9. Security and authority

OmegaFold operates on Python callables and is not a security sandbox. Untrusted callables must not be executed in-process. The subsystem cannot mutate canonical VM state unless a future integration layer performs explicit policy checks, shadow evaluation, transaction validation and commit.

## 10. Promotion path

1. accept the specification and ADR;
2. validate the CPU reference resolver;
3. add property-based and adversarial tests;
4. add memoization with bounded storage and collision tests;
5. add graph and spectral reductions with independent numerical references;
6. define VM integration and rollback semantics;
7. consider CUDA only after a workload justifies it;
8. consider RTL only after widths, latency, resource use and verification are specified.
