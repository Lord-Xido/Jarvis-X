# DM80K Fast Path — Benchmark-Gated 1000x Program

**Status:** optimization program / portable fast-path reference  
**Base runtime:** `DrMoagi-80K3-MP4`  
**Authority:** ADR-016 + ADR-017

## Objective

The optimization target is an end-to-end speedup of up to 1000x relative to a
declared baseline workload while preserving the required CTR and transaction
semantics. The number is a target, not an architectural consequence.

The correct acceptance criterion is

\[
S_{e2e} = \frac{T_{baseline}}{T_{candidate}} \ge 1000
\]

on the same input population, hardware, output requirements, warm-up policy,
quality thresholds and verification policy.

## Cost law

A useful first-order decomposition is

\[
T_{total}
=
T_{select}
+
T_{io}
+
T_{core}
+
T_{ctr}
+
T_{collapse}.
\]

For a sparse multi-resolution implementation the core is better represented as

\[
E[T_{core}]
\approx
N_{cand}\,p_{sel}\,p_{miss}
\sum_k
p_{ref,k}\,R_k\,
\max\left(
\frac{F_k}{P_{eff}},
\frac{B_k}{BW_{eff}}
\right).
\]

The terms are:

- `N_cand`: candidate bricks considered;
- `p_sel`: selection survival probability;
- `p_miss`: temporal-cache miss probability;
- `p_ref,k`: probability that a brick reaches LOD level `k`;
- `R_k`: adaptive fixed-point work at that level;
- `F_k`: arithmetic work;
- `B_k`: memory traffic;
- `P_eff`: achieved compute throughput;
- `BW_eff`: achieved memory bandwidth.

This roofline-style law prevents double-counting gains. Kernel fusion,
precision reduction and cache reuse can all reduce the same memory or compute
term, so their individual headline ratios are not assumed independent.

## Implemented now

The portable fast path implements two directly measurable levers:

1. **priority sparsity** — rank a deterministic bounded candidate set and
   execute only the configured fraction;
2. **temporal coherence cache** — reuse recently verified bricks when the
   declared temporal phase/error bounds permit it.

The base engine now exposes `step_at(t, voxel)` so baseline and optimized
paths can be benchmarked over exactly the same candidate coordinates.

The portable benchmark reports:

```text
selection ratio
cache hit rate
executed work rate
baseline wall time
fast-path wall time
measured speedup
```

Run the A/B benchmark:

```bash
./build/cpp-runtime/DrMoagi-80K3-Fast-Bench \
  --ticks 4 \
  --candidates 64 \
  --select-ratio 0.10 \
  --active-bricks 64
```

Run the operational DM80K runtime with the portable scheduler enabled:

```bash
./build/cpp-runtime/DrMoagi-80K3-MP4 \
  --cycles 60 \
  --fast \
  --candidates 64 \
  --select-ratio 0.10 \
  --active-bricks 64
```

## GPU phase

The following proposed levers belong in a dedicated CUDA backend and are not
claimed by the portable C++ reference:

- fused AE/AD kernel with one brick load/store boundary;
- FP8/FP16 tensor-core contractions;
- shared-memory or warp-shuffle spatial exchange;
- TMA / asynchronous brick prefetch and store;
- warp-specialized producer/consumer stages;
- low-rank or implicit latent representation;
- differential render and hardware-codec integration.

A CUDA backend should emit receipts for achieved FLOP/s, achieved bandwidth,
HBM bytes, tensor-core utilization, fixed-point iterations, selected/refined
brick counts, cache hits and reconstruction/CTR quality.

## Multiplicative budget rule

A paper budget such as

\[
10\times4\times3\times4\times3\times2
\]

is an upper-bound planning device only. The deployable speedup is

\[
S_{e2e}
=
\frac{T_{baseline}}
{
T_{select}
+
T_{fast\ core}
+
T_{ctr}
+
T_{render}
+
T_{codec}
},
\]

and must be measured. A 1000x result is accepted only when the end-to-end
benchmark demonstrates it under the declared quality constraints.

## Invariants

\[
80000^3 = 512,000,000,000,000
\]

remains a logical address extent, not resident memory.

The optimization layer must also preserve

\[
\text{candidate}
\to
\text{verify}
\to
\text{commit OR rollback}.
\]

Skipping work is legal only through an explicit selection/cache/LOD policy
whose effect is observable in receipts. Optimization does not bypass CTR or
the ADR-016 authority boundary.
