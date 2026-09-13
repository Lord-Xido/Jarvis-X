# Inward-Folded Exavoxel Multimodal Swarm

## Purpose

This runtime extends the bounded Jarvis-X multimodal 3D autoencoding loop with a virtualized
`1_000_000 x 1_000_000 x 1_000_000` logical lattice (`10^18` cells).

The implementation does **not** allocate `10^18` Python objects and does **not** claim a measured
`1,000,000x` hardware speedup. Instead it combines:

1. a bounded multimodal encode/decode runtime,
2. a deterministic Monte-Carlo proxy swarm,
3. recursive `10 x 10 x 10` inward spatial aggregation,
4. residual-driven fine-region reactivation,
5. adaptive thresholding for sparse refinement, and
6. explicit mathematical/geometric invariant checks.

## Recursive inward loop

For multimodal input `X_t` and latent state `Z_t`:

```text
X_t
  -> E(X_t)
  -> Z_t
  -> A_10x10x10(Z_t)         level 1: 10^3 contraction
  -> A_10x10x10(C_1)         level 2: another 10^3 contraction
  -> coarse latent C_2
  -> residual R_t = Z_t - U(C_2)
  -> mask H_t = 1(|R_t| > tau_t)
  -> selective refinement F(H_t * R_t)
  -> D(Z'_t)
  -> X_hat_t
  -> reconstruction/cycle error
  -> update codec + threshold
  -> recur
```

One local fold contains:

```text
10 x 10 x 10 = 10^3
```

fine cells. Two self-similar nested levels therefore represent:

```text
(10^3)^2 = 10^6
```

fine logical cells per top-level coarse representative.

Geometrically, two levels reduce each axis by `10^2 = 100`:

```text
1,000,000 / 100 = 10,000
```

so the recursively folded coarse lattice is:

```text
10,000 x 10,000 x 10,000 = 10^12 coarse cells
```

and the exact volume identity is:

```text
10^12 coarse cells x 10^6 fine cells/coarse cell = 10^18 logical cells.
```

The runtime verifies this identity on every cycle.

## Work model

Let `a` be the estimated fraction of the logical field that still requires fine-resolution work and
let `F = 10^6` be the recursive fold volume. The normalized idealized work fraction is:

```text
W(a) / W_dense = 1/F + a * (1 - 1/F)
```

and the modeled work reduction is:

```text
S_model(a) = 1 / (1/F + a * (1 - 1/F)).
```

Therefore:

```text
a = 1   -> S_model = 1x
a -> 0  -> S_model -> 1,000,000x
```

This is an algorithmic work model, not a wall-clock benchmark. It means a perfectly reusable coarse
representation with essentially no residual fine work has a theoretical work-reduction ceiling of
`10^6`; it does not mean current hardware executes the end-to-end application one million times faster.

Actual acceleration depends on memory bandwidth, cache/SRAM locality, synchronization, kernel fusion,
vectorization, GPU occupancy, communication costs, branch sparsity, decoder cost, and the true residual
distribution.

## Operational verification

Every step reports three validation flags:

```text
geometry_verified
work_model_verified
operational_mechanics_verified
```

`geometry_verified` requires exact integer consistency:

```text
total_fold_edge = fold_edge ^ fold_levels
fold_volume     = (fold_edge^3) ^ fold_levels
coarse_axis     = logical_axis / total_fold_edge
coarse_cells * fold_volume = logical_cells
```

`work_model_verified` recomputes the sparse-work equation independently and verifies that the reported
reduction is finite and remains in `[1, fold_volume]`.

`operational_mechanics_verified` additionally requires finite reconstruction, cycle and residual metrics
and confirms that the bounded proxy swarm remains intact.

These checks verify the mechanics of the reference model. They do not substitute for physical timing
benchmarks.

## Virtualization

The logical substrate is fixed at:

```text
axis          = 1_000_000
logical cells = axis^3 = 10^18
```

A bounded number of `ProxyAgent` samples estimate residual activity. Each proxy stands in for a large
logical region. Increasing `--proxy-agents` improves sampling resolution but increases real runtime cost.

## Auto-optimization

The scheduler adapts the residual threshold `tau`:

```text
if reconstruction_mse <= target_mse:
    tau <- tau * (1 + learning_rate)  # prune more fine work
else:
    tau <- tau * (1 - learning_rate)  # refine more of the field
```

The threshold is clamped to configured bounds. Codec parameter updates remain transactional in the
existing `DrMoagiMultimodal3DLoop`: parameter changes are accepted only when the local objective does
not regress.

## Run

Default two-level recursive fold:

```bash
python -m jarvisx.dr_moagi_inward_swarm --cycles 8
```

JSON metrics:

```bash
python -m jarvisx.dr_moagi_inward_swarm \
  --cycles 16 \
  --proxy-agents 4000 \
  --fold-edge 10 \
  --fold-levels 2 \
  --threshold 0.12 \
  --target-mse 0.25 \
  --json
```

The original single-level `1000x` ceiling remains available:

```bash
python -m jarvisx.dr_moagi_inward_swarm --fold-edge 10 --fold-levels 1
```

Disable scheduler auto-tuning for controlled experiments:

```bash
python -m jarvisx.dr_moagi_inward_swarm --cycles 8 --no-auto-optimize
```

## Metrics to benchmark on hardware

Do not use `modeled_work_reduction` as a physical performance result. For a real CPU/GPU implementation,
measure at minimum:

- end-to-end wall-clock latency per cycle,
- cells/second or bytes/second processed physically,
- encoder/decoder kernel time,
- residual-mask construction time,
- active refinement fraction,
- HBM/DRAM bandwidth and cache hit rate,
- synchronization/launch overhead,
- reconstruction MSE and cycle MSE,
- quality-vs-throughput Pareto frontier.

The `1,000,000x` target is physically validated only if measured elapsed time at equivalent output quality
improves by approximately six orders of magnitude against the dense baseline on the same hardware and
workload. Until then it remains a mathematically verified recursive work-reduction ceiling.
