# Inward-Folded Exavoxel Multimodal Swarm

## Purpose

This runtime extends the bounded Jarvis-X multimodal 3D autoencoding loop with a virtualized
`1_000_000 x 1_000_000 x 1_000_000` logical lattice (`10^18` cells).

The implementation does **not** allocate `10^18` Python objects and does **not** claim a measured
1000x hardware speedup. Instead it combines:

1. a bounded multimodal encode/decode runtime,
2. a deterministic Monte-Carlo proxy swarm,
3. `10 x 10 x 10` inward spatial aggregation,
4. residual-driven fine-region reactivation, and
5. an adaptive threshold that trades refinement density against reconstruction error.

## Inward loop

For multimodal input `X_t` and latent state `Z_t`:

```text
X_t
  -> E(X_t)
  -> Z_t
  -> A_10x10x10(Z_t)         coarse inward fold
  -> residual R_t
  -> mask H_t = 1(|R_t| > tau_t)
  -> selective refinement F(H_t * R_t)
  -> D(Z'_t)
  -> X_hat_t
  -> reconstruction/cycle error
  -> update codec + threshold
  -> recur
```

A single fold block contains

```text
10 x 10 x 10 = 1000
```

fine cells.

## Work model

Let `a` be the estimated fraction of the logical field that still requires fine-resolution work and
let `F = 1000` be the fold volume. The normalized idealized work fraction is

```text
W(a) / W_dense = 1/F + a * (1 - 1/F)
```

and the modeled work reduction is

```text
S_model(a) = 1 / (1/F + a * (1 - 1/F)).
```

Therefore:

```text
a = 1   -> S_model = 1x
a -> 0  -> S_model -> 1000x
```

This is an algorithmic work model, not a wall-clock benchmark. Actual acceleration depends on
memory bandwidth, cache/SRAM locality, synchronization, kernel fusion, vectorization, GPU occupancy,
communication costs, and the true distribution of residuals.

## Virtualization

The logical substrate is fixed at:

```text
axis          = 1_000_000
logical cells = axis^3 = 10^18
```

A bounded number of `ProxyAgent` samples estimate residual activity. Each proxy stands in for a
large logical region. Increasing `--proxy-agents` improves the sampling resolution but increases real
runtime cost.

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

From the repository root:

```bash
python -m jarvisx.dr_moagi_inward_swarm --cycles 8
```

JSON metrics:

```bash
python -m jarvisx.dr_moagi_inward_swarm \
  --cycles 16 \
  --proxy-agents 4000 \
  --fold-edge 10 \
  --threshold 0.12 \
  --target-mse 0.25 \
  --json
```

Disable scheduler auto-tuning for controlled experiments:

```bash
python -m jarvisx.dr_moagi_inward_swarm --cycles 8 --no-auto-optimize
```

## Metrics to benchmark on hardware

Do not use `modeled_work_reduction` as a performance result. For a real CPU/GPU implementation,
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

The 1000x target is validated only if measured elapsed time at equivalent output quality improves by
approximately three orders of magnitude against the dense baseline on the same hardware and workload.
