# Dr. Moagi Self-Evolutionary Meta-Volume

## Status

This document defines an executable **architecture-governor prototype** for a future neural renderer. It does not claim that Jarvis-X currently outperforms NeRF, Instant-NGP, or 3D Gaussian Splatting, and it does not claim 4K60 performance without measured benchmarks.

The implemented contribution is the control plane that can change bounded structural parameters in response to scene error and hardware telemetry while preserving deterministic replay and transactional rollback.

## Twelve-layer logical volume

The renderer contract is represented as twelve logical layers:

| Layer | Name | Logical content |
|---:|---|---|
| 0 | Input pixel buffer | Incoming RGB image |
| 1 | Encoded latent volume | Compact scene representation |
| 2 | Decoded pixel buffer | Reconstructed image |
| 3 | Error volume | Per-pixel or regional reconstruction error |
| 4 | Gradient volume | Weight and latent gradients |
| 5 | Bytecode and weights | Executable renderer parameters |
| 6 | Shader registers | Hardware execution state |
| 7 | Rendered self-image | Visualised internal state |
| 8 | Architecture DNA | Per-region depth, width, and kernel selection |
| 9 | Step allocation map | Per-region ray-marching sample budget |
| 10 | Sparsity mask | Active parameter or feature subset |
| 11 | Meta-gradient accumulator | Structural update targets |

Layers 0-7 are currently a logical interface. Layers 8-11 are operationalised by `jarvisx.meta_volume`.

## Two-loop model

The intended optimisation is bilevel:

\[
\theta^*(a)=\arg\min_\theta \mathcal L_{\mathrm{render}}(\theta,a)
\]

\[
a_{t+1}=\Pi_{\mathcal A}\left[a_t-\eta_a\nabla_a\left(
\mathcal L_{\mathrm{render}}
+\lambda_C\mathcal C_{\mathrm{compute}}
+\lambda_M\mathcal M_{\mathrm{memory}}
\right)\right]
\]

where `a` contains depth, width, sample count, and pruning variables. The current prototype implements a deterministic projected surrogate update rather than full differentiable NAS.

## Operational state

\[
S_t=(A_t,\;P_t,\;Q_t,\;M_t,\;G_t,\;J_t)
\]

- `A_t`: architecture DNA;
- `P_t`: step allocation map;
- `Q_t`: continuous pruning scores;
- `M_t`: binary execution mask;
- `G_t`: meta-gradient targets;
- `J_t`: SHA-256 structural journal hash.

The controller receives:

\[
X_t=(e_t,\;g_t,\;o_t,\;h_t)
\]

where `e` is normalised error, `g` is edge density, `o` is occupancy, and `h` is hardware telemetry.

## Bounded structural instructions

A committed meta-step emits auditable instructions:

```text
SET_DEPTH region depth
SET_WIDTH region width
SET_STEPS region samples
SET_MASK parameter enabled
```

The implemented constraints are:

```text
4 <= depth <= 12
32 <= width <= 256, aligned to 16
4 <= samples <= 128
kernel in {1, 3, 5}
active mask ratio >= configured floor
```

The current prototype changes depth, width, steps, and pruning state. Kernel selection is represented and verified but remains fixed at `3` until a kernel-cost model is added.

## Transaction protocol

Each evolution cycle executes:

```text
observe scene signals and hardware telemetry
-> evaluate current architecture
-> compute structural targets
-> project into legal architecture bounds
-> evaluate candidate objective
-> verify dimensions, bounds, and mask floor
-> emit structural instructions
-> commit and seal journal, or roll back atomically
```

The candidate commits only when its surrogate objective does not regress:

\[
J(a') \le J(a)+\varepsilon.
\]

A failed candidate leaves the architecture, cycle, and prior journal hash unchanged.

## Hardware-aware objective

The prototype uses:

\[
J=
\mathcal L_{\mathrm{render}}
+\lambda_C\widehat{\mathcal C}
+\lambda_M\widehat{\mathcal M}.
\]

`render_loss` is a normalised regional proxy. Compute cost is estimated from depth, width, sample count, mask activity, frame time, FLOPs, and SM cycles. Memory cost uses active ratio, width, and measured VRAM usage.

These proxies are deliberately separated from real benchmark claims. A CUDA renderer must replace them with measured kernel timings and allocator telemetry before performance comparisons are valid.

## Pareto analysis

`pareto_front()` computes non-dominated candidates under:

- maximise PSNR;
- minimise FLOPs;
- minimise memory.

A raw score such as `PSNR / (FLOPs * memory)` is not used as the sole decision rule because it is unit-dependent and can hide trade-offs. Scalarisation remains available through the weighted objective.

## Example

```python
from jarvisx.meta_volume import (
    FrameSignals,
    HardwareTelemetry,
    MetaVolumeConfig,
    SelfEvolutionaryMetaVolume,
)

engine = SelfEvolutionaryMetaVolume(
    MetaVolumeConfig(region_count=4, parameter_count=8)
)

signals = FrameSignals(
    error=(0.9, 0.1, 0.8, 0.05),
    edge_density=(0.8, 0.1, 0.7, 0.0),
    occupancy=(0.9, 0.0, 0.8, 0.0),
)
telemetry = HardwareTelemetry(
    frame_ms=12.0,
    flops=1.0e11,
    memory_mb=2048.0,
)

result = engine.evolve(signals, telemetry)
assert result.committed
assert engine.state.architecture.depth[0] > engine.state.architecture.depth[1]
```

## Research path to a real renderer

The next implementation gates are:

1. connect the controller to a concrete NeRF, hash-grid, or Gaussian renderer;
2. replace regional quality proxies with MSE, SSIM, LPIPS, and measured PSNR;
3. use differentiable gates or straight-through estimators for depth and pruning;
4. profile actual CUDA kernels and memory allocation;
5. train on public datasets with fixed evaluation protocols;
6. compare quality, training time, render latency, energy, and VRAM against reproduced baselines;
7. publish confidence intervals and ablation studies.

“Beyond SOTA” remains a falsifiable research hypothesis until those measurements exist.
