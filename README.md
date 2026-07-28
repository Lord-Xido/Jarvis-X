# Jarvis-X

Jarvis-X is a deterministic, auditable virtual machine with a reflex
control layer and policy gate.

## Install
```bash
git clone https://github.com/Lord-Xido/Jarvis-X.git
cd Jarvis-X
pip install -r requirements.txt
pip install .
```

## Sparse Fractal Octree

The concrete recursive spatial substrate is available as
`jarvisx.fractal_octree`. It materializes an eight-location octree while
retaining four active children per active parent under the inward-folding
rule `dx + dy + dz < 2`.

```python
from jarvisx.fractal_octree import build_fractal_octree

root = build_fractal_octree(size=1.0, max_depth=3)
metrics = root.metrics()

assert metrics.active_nodes == 85
assert metrics.active_leaves == 64
assert metrics.retained_volume == 0.125
```

At depth `D`, the deterministic invariants are:

- active leaves: `4 ** D`
- active nodes: `(4 ** (D + 1) - 1) // 3`
- retained volume: `2 ** (-D)` for a unit cube
- similarity dimension: `2`

## Self-Evolutionary Meta-Volume

`jarvisx.meta_volume` implements a deterministic architecture-governor for a
future adaptive neural renderer. It evolves bounded per-region depth, width,
ray-sampling budgets, and pruning masks from scene error and hardware telemetry.
Every accepted structural update emits auditable instructions and is committed
through a SHA-256 journal; invalid or regressive candidates roll back.

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
result = engine.evolve(
    FrameSignals(
        error=(0.9, 0.1, 0.8, 0.05),
        edge_density=(0.8, 0.1, 0.7, 0.0),
        occupancy=(0.9, 0.0, 0.8, 0.0),
    ),
    HardwareTelemetry(frame_ms=12.0, flops=1.0e11, memory_mb=2048.0),
)
assert result.committed
```

The module is a tested control-plane prototype, not a claim of measured
superiority over NeRF, Instant-NGP, or 3D Gaussian Splatting. See
[`docs/SELF_EVOLUTIONARY_META_VOLUME.md`](docs/SELF_EVOLUTIONARY_META_VOLUME.md)
for the equations, transaction protocol, structural bytecode, and benchmark
requirements.
