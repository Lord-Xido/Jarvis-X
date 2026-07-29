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

## Hierarchical 3D Fractional Smoothing

`jarvisx.fractional_smoothing_3d` implements a deterministic reference solver
for periodic three-dimensional fractional diffusion, multiscale smoothing, and
equilibrium search. It builds a 2×2×2 hierarchy, applies a separately configured
fractional heat operator at each level, and fuses the coarse equilibrium back
into the fine field.

```python
from jarvisx.fractional_smoothing_3d import (
    FractionalHierarchyConfig,
    Grid3D,
    hierarchical_fractional_smooth,
)

field = Grid3D.impulse((8, 8, 8), (4, 4, 4), amplitude=1.0)
result = hierarchical_fractional_smooth(
    field,
    FractionalHierarchyConfig(
        alphas=(1.0, 0.75, 0.50),
        taus=(0.05, 0.10, 0.20),
        coarse_blends=(0.20, 0.35),
    ),
)

assert abs(result.mass_drift) < 1.0e-9
```

The implementation emits a mechanistic instruction trace covering restriction,
3D fractional heat evolution, prolongation, coarse/fine fusion, and invariant
verification. See
[`docs/HIERARCHICAL_3D_FRACTIONAL_SMOOTHING.md`](docs/HIERARCHICAL_3D_FRACTIONAL_SMOOTHING.md)
for the complete arithmetic derivation and engineering boundaries.
