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

## C++ Inward Autopoietic Runtime

The `cpp_runtime/` subsystem implements a dependency-free C++17 processor with:

- a sparse virtual `8192 × 8192 × 8192` lattice;
- 3-bit auto-encoding and decoding;
- deterministic 64-bit bytecode synthesis;
- inward ingestion of its own executable image;
- candidate genome and bytecode-schedule mutation;
- isolated evaluation, coherence gating, commit, and rollback;
- persistent checkpoints, binary ROM output, and CSV evolution journals.

```bash
cmake -S cpp_runtime -B build/cpp-runtime -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp-runtime --parallel
ctest --test-dir build/cpp-runtime --output-on-failure

./build/cpp-runtime/jarvisx-runtime \
  --generations 8 \
  --population 6
```

The self-evolution mechanism is bounded and auditable: it optimizes runtime parameters and bytecode schedules rather than rewriting arbitrary native machine code.
