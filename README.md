# Jarvis-X

[![Jarvis-X CI](https://github.com/Lord-Xido/Jarvis-X/actions/workflows/ci.yml/badge.svg)](https://github.com/Lord-Xido/Jarvis-X/actions/workflows/ci.yml)
[![C++ Runtime](https://github.com/Lord-Xido/Jarvis-X/actions/workflows/cpp-autopoietic-runtime.yml/badge.svg)](https://github.com/Lord-Xido/Jarvis-X/actions/workflows/cpp-autopoietic-runtime.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)](docs/PROJECT_STATUS.md)

**Jarvis-X is a deterministic, auditable bytecode virtual machine and sparse-computing research platform.**

The project investigates how large virtual state spaces, geometric representations, residual memory and bounded adaptation can be implemented as reproducible software without confusing virtual extent with physical allocation or simulation with deployed intelligence.

> **Current status:** alpha research software. The repository contains a stable reference VM foundation, validated sparse and numerical components, a bounded C++ processor laboratory and experimental integration tracks. See [Project Status](docs/PROJECT_STATUS.md) for the authoritative capability boundary.

## Why Jarvis-X exists

Jarvis-X develops one coherent systems thesis:

1. represent large logical spaces sparsely;
2. encode execution in deterministic fixed-width formats;
3. measure prediction or reconstruction error explicitly;
4. retain bounded correction memory;
5. verify proposed state transitions before commit;
6. journal enough information to audit and replay decisions.

The symbolic vocabulary used in the research documents maps to ordinary engineering mechanisms:

| Symbol | Engineering interpretation |
|---|---|
| Ψ / Φ | observed and internal state |
| Θ | model or execution parameters |
| Ω | residual correction memory and journal state |
| Λ | admissibility, policy and coherence constraints |
| Π | projection into a valid state set |
| Ξ | integrated runtime state |

## Capabilities on `main`

| Area | Implemented capability | Maturity |
|---|---|---|
| Bytecode VM | parser, assembler, decoder, registers and minimal 64-bit instruction execution | Alpha |
| Core ISA | `SET`, `ADD`, `SUB`, `HALT` | Alpha |
| Runtime controls | policy check, cycle sandbox, tracing and verifiable ledger integration | Reference foundation |
| Bit-serial mixed-signal control | delta-sigma, XNOR/popcount, 16-bit Omega memory, Theta masking, independent interlocks and PDM logic frames | Reference laboratory |
| C++ processor laboratory | sparse virtual `8192³` lattice, signed 3-bit latent cycle, deterministic bounded genome/schedule search | Reference laboratory |
| Fractional 3D smoothing | periodic spectral fractional diffusion, analytic forcing and multiresolution fusion | Numerical reference |
| Sparse geometry | deterministic inward-folding fractal octree with closed-form invariants | Reference |
| Inward 4D graph ANN | deterministic 1,000-node folded graph autoencoder with exact gradients, guarded pruning and rollback | Reference laboratory |
| Model packaging | Hugging Face-compatible configuration, model and safetensors exporter | Reference |
| Research specifications | reality-grounded observer dynamics, spatial bytecode and bounded optimization documents | Proposed / reference |

Experimental engines remain in draft pull requests until their tests, interfaces and capability claims are reconciled with the canonical core.

## Quick start

### Install the Python package for development

```bash
git clone https://github.com/Lord-Xido/Jarvis-X.git
cd Jarvis-X
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
```

### Execute a bytecode program

```python
from jarvisx.assembler import Assembler
from jarvisx.core import CodexVM
from jarvisx.parser import Parser

source = """
SET Ψ 10
SET Φ 20
ADD A Ψ Φ
HALT
"""

program = Assembler().assemble(Parser().parse(source))
vm = CodexVM()
vm.load(program)
state = vm.run()

assert state["A"] == 30
assert vm.ledger.verify()
```

Persistent journaling is explicit:

```python
vm = CodexVM(ledger_path="state/omega-ledger.json")
```

Adaptive reflex correction is also explicit and disabled by default:

```python
vm = CodexVM(enable_reflex=True)
```

### Run fractional 3D smoothing

```python
from jarvisx.fractional_smoothing_3d import (
    FractionalHierarchyConfig,
    Grid3D,
    hierarchical_fractional_smooth,
)

field = Grid3D.impulse((4, 4, 4), (1, 1, 1), amplitude=8.0)
config = FractionalHierarchyConfig(
    alphas=(1.0, 0.65),
    taus=(0.08, 0.20),
    coarse_blends=(0.25,),
)
result = hierarchical_fractional_smooth(field, config)

assert abs(result.mass_drift) < 1.0e-9
assert result.field.variance < field.variance
```

The solver uses a dependency-free separable direct DFT for small correctness fixtures. See [Hierarchical 3D Fractional Smoothing](docs/HIERARCHICAL_3D_FRACTIONAL_SMOOTHING.md) for the equations, complexity and production boundary.

### Run the 10x10x10 inward 4D ANN reference

```bash
python examples/inward4d_ann_demo.py --epochs 25
```

The reference executes a same-width graph autoencoder over exactly 1,000 nodes
and 3,000 fully wrapped undirected synapses. It reports the complete
self-description objective and commits an update only when the candidate does
not regress. See the
[end-to-end arithmetic](docs/DR_MOAGI_10X10X10_INWARD_4D_ANN.md) for the fold,
forward pass, analytic gradient, pruning, and capability boundary.

### Build the C++ processor laboratory

```bash
cmake -S cpp_runtime -B build/cpp-runtime -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp-runtime --config Release --parallel
ctest --test-dir build/cpp-runtime -C Release --output-on-failure
```

Run a bounded inward experiment:

```bash
./build/cpp-runtime/jarvisx-runtime \
  --generations 8 \
  --population 6
```

See [`cpp_runtime/README.md`](cpp_runtime/README.md) for its state artifacts, determinism contract, sanitizer build and capability limits.

## Architecture

```text
Source assembly
      │
      ▼
Parser → Assembler → 64-bit bytecode
                         │
                         ▼
             ┌─────────────────────┐
             │      CodexVM        │
             │ decode → authorize  │
             │ execute → trace     │
             │ journal → constrain │
             └─────────────────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
      authoritative state    isolated research layers
      registers / memory     C++ / numerical / visual
```

The canonical design rules are documented in [Architecture](docs/ARCHITECTURE.md).

## Sparse fractal octree

```python
from jarvisx.fractal_octree import build_fractal_octree

root = build_fractal_octree(size=1.0, max_depth=3)
metrics = root.metrics()

assert metrics.active_nodes == 85
assert metrics.active_leaves == 64
assert metrics.retained_volume == 0.125
```

At depth `D`:

- active leaves: `4 ** D`
- active nodes: `(4 ** (D + 1) - 1) // 3`
- retained volume: `2 ** (-D)` for a unit cube
- similarity dimension: `2`

## Active integration tracks

| Track | Purpose | Status |
|---|---|---|
| Backlog consolidation | select canonical implementations and close superseded research branches | Issue #48 |
| Repository protection | required checks, secret scanning and private vulnerability reporting | Issue #49 |
| Public profile | account-level profile README and pinned-project cleanup | Issue #50 |
| Browser engines | bounded interactive 3D visual-computing demonstrations | Separate repository |

Draft status is intentional: experimental subsystems are not represented as canonical until CI, review and integration boundaries are satisfied.

## Repository structure

```text
src/jarvisx/       canonical Python package and numerical references
tests/             regression and invariant tests
docs/              specifications and architecture records
scripts/           packaging and export utilities
examples/          runnable demonstrations
cuda/              accelerator reference work
cpp_runtime/       bounded C++ processor laboratory
.github/           CI, templates and repository automation
```

[Project Status](docs/PROJECT_STATUS.md) identifies what is implemented, experimental or proposed.

## Engineering standards

Every canonical subsystem should provide:

- deterministic behavior under a documented seed or input;
- explicit state and dimensional contracts;
- bounded memory and execution behavior;
- validation of malformed or adversarial inputs;
- reproducible tests and examples;
- honest implemented-versus-proposed boundaries;
- transaction, rollback or failure semantics where state is mutated;
- no performance or intelligence claim without measurement.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Project status](docs/PROJECT_STATUS.md)
- [Inward 3D kinetic end-to-end specification](docs/INWARD_3D_KINETIC_END_TO_END.md)
- [10x10x10 inward 4D graph ANN](docs/DR_MOAGI_10X10X10_INWARD_4D_ANN.md)
- [Dr. Moagi 4D quantum-inspired autoencoding equation](docs/DR_MOAGI_4D_QUANTUM_INSPIRED_AUTOENCODING.md)
- [Hierarchical 3D fractional smoothing](docs/HIERARCHICAL_3D_FRACTIONAL_SMOOTHING.md)
- [DM-vΩΞ⁺ bit-serial mixed-signal control](docs/DM_VOMEGAXI_MIXED_SIGNAL_CONTROL.md)
- [Roadmap](ROADMAP.md)
- [Governance](GOVERNANCE.md)
- [Security policy](SECURITY.md)
- [Contributing](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)

## Contributing

Jarvis-X welcomes focused improvements in VM correctness, bytecode formats, sparse spatial computation, deterministic testing, performance measurement and documentation. Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

Large architectural proposals should begin as an issue or specification. Production claims must include reproducible evidence.

## Research boundary

Jarvis-X is an experimental software and mathematical research project. It does not claim consciousness, unrestricted autonomous self-modification, lossless compression of arbitrary high-dimensional inputs into smaller states, or production safety merely because a policy layer is present.

Virtual address-space size is not resident memory. A deterministic simulation is not evidence of general intelligence. A cryptographic digest provides integrity, not reversibility. The C++ processor mutates bounded parameters and schedules; it does not rewrite arbitrary native code. The fractional solver is a small-grid CPU reference, not a calibrated physical model or production FFT implementation.

## Citation

Academic and technical users can cite the project using [`CITATION.cff`](CITATION.cff).

## License

Jarvis-X is released under the [MIT License](LICENSE).
