# Jarvis-X

[![Jarvis-X CI](https://github.com/Lord-Xido/Jarvis-X/actions/workflows/ci.yml/badge.svg)](https://github.com/Lord-Xido/Jarvis-X/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)](docs/PROJECT_STATUS.md)

**Jarvis-X is a deterministic, auditable bytecode virtual machine and sparse-computing research platform.**

The project investigates how large virtual state spaces, geometric representations, residual memory and bounded adaptation can be implemented as reproducible software without confusing virtual extent with physical allocation or simulation with deployed intelligence.

> **Current status:** alpha research software. The repository contains a small stable VM core, validated reference components and multiple experimental integration tracks. See [Project Status](docs/PROJECT_STATUS.md) for the authoritative capability boundary.

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

## Stable capabilities on `main`

| Area | Implemented capability | Maturity |
|---|---|---|
| Bytecode VM | parser, assembler, decoder, registers and minimal 64-bit instruction execution | Alpha |
| Core ISA | `SET`, `ADD`, `SUB`, `HALT` | Alpha |
| Runtime controls | policy check, cycle sandbox, tracing and ledger integration | Alpha |
| Sparse geometry | deterministic inward-folding fractal octree with closed-form invariants | Reference |
| Model packaging | Hugging Face-compatible configuration, model and safetensors exporter | Reference |
| Research specifications | reality-grounded observer dynamics, spatial bytecode and bounded optimization documents | Proposed / reference |

Experimental engines remain in draft pull requests until their tests, interfaces and capability claims are reconciled with the canonical core.

## Quick start

### Install for development

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
      authoritative state    optional research layers
      registers / memory     spatial / adaptive / visual
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
| C++ inward runtime | sparse `8192³` virtual processor with bounded genome and bytecode search | Draft PR #45 |
| Fractional smoothing | deterministic hierarchical 3D fractional-diffusion reference solver | Draft PR #46 |
| Platform foundation | canonical VM repair, CI enforcement and contributor-ready governance | Draft PR #47 |
| Browser engines | bounded interactive 3D visual-computing demonstrations | Separate repository |

Draft status is intentional: experimental subsystems are not represented as canonical until CI, review and integration boundaries are satisfied.

## Repository structure

```text
src/jarvisx/       canonical Python package
tests/             regression and invariant tests
docs/              specifications and architecture records
scripts/           packaging and export utilities
examples/          runnable demonstrations
cuda/              accelerator reference work
cpp_runtime/       C++ runtime integration track
.github/           CI, templates and repository automation
```

Some paths exist only on active branches. [Project Status](docs/PROJECT_STATUS.md) identifies what is present on the default branch.

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

Virtual address-space size is not resident memory. A deterministic simulation is not evidence of general intelligence. A cryptographic digest provides integrity, not reversibility.

## Citation

Academic and technical users can cite the project using [`CITATION.cff`](CITATION.cff).

## License

Jarvis-X is released under the [MIT License](LICENSE).
