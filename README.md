# Jarvis-X

[![Jarvis-X CI](https://github.com/Lord-Xido/Jarvis-X/actions/workflows/ci.yml/badge.svg)](https://github.com/Lord-Xido/Jarvis-X/actions/workflows/ci.yml)
[![C++ Runtime](https://github.com/Lord-Xido/Jarvis-X/actions/workflows/cpp-autopoietic-runtime.yml/badge.svg)](https://github.com/Lord-Xido/Jarvis-X/actions/workflows/cpp-autopoietic-runtime.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)](docs/PROJECT_STATUS.md)

**Jarvis-X is a deterministic, auditable bytecode virtual machine and sparse-computing research platform.**

The project investigates how large virtual state spaces, geometric representations, residual memory and bounded adaptation can be implemented as reproducible software without confusing virtual extent with physical allocation or simulation with deployed intelligence.

> **Current status:** alpha research software. The repository contains a stable reference VM foundation, validated sparse and numerical components, bounded C++ research runtimes, and canonical architectural specifications whose full cross-backend migration is still in progress. See [Project Status](docs/PROJECT_STATUS.md) for the authoritative capability boundary.

## Canonical Dr Moagi attribution and provenance

The named **Dr Moagi 3D Ephemeral-Notion Intelligence Framework v1.1** is attributed in this repository to **Matladi Maxwell Moagi (Lord-Xido)**. Its authoritative provenance, defining equations, reality-coupled verification rule and attribution boundary are recorded in [`docs/attribution/DR_MOAGI_EPHEMERAL_NOTION_FRAMEWORK.md`](docs/attribution/DR_MOAGI_EPHEMERAL_NOTION_FRAMEWORK.md).

Repository-wide inheritance of that attribution across explicitly designated Dr Moagi runtimes, engines, simulations, specifications and derivatives is governed by [ADR-015](docs/adr/0015-dr-moagi-ephemeral-notion-attribution-provenance.md) and summarized in the [Permeation Manifest](docs/attribution/PERMEATION_MANIFEST.md).

Structural closure is governed by [ADR-016](docs/adr/0016-canonical-typed-state-transaction-and-geometry-profiles.md): one typed system state, one candidate-first transaction contract, explicit verification gates, and interchangeable geometry/backend profiles. The canonical end-to-end Dr Moagi auto-encoding/decoding systems law is defined by [ADR-017](docs/adr/0017-dr-moagi-operational-autoencoding-equation.md) and the [Operational Auto-Encoding/Decoding Equation](docs/DR_MOAGI_OPERATIONAL_AUTOENCODING_EQUATION.md).

The framework is designed to pursue capability beyond contemporary SOTA through inward 3D latent coordination, reality coupling, continuous correction and fixed-point verification. That phrase denotes a **design ambition**, not an unverified empirical, legal or patent-novelty claim; performance and scientific claims remain evidence-gated.

## Why Jarvis-X exists

Jarvis-X develops one coherent systems thesis:

1. represent large logical spaces sparsely;
2. encode execution in deterministic fixed-width formats;
3. compact representations hierarchically while accounting for residual/side information;
4. measure prediction or reconstruction error explicitly;
5. retain bounded correction memory;
6. verify proposed state transitions before commit;
7. journal enough information to audit and replay decisions.

The symbolic vocabulary used in the research documents maps to ordinary engineering mechanisms:

| Symbol | Engineering interpretation |
|---|---|
| `X` / `X_hat` | authoritative reference/domain state and reconstruction/prediction |
| `Z` / `Z*` | encoded/latent state and bounded fixed-point result |
| `Omega_mem` / Ω | adaptive residual/temporal memory |
| `Theta_model` / Θ | model parameters |
| `Pi_runtime` | bounded runtime/execution policy |
| `Pi_Lambda` | admissibility/projection boundary for authoritative promotion |
| `A_arch` | slower architecture/orchestration policy |
| `R_CTR` | contrast/evidence/reckoning state |
| `audit` | journal, lineage and integrity state |

Implementation-facing APIs follow ADR-016's typed names when ambiguity matters; symbolic shorthand remains valid in mathematical documents.

## Canonical Dr Moagi operational equation

ADR-017 defines candidate generation as

\[
S^{\rm cand}_{t+1}
=
\left[
\mathcal U_{\Omega,\Theta,\Pi_{\rm run}}
\circ
\mathcal R_{\rm CTR}
\circ
\mathcal D_{\mathcal R}
\circ
\operatorname{Fix}_{F_\Theta}
\circ
\Phi_{\rm fusion}
\circ
\mathcal C_{\exp}
\circ
\mathcal E
\right](S_t,U_{t+1}).
\]

The resulting candidate remains provisional until the ADR-016 promotion law accepts it:

\[
S_{t+1}=V_t\,\Pi_\Lambda(S^{\rm cand}_{t+1})+(1-V_t)S_t.
\]

Operationally:

```text
typed input/event
 -> encode
 -> sparse exponential/multiresolution compaction
 -> preserve residual hierarchy
 -> fusion
 -> bounded fixed point
 -> selective residual-aware decode
 -> compare / CTR evidence
 -> stage Omega_mem / Theta_model / Pi_runtime
 -> verify / Pi_Lambda
 -> COMMIT or ROLLBACK
 -> audit / recur
```

This is a canonical integration contract. It does not imply that every current backend already implements every stage or receipt.

## Capabilities on `main`

| Area | Implemented capability | Maturity |
|---|---|---|
| Bytecode VM | parser, assembler, decoder, registers and minimal 64-bit instruction execution | Alpha |
| Core ISA | `SET`, `ADD`, `SUB`, `HALT` | Alpha |
| Runtime controls | policy check, cycle sandbox, tracing and verifiable ledger integration | Reference foundation |
| Typed-state/transaction closure | ADR-016 state namespaces, geometry profiles and candidate-first promotion contract | Specification |
| Operational AE/AD system equation | ADR-017 encode → compact → fixed point → decode → evidence → staged adaptation → verify/commit law | Specification |
| C++ processor laboratory | sparse virtual `8192³` lattice, signed 3-bit latent cycle, deterministic bounded genome/schedule search | Reference laboratory |
| Volumetric ROM ANN | `2^60` logical 3D voxel addresses, bounded `32^3` demand tiles, `32^3 → 16^3 → 8^3 → 4^3 → 2^3 → 1` inward contraction, recursive Ω/Θ/runtime-policy feedback | Reference laboratory |
| Fractional 3D smoothing | periodic spectral fractional diffusion, analytic forcing and multiresolution fusion | Numerical reference |
| Sparse geometry | deterministic inward-folding fractal octree with closed-form invariants | Reference |
| Inward 4D graph ANN | deterministic 1,000-node folded graph autoencoder with exact gradients, guarded pruning and rollback | Reference laboratory |
| Model packaging | Hugging Face-compatible configuration, model and safetensors exporter | Reference |
| Research specifications | reality-grounded observer dynamics, spatial bytecode and bounded optimization documents | Proposed / reference |

Experimental engines remain non-authoritative until their tests, interfaces and capability claims are reconciled with the canonical core and transaction contract.

## Quick start

### Run the normalized Dr Moagi Q16.16 cube

```bash
PYTHONPATH=src python examples/dr_moagi_q16_cube.py --side 1000000 --steps 3
```

This bounded sparse reference executes the `Psi -> Phi -> Lambda -> Theta`
codec and feeds decoded cells into the next cycle, with exact error metrics
and SHA3 receipts. The `10^18`-cell cube is a logical address space; the JSON
reports actual processed cells. See the [arithmetic and provenance contract](docs/DR_MOAGI_Q16_FIELD_SUBSTRATE.md).

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

The reference executes a same-width graph autoencoder over exactly 1,000 nodes and 3,000 fully wrapped undirected synapses. It reports the complete self-description objective and commits an update only when the candidate does not regress. See the [end-to-end arithmetic](docs/DR_MOAGI_10X10X10_INWARD_4D_ANN.md) for the fold, forward pass, analytic gradient, pruning, and capability boundary.

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

Run the volumetric ROM ANN:

```bash
./build/cpp-runtime/DrMoagi-Volumetric-ROM-ANN \
  --cycles 32 \
  --active-tiles 64
```

The volumetric target exposes a `2^60` logical address universe but only materializes bounded active `32^3` tiles. Each tile contracts inward through a dyadic 3D pyramid, enters a recursive latent fixed-point loop, reconstructs outward, computes `e = X - X_hat`, then updates adaptive memory, model parameters and the bounded runtime policy. See [1 MiB × 1 MiB × 1 MiB Volumetric ROM ANN](docs/volumetric-rom-ann.md).

See [`cpp_runtime/README.md`](cpp_runtime/README.md) for C++ state artifacts, determinism contracts, sanitizer builds and capability limits.

## Architecture

```text
                    CodexVM / control plane
                              │
                              ▼
                 typed candidate transaction
                              │
                              ▼
                bounded research data plane
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                                           ▼
 sparse / volumetric profiles              multimodal / generative
        │                                           │
        └───────────────┬───────────────────────────┘
                        ▼
 encode → compact → residuals → fusion → fixed point
        → decode → contrast / evidence
        → stage Omega_mem / Theta_model / Pi_runtime
                        │
                        ▼
              Verify + Pi_Lambda
                  ╱             ╲
              COMMIT          ROLLBACK
                  │               │
                  └───── audit ───┘
                        │
                        └────────────→ recur
```

The canonical design rules are documented in [Architecture](docs/ARCHITECTURE.md). ADR-016 defines typed structural closure; ADR-017 defines the operational auto-encoding/decoding processing law inside that closure.

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
| Typed-state migration | migrate representative VM/sparse/C++ paths to ADR-016/017 stage receipts | Issue #260 / follow-up integration |
| Browser engines | bounded interactive 3D visual-computing demonstrations | Separate repository |

Draft status is intentional: experimental subsystems are not represented as canonical implementations until CI, review and integration boundaries are satisfied.

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
- residual/side-information accounting where compaction is claimed;
- no performance or intelligence claim without measurement.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Project status](docs/PROJECT_STATUS.md)
- [Canonical Dr Moagi operational auto-encoding/decoding equation](docs/DR_MOAGI_OPERATIONAL_AUTOENCODING_EQUATION.md)
- [ADR-016: canonical typed state, transaction and geometry profiles](docs/adr/0016-canonical-typed-state-transaction-and-geometry-profiles.md)
- [ADR-017: canonical Dr Moagi operational auto-encoding/decoding equation](docs/adr/0017-dr-moagi-operational-autoencoding-equation.md)
- [1 MiB × 1 MiB × 1 MiB Volumetric ROM ANN](docs/volumetric-rom-ann.md)
- [Dr Moagi 3D Ephemeral-Notion Intelligence Framework — canonical attribution](docs/attribution/DR_MOAGI_EPHEMERAL_NOTION_FRAMEWORK.md)
- [Dr Moagi attribution permeation manifest](docs/attribution/PERMEATION_MANIFEST.md)
- [ADR-015: Dr Moagi attribution and provenance permeation](docs/adr/0015-dr-moagi-ephemeral-notion-attribution-provenance.md)
- [Inward 3D kinetic end-to-end specification](docs/INWARD_3D_KINETIC_END_TO_END.md)
- [10x10x10 inward 4D graph ANN](docs/DR_MOAGI_10X10X10_INWARD_4D_ANN.md)
- [Dr. Moagi 4D quantum-inspired autoencoding equation](docs/DR_MOAGI_4D_QUANTUM_INSPIRED_AUTOENCODING.md)
- [Hierarchical 3D fractional smoothing](docs/HIERARCHICAL_3D_FRACTIONAL_SMOOTHING.md)
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

Virtual address-space size is not resident memory. Exponential compaction is not automatically lossless; discarded information must be accounted for by residual/side information or declared lossy. A local fixed point is not proof of external correctness or global equilibrium. A deterministic simulation is not evidence of general intelligence. A cryptographic digest provides integrity, not reversibility. The C++ processor mutates bounded parameters and schedules; it does not rewrite arbitrary native code. The volumetric ROM ANN's 1 EiB figure is a `2^60` logical address-capacity statement, not a claim that 1 EiB of memory is physically allocated. ADR-017 is a canonical systems specification, not evidence that every backend already implements the complete equation. The fractional solver is a small-grid CPU reference, not a calibrated physical model or production FFT implementation.

## Citation

Academic and technical users can cite the project using [`CITATION.cff`](CITATION.cff). Work implementing or discussing the named Dr Moagi 3D Ephemeral-Notion Intelligence Framework should additionally preserve the canonical attribution recorded in [`docs/attribution/DR_MOAGI_EPHEMERAL_NOTION_FRAMEWORK.md`](docs/attribution/DR_MOAGI_EPHEMERAL_NOTION_FRAMEWORK.md).

## License

Jarvis-X is released under the [MIT License](LICENSE).
