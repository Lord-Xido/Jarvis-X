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

## MM3D Auto-Encoding/Decoding

The VM includes a deterministic signed-3-bit AED subsystem implementing the
JARVISX-HSLF-QSOL-DM-vΩΞ+++ master mapping:

```python
from jarvisx.core import CodexVM

vm = CodexVM()
state = vm.aed_cycle(
    [0, 64, 128, 192, 255],
    memory=[0],
    intent=[0.25],
    constraints=[(-4.0, 3.0)],
)

print(state.ambient_output)
```

See [`docs/JARVISX_HSLF_QSOL_MM3D_AED_MASTER_EQUATION.md`](docs/JARVISX_HSLF_QSOL_MM3D_AED_MASTER_EQUATION.md)
for the closed-form equation, operator semantics, invariants, and VM mapping.

## JX-AAPE-Ω Bit-Packed Topological Engine

The 8× target variant evolves a 64³ Boolean torus, applies exact
majority-of-seven dynamics, extracts Python vocabulary tokens through a sparse
intent gate, and binds each commit into an Ω SHA3-256 chain:

```python
from jarvisx.core import CodexVM

vm = CodexVM()
intent = vm.aape.lattice([0, 1, 2, 3])

state = vm.aape_cycle(
    [0x5000, 0x6000, 0x7000],
    intent_mask=intent,
    quality_signal=1,
    lambda_tag=b"policy-v1",
)

print(state.tokens)
print(state.convergence)
print(state.omega_digest)
```

The decoder currently emits lexical tokens; executable Python requires a
separate grammar or AST-construction stage.

See [`docs/JX_AAPE_OMEGA_8X_TOPOLOGICAL_DYNAMICS.md`](docs/JX_AAPE_OMEGA_8X_TOPOLOGICAL_DYNAMICS.md)
for the corrected GF(2) state model, exact Boolean threshold network,
convergence semantics, invariants, and throughput derivation.

Measure the scalar-to-packed speedup on the executing host with:

```bash
python benchmarks/benchmark_aape.py --side 16 --repeats 7
```
