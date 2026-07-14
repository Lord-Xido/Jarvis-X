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
