# Jarvis-X

Jarvis-X is a deterministic, auditable virtual machine with a reflex
control layer and policy gate.

## Geometric auto-encoding runtime

Jarvis-X includes a dependency-free geometric execution subsystem that maps
linear arithmetic state onto a validated 3D lattice, applies exact bijective
coordination transforms, and commits candidates transactionally.

```python
from jarvisx.core import CodexVM

vm = CodexVM()
vm.load_geometry(list(range(8)), (2, 2, 2))
vm.execute_geometry((1, 2, 3, 4, 5, 6, 7, 0))

assert vm.geometry.committed is not None
```

The complete mathematical contract, validation gate, spatial Omega memory,
and extension boundary are documented in
[`docs/GEOMETRIC_AUTOENCODING_RUNTIME.md`](docs/GEOMETRIC_AUTOENCODING_RUNTIME.md).

## Install
```bash
git clone https://github.com/Lord-Xido/Jarvis-X.git
cd Jarvis-X
pip install -r requirements.txt
pip install .
```
