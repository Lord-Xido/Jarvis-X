# Jarvis-X

Jarvis-X is a deterministic, auditable virtual machine with a reflex
control layer and policy gate.

## Architecture constitutions

- [Dr Moagi Unified Auto-Encoding Dynamics (D-MUAD v2.0-C)](docs/DR_MOAGI_UNIFIED_AUTO_ENCODING_DYNAMICS.md)
- The executable D-MUAD contract is implemented in `src/jarvisx/d_muad.py`.

D-MUAD formalizes a compressive, recurrent, physics-regularized, dual-branch
geometric autoencoding architecture. It exposes exact padding and tensor-shape
rules, arithmetic budget functions, the corrected five-term loss, Adam
recurrence, and explicit injectivity, determinism, and differentiability
boundaries.

```python
from jarvisx.d_muad import derive_contract

contract = derive_contract((1, 4, 9, 17, 25))
assert contract.padded == (1, 4, 16, 24, 32)
assert contract.compressive
assert not contract.exact_global_inverse_possible
```

## Install

```bash
git clone https://github.com/Lord-Xido/Jarvis-X.git
cd Jarvis-X
pip install -r requirements.txt
pip install .
```
