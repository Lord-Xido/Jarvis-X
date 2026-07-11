# Jarvis-X

Jarvis-X is a deterministic, auditable virtual machine with a reflex
control layer and policy gate.

## 30D auto-encoding/decoding engine

Every committed VM instruction is observed by a bounded sparse cognitive
runtime:

```text
observe -> route30 -> encode -> predict -> residual -> update omega
        -> correct -> decode -> coherence projection -> commit
```

The 30D coordinate space is virtual. Only deterministically routed cells are
materialised, and `max_active_cells` bounds physical memory consumption. The
cognitive layer does not mutate the authoritative VM register state.

```python
from jarvisx.core import CodexVM

vm = CodexVM()
result = vm.cognitive_cycle("Jarvis X, echo through.")

print(result.active_coordinates)
print(result.reconstruction_error)
print(result.coherence)
```

See `docs/30D_AUTOENCODING_ENGINE.md` for the operational contract.

## Install
```bash
git clone https://github.com/Lord-Xido/Jarvis-X.git
cd Jarvis-X
pip install -r requirements.txt
pip install .
```
