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

## Electronic permeation runtime

Every `CodexVM.step()` can now be projected into a deterministic electronic
state-transition trace:

```text
instruction → register transitions → gate activity → timing
            → energy/power → thermal state → electronic Λ gate
```

The model exposes register and instruction-bus Hamming transitions,
opcode-specific gate activity, critical-path timing, switching energy, power,
thermal evolution, and a transactional timing/thermal validity decision.

```python
from jarvisx.core import CodexVM
from jarvisx.electronic import ElectronicConfig, ElectronicSubstrate

substrate = ElectronicSubstrate(
    ElectronicConfig(clock_hz=1_000_000_000.0, enforce_limits=True)
)
vm = CodexVM(electronics=substrate)
vm.load(bytecode)
vm.run()
print(vm.electronics.snapshot())
```

These values are deterministic model outputs, not direct hardware sensor
measurements. See
[`docs/DR_MOAGI_ELECTRONIC_PERMEATION_RUNTIME.md`](docs/DR_MOAGI_ELECTRONIC_PERMEATION_RUNTIME.md)
for the equations, execution sequence, provenance rules, and measured-backend
extension points.
