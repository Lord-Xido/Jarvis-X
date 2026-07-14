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

## 30D Virtual ANN Processor

Jarvis-X includes a sparse, deterministic 30-dimensional neural bytecode
processor. It maps arbitrary input vectors into 30 signed three-bit latent axes,
materializes only activated coordinates in the theoretical `8 ** 30` address
space, advances 30-component virtual field state, predicts, learns from a
residual, applies bounded-state projection, and decodes an output.

```python
from jarvisx.ann30d import VirtualANNProcessor30D

processor = VirtualANNProcessor30D()
state = processor.run([0.8, -0.3, 0.5, 1.0], target=0.8)

print(state.coordinate)
print(state.prediction, state.residual, state.memory)
print(state.output)
```

See `docs/DR_MOAGI_30D_VIRTUAL_ANN_PROCESSOR.md` for the operational arithmetic
and bytecode cycle.
