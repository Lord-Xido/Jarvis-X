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

## VANN-ROM Ω³ Virtual ANN SDK

The repository includes the executable Python reference implementation of the
**3D 1000 GB/s ROM Bytecode Auto-Encoding/Decoding Virtual ANN Processor** in
[`sdk/vann_rom_sdk`](sdk/vann_rom_sdk).

```bash
cd sdk/vann_rom_sdk
python -m pip install .
vann-rom demo
```

The SDK provides sparse 3D ROM addressing, a 128-bit neural bytecode ISA,
assembler, auto-encoding and decoding runtime, Ω adaptive memory, Λ-gated
transactional commits, bounded auto-optimization, CLI tools, and a local
Tkinter virtual IDE.
