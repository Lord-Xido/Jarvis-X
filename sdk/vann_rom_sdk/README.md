# VANN-ROM Ω³ SDK

A runnable Python reference implementation of the **3D 1000 GB/s ROM Bytecode Auto-Encoding/Decoding Virtual ANN Processor**.

The SDK models the architecture rather than claiming physical 1 TB/s memory throughput from Python. It implements the control semantics, sparse 3D virtual ROM, 128-bit bytecode format, autoencoder/decoder pipeline, Ω adaptive overlay, Λ transaction boundary, metrics, bounded auto-optimizer, assembler, CLI and a lightweight Tkinter IDE.

## Components

- **Sparse 3D ROM**: only mapped voxel pages consume memory.
- **128-bit ISA**: fixed-width instructions with CRC validation.
- **3D program layout**: sequential bytecode is mapped onto XYZ coordinates.
- **ANN engine**: NumPy encoder, latent field, predictor and decoder.
- **Ω overlay**: writable residual-driven adaptation separate from immutable ROM.
- **Λ boundary**: staged verification and atomic commit.
- **Auto-optimizer**: bounded changes to learning rate, sparsity, prefetch and fusion policy.
- **Assembler**: readable `.vann` source to bytecode.
- **CLI and IDE**: command-line execution and local Tkinter development environment.

## Installation

```bash
cd vann_rom_sdk
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
python -m pip install -e .
```

## Run the demonstration

```bash
vann-rom demo
```

Or:

```bash
python -m vann_rom_sdk.cli demo
```

## Launch the virtual IDE

```bash
vann-rom ide
```

The IDE provides a VANN source editor, JSON tensor input panel, assembler listing, runtime report, metrics, ROM manifest and execution journal.

## Assemble bytecode

```bash
vann-rom assemble examples/demo.vann -o program.vbc
```

Each instruction is exactly 16 bytes.

## Run a source program

```bash
vann-rom run --source examples/demo.vann --input examples/input.json --latent-dim 4
```

## Python API

```python
import numpy as np
from vann_rom_sdk import Assembler, TinyAutoencoder, VANNVirtualMachine

source = """
LOAD_INPUT
NORMALIZE
ENCODE3D
PREDICT
COMPARE
UPDATE_OMEGA
PROJECT_LAMBDA
VERIFY
COMMIT
DECODE3D
STAGE
VERIFY
COMMIT
RENDER
HALT
"""

program = Assembler().assemble(source)
model = TinyAutoencoder(input_dim=12, latent_dim=4)
vm = VANNVirtualMachine(model, output_sink=print)
vm.load_program(program.instructions)
vm.set_input(np.random.default_rng(7).random((1, 12), dtype=np.float32))
result = vm.run()
print(result.metrics)
```

## Architectural mapping

```text
ROM       -> Sparse3DROM + immutable VoxelPage bytecode
RAM       -> VM tensor registers, caches and transaction staging
Ω         -> residual EMA, working correction bias and episodic journal
Λ         -> root mask, range projection, finite-value verification
Σ         -> X, Z, P, R, E, Ω, Y, G and PC3 runtime state
Optimizer -> bounded RuntimePolicy controller
```

## Throughput interpretation

The 1000 GB/s specification is represented as an architectural target. This Python SDK is an executable semantic model, not a hardware bandwidth benchmark. A hardware implementation would map the same ISA and execution model to HBM-class channels, DMA engines, tensor tiles and hierarchical caches.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Jarvis X integration

This SDK is maintained inside the Jarvis X repository at `sdk/vann_rom_sdk`.
Its immutable ROM, mutable Ω overlay, and Λ-gated commit model implement the
Jarvis X / Dr Moagi operational separation:

```text
ROM defines → RAM executes → Ω adapts → Λ verifies → Σ commits
```

The Python runtime is the semantic reference model for later native, GPU,
FPGA, or HBM-backed implementations.
