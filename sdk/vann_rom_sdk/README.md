# VANN-ROM Ω³ SDK

A runnable Python semantic reference implementation of the **3D 1000 GB/s ROM Bytecode Auto-Encoding/Decoding Virtual ANN Processor**.

Version `0.2.0` hardens the processor around non-bypassable transactions, sealed ROM images, CRC-validated binary execution, deterministic journaling, bounded runtime optimization, and rollback-safe ANN adaptation.

The 1000 GB/s value remains an architectural hardware target. This SDK validates processor semantics; it does not claim physical 1 TB/s throughput from Python.

## Implemented architecture

- **Sparse 3D ROM** — only mapped voxel pages consume memory.
- **Sealed ROM manifests** — page topology, Λ masks, instructions, parameters, metadata, and neighbours are integrity-hashed.
- **128-bit ISA** — fixed-width instructions with CRC-8 validation.
- **Verified bytecode images** — `.vann` source compiles to executable `.vbc` streams.
- **3D program layout** — sequential instructions map onto XYZ coordinates with Morton address support.
- **ANN engine** — NumPy encoder, latent field, predictor, residual and decoder.
- **Transactional Ω adaptation** — model weights and residual memory are staged together.
- **Mandatory Λ boundary** — every commit projects and verifies internally, even when bytecode omits explicit preflight instructions.
- **Atomic rollback** — model, Ω and output snapshots are restored if verification fails.
- **Automatic journal** — every commit, rollback, checkpoint and policy decision records ROM, model and state hashes.
- **Bounded optimizer** — policy candidates are range-checked and accepted only when predicted constrained cost does not regress.
- **CLI and IDE** — source editing, assembly, binary inspection, execution and runtime reporting.

## Installation

```bash
cd sdk/vann_rom_sdk
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -e .
```

## Run the source demonstration

```bash
vann-rom demo
```

## Compile, inspect and execute bytecode

```bash
vann-rom assemble examples/demo.vann -o program.vbc
vann-rom inspect program.vbc
vann-rom run-bytecode \
  --bytecode program.vbc \
  --input examples/input.json \
  --latent-dim 4
```

Every instruction is exactly 16 bytes:

\[
16\text{ bytes}=128\text{ bits}
\]

The loader rejects incomplete images and any instruction whose CRC no longer matches its payload.

## Run source directly

```bash
vann-rom run \
  --source examples/demo.vann \
  --input examples/input.json \
  --latent-dim 4
```

## Launch the virtual IDE

```bash
vann-rom ide
```

The IDE provides a VANN source editor, JSON tensor input panel, assembler listing, runtime report, metrics, ROM manifest and execution journal.

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
COMMIT
DECODE3D
STAGE
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
print(result.to_dict())
```

The example deliberately omits explicit `PROJECT_LAMBDA` and `VERIFY` opcodes. `COMMIT` still performs both operations internally; governance cannot be bypassed by source structure.

## Architectural mapping

```text
ROM       -> sealed Sparse3DROM + VoxelPage bytecode
RAM       -> tensor registers, prefetch cache and transaction staging
Ω         -> residual EMA, correction bias and episodic journal
Λ         -> root mask, projection, finite-value and shape verification
Σ         -> X, Z, P, R, E, Ω, Y, G and PC3 runtime state
Optimizer -> bounded RuntimePolicy proposal and constrained acceptance
```

## Invariants

```text
ROM defines
RAM executes
Ω adapts
Λ verifies
COMMIT journals
ROLLBACK restores
Σ evolves
```

Model training does not mutate authoritative weights until its complete transaction is verified and committed.

## Tests

```bash
python -m unittest discover -s tests -v
```

GitHub Actions validates Python 3.10, 3.11, 3.12 and 3.13 and exercises:

- package installation and dependency integrity;
- source compilation;
- CRC and bytecode-stream validation;
- sealed-ROM tamper detection;
- mandatory commit verification;
- uncommitted model rollback;
- source demonstration execution;
- `.vbc` assembly, inspection and execution.

## Jarvis-X integration

The SDK lives at `sdk/vann_rom_sdk` inside Jarvis-X. The root CodexVM was also hardened during this integration:

- canonical JSON-safe hash-chained ledger;
- atomic persistent-ledger replacement;
- deterministic arithmetic by default;
- reflex adaptation explicitly opt-in;
- no reflex mutation after `HALT`;
- clean execution reset when a new program is loaded.

The Python runtime remains the semantic reference model for later cost-model, native CPU, GPU, FPGA and HBM-backed implementations.
