# VANN-ROM Ω³ SDK

A runnable Python semantic reference implementation of the **3D ROM Bytecode Virtual ANN Processor** and the **Aether sparse 4D multimodal auto-encoding engine**.

Version `0.3.0` adds a Morton-ordered four-dimensional video, audio, graph, and context processor with hybrid SSM/KAN/liquid encoding, cross-modal attention, selectable latent evolution, modality decoders, multi-objective evaluation, verified Ω adaptation, and bounded policy search.

The 1000 GB/s value remains an architectural hardware target. This SDK validates processor semantics and cost models; it does not claim physical 1 TB/s throughput from Python.

## Implemented architecture

### VANN-ROM Ω³

- **Sparse 3D ROM** — only mapped voxel pages consume memory.
- **Sealed ROM manifests** — page topology, Λ masks, instructions, parameters, metadata, and neighbours are integrity-hashed.
- **128-bit ISA** — fixed-width instructions with CRC-8 validation.
- **Verified bytecode images** — `.vann` source compiles to executable `.vbc` streams.
- **3D program layout** — sequential instructions map onto XYZ coordinates with Morton address support.
- **ANN engine** — NumPy encoder, latent field, predictor, residual and decoder.
- **Transactional Ω adaptation** — model weights and residual memory are staged together.
- **Mandatory Λ boundary** — every commit projects and verifies internally.
- **Atomic rollback** — model, Ω and output snapshots are restored if verification fails.
- **Automatic journal** — every commit, rollback, checkpoint and policy decision records integrity hashes.
- **Bounded optimizer** — candidates are accepted only when constrained cost does not regress.

### Aether Engine v1

- **Multimodal state** — normalized video, audio, graph, and context tensors.
- **Sparse 4D field** — active tokens receive `(time, x, y, modality-plane)` coordinates.
- **Morton4D ordering** — four unsigned 16-bit coordinates interleave into a 64-bit locality key.
- **Hybrid encoder** — SSM recurrence, KAN-style nonlinear basis, liquid-state recurrence, gating, and layer normalization.
- **Cross-modal attention** — configurable bias promotes interactions between modality planes.
- **Latent processor** — bounded SSM or Euler neural-flow evolution.
- **Multimodal decoder** — reconstructs video, audio, graph nodes and adjacency, and context.
- **Multi-objective loss** — reconstruction, perceptual, semantic, efficiency, and constrained novelty terms.
- **Verified Ω overlay** — candidate online updates execute in shadow and commit only when the constrained objective improves.
- **Bounded policy search** — evaluates declared evolution, recurrence, and cross-modal-gain candidates; no arbitrary source mutation.
- **Hash journal** — sealed-base, adaptation, rollback, and policy decisions form a deterministic hash chain.

See [`docs/AETHER_ENGINE_V1.md`](docs/AETHER_ENGINE_V1.md) for the complete execution contract.

## Installation

```bash
cd sdk/vann_rom_sdk
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -e .
```

## Run Aether Engine

Synthetic deterministic workload:

```bash
vann-rom aether-demo
```

Verified adaptation and bounded policy search:

```bash
vann-rom aether-demo --adapt --optimize
```

External normalized multimodal JSON:

```bash
vann-rom aether-run \
  --input examples/aether_input.json \
  --adapt \
  --optimize
```

Add `--include-arrays` to include reconstructed tensors in the JSON report.

## Run the VANN source demonstration

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

## Run VANN source directly

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

### Aether

```python
from vann_rom_sdk import AetherEngine, synthetic_aether_input

engine = AetherEngine()
result = engine.run(
    synthetic_aether_input(),
    adapt=True,
    optimize=True,
)

print(result.loss.total)
print(result.policy)
print(result.state_digest)
```

### VANN-ROM

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
ROM       -> sealed Sparse3DROM + sealed Aether base parameter bank
RAM       -> tensor registers, sparse 4D fields, caches and transaction staging
Ω         -> residual memory and bounded Aether overlay
Λ         -> projection, shape, range, finite-value and semantic verification
Σ         -> multimodal input, latent state, output, residual, policy and journal
Optimizer -> declared candidate generation, shadow execution and constrained acceptance
```

## Invariants

```text
ROM defines
RAM executes
4D encoder represents
latent processor evolves
multimodal decoder reconstructs
Ω adapts
Λ verifies
COMMIT journals
ROLLBACK restores
Σ evolves
```

Model adaptation never mutates the sealed base parameter bank. The active model is `θsealed + ΔθΩ`, and the overlay changes only after a successful shadow comparison.

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
- sparse 4D Morton round trips;
- multimodal shape reconstruction;
- deterministic Aether execution;
- non-regressing transactional adaptation;
- bounded policy search;
- synthetic and external JSON Aether execution;
- `.vbc` assembly, inspection and execution.

## Jarvis-X integration

The SDK lives at `sdk/vann_rom_sdk` inside Jarvis-X. The root CodexVM is also hardened with:

- canonical JSON-safe hash-chained ledger;
- atomic persistent-ledger replacement;
- deterministic arithmetic by default;
- reflex adaptation explicitly opt-in;
- no reflex mutation after `HALT`;
- clean execution reset when a new program is loaded.

The Python runtime remains the semantic reference model for later cost-model, native CPU, GPU, distributed, FPGA and HBM-backed implementations.
