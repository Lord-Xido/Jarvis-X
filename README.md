# Jarvis-X

Jarvis-X is a deterministic, auditable virtual machine with sparse 30-dimensional and geometric 3-dimensional ANN execution units.

## Install

```bash
git clone https://github.com/Lord-Xido/Jarvis-X.git
cd Jarvis-X
pip install -e .
```

## Mathematical 3D abstraction core

The 3D abstraction processor projects arbitrary vectors into a normalized 16-component feature manifold, maps the feature state into a continuous 3D coordinate, and activates at most eight trilinear lattice nodes. Sparse attention, local prototype learning, residual memory, projection, and decoding therefore operate in constant routed width rather than over the full `64 ** 3` lattice.

```text
LOAD3D
ABSTRACT3D
ROUTE3D
ATTEND3D
PREDICT3D
COMPARE3D
LEARN3D
PROJECT3D
DECODE3D
HALT3D
```

```bash
jarvisx abstract3d '[0.8, -0.3, 0.5, 1.0]' --target 0.8
```

```python
from jarvisx.abstraction3d import AbstractionANNCore3D

core = AbstractionANNCore3D()
state = core.run([0.8, -0.3, 0.5, 1.0], target=0.8)

print(state.route)
print(state.attention)
print(state.prediction, state.loss)
print(state.output)
```

See `docs/DR_MOAGI_3D_ABSTRACTION_ANN_CORE.md` for the complete arithmetic model.

## Unified 30D bytecode

The 30D processor executes through the same assembler, 64-bit decoder, policy gate, sandbox, tracer, register file, and audit ledger as scalar Jarvis-X instructions.

```text
LOAD30
ENCODE30
PLACE30
FIELD30
PREDICT30
COMPARE30
UPDATE_MEMORY30
PROJECT30
DECODE30
HALT30
```

```python
from jarvisx.assembler import Assembler
from jarvisx.core import CodexVM
from jarvisx.parser import Parser

source = """LOAD30
ENCODE30
PLACE30
FIELD30
PREDICT30
COMPARE30
UPDATE_MEMORY30
PROJECT30
DECODE30
HALT30"""

vm = CodexVM()
vm.load(
    Assembler().assemble(Parser().parse(source)),
    ann_input=[0.8, -0.3, 0.5, 1.0],
    ann_target=0.8,
)
print(vm.run())
```

## CLI

```bash
jarvisx abstract3d '[0.8, -0.3, 0.5, 1.0]' --target 0.8
jarvisx ann30d '[0.8, -0.3, 0.5, 1.0]' --target 0.8
jarvisx run program.jx --ann-input '[0.8, -0.3, 0.5, 1.0]' --ann-target 0.8
jarvisx api
```

## API

```bash
jarvisx api --host 127.0.0.1 --port 8080
```

Endpoints:

- `GET /health`
- `POST /v1/run/assembly`
- `POST /v1/run/ann30d`
- `POST /v1/run/abstraction3d`

Set `JARVISX_API_TOKEN` to require `Authorization: Bearer <token>`.

## Operational guarantees

- strict registered-opcode allowlist
- program, cycle, input, and active-state quotas
- transactional rollback around ANN mutations
- bounded prototype, memory, field, activation, and confidence projection
- canonical JSON hash-chain ledger with atomic persistence
- per-request isolated service execution
- sparse `8 ** 30` virtual addressing
- sparse trilinear 3D routing with at most eight nodes per observation

The 30 dimensions are virtual computational axes. `FIELD30` is a bounded software coupled-field operator, not a claim of literal 30-dimensional Maxwell physics.

See `docs/DR_MOAGI_30D_VIRTUAL_ANN_PROCESSOR.md` for the 30D arithmetic model.
