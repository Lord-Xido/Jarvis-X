# Jarvis-X

Jarvis-X is a deterministic, auditable virtual machine with a sparse 30-dimensional virtual ANN execution unit.

## Install

```bash
git clone https://github.com/Lord-Xido/Jarvis-X.git
cd Jarvis-X
pip install -e .
```

## Unified 30D bytecode

The 30D processor now executes through the same assembler, 64-bit decoder, policy gate, sandbox, tracer, register file, and audit ledger as scalar Jarvis-X instructions.

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

Set `JARVISX_API_TOKEN` to require `Authorization: Bearer <token>`.

## Operational guarantees

- strict registered-opcode allowlist
- program, cycle, input, and active-cell quotas
- transactional rollback around 30D mutations
- bounded virtual-field and memory projection
- canonical JSON hash-chain ledger with atomic persistence
- per-request isolated service execution
- deterministic sparse `8 ** 30` addressing

The 30 dimensions are virtual computational axes. `FIELD30` is a bounded software coupled-field operator, not a claim of literal 30-dimensional Maxwell physics.

See `docs/DR_MOAGI_30D_VIRTUAL_ANN_PROCESSOR.md` for the arithmetic model.
