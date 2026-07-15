# Jarvis-X

Jarvis-X is a deterministic, auditable virtual machine and transactional agent-control substrate. It combines a compact 64-bit instruction format with bounded memory, a policy gate, a verifiable Omega provenance ledger, whole-cycle rollback, and a deterministic electronic execution model.

Jarvis-X is not a frontier language model. It is the execution and verification layer through which models, tools, or human-authored programs can act under explicit constraints.

## Operational invariant

```text
proposal → decode → policy check → checkpoint → execute
         → electronic projection → Lambda validation
         → ledger + trace commit
```

If any enforced constraint fails, registers, memory, instruction trace, electronic counters, energy, thermal state, cycle state, and ledger state are restored to the pre-instruction checkpoint.

## Install

```bash
git clone https://github.com/Lord-Xido/Jarvis-X.git
cd Jarvis-X
pip install -e .
```

For development:

```bash
pip install -e '.[dev]'
pytest
```

## Run a program

```bash
jarvisx run examples/sum_loop.jx
```

Persist the deterministic Omega ledger:

```bash
jarvisx run examples/sum_loop.jx --ledger omega_ledger.json
```

## Assembly example

```text
SET A 3
SET B 1
SET C 0

loop:
ADD C C A
SUB A A B
CMP A Ξ
JNZ loop

STORE C 0
LOAD D 0
HALT
```

The program computes `3 + 2 + 1`, stores the result in memory, loads it into `D`, and commits every successful instruction to the Omega ledger.

## Implemented ISA

| Instruction | Form | Operation |
|---|---|---|
| `SET` | `SET dst imm` | Write a signed 16-bit immediate |
| `MOV` | `MOV dst src` | Copy a register |
| `ADD` | `ADD dst a b` | Integer addition |
| `SUB` | `SUB dst a b` | Integer subtraction |
| `MUL` | `MUL dst a b` | Integer multiplication |
| `XOR` | `XOR dst a b` | Bitwise XOR |
| `AND` | `AND dst a b` | Bitwise AND |
| `OR` | `OR dst a b` | Bitwise OR |
| `LOAD` | `LOAD dst address` | Load a signed 64-bit value |
| `STORE` | `STORE src address` | Store a 64-bit value |
| `CMP` | `CMP a b` | Set zero and negative flags |
| `JMP` | `JMP label` | Unconditional branch |
| `JZ` | `JZ label` | Branch when zero flag is set |
| `JNZ` | `JNZ label` | Branch when zero flag is clear |
| `HALT` | `HALT` | Stop execution |

Labels use `name:` syntax. `#` and `;` introduce comments.

## Services

Start the local API:

```bash
jarvisx api
```

- `GET /health`
- `POST /run` with `{"source": "SET A 1\nHALT"}`

Start the browser control panel:

```bash
jarvisx web
```

Start the bounded local TCP node:

```bash
jarvisx node --host 127.0.0.1 --port 9000
```

The services bind to loopback by default. Production deployment still requires authentication, authorization, TLS, tenancy controls, rate limiting, and process isolation.

## Deterministic Omega ledger

Each committed instruction records:

- logical time;
- opcode;
- canonical register state;
- electronic register checksum;
- electronic Lambda decision;
- telemetry provenance;
- previous entry hash;
- current entry hash.

The ledger excludes wall-clock time from the hash material, so identical initial state and bytecode produce identical ledger hashes.

## Electronic permeation runtime

Every committed `CodexVM.step()` is projected into a deterministic electronic state-transition trace:

```text
instruction → register transitions → gate activity → timing
            → energy/power → thermal state → electronic Lambda gate
```

The model reports:

- register and instruction-bus Hamming transitions;
- opcode-specific estimated gate activity;
- critical-path timing and timing margin;
- switching energy and modeled power;
- first-order thermal evolution;
- a timing-and-thermal acceptance decision.

These values are deterministic model outputs, not direct hardware sensor measurements. A measured backend may replace the estimator while preserving telemetry provenance.

See:

- `docs/DR_MOAGI_ELECTRONIC_PERMEATION_RUNTIME.md`
- `docs/OPERATIONAL_KERNEL_V0_2.md`

## Current maturity

Version 0.2.0 is an experimental deterministic execution kernel. It now has operational control flow, memory, provenance, rollback, services, and tests. Frontier model integration, tool adapters, semantic policy evaluation, authentication, distributed scheduling, and external agent benchmarks remain future layers.
