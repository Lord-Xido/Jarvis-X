# Jarvis-X Operational Kernel v0.2

## Status

This document defines the implemented runtime contract for Jarvis-X v0.2. It separates executable behavior from future architecture.

## 1. Transaction boundary

For instruction `I_t` and committed machine state `S_t`, Jarvis-X evaluates:

\[
S_{t+1}^{candidate}=F(I_t,S_t)
\]

The candidate is projected through the deterministic electronic model:

\[
T_{e,t}=E(I_t,S_t,S_{t+1}^{candidate})
\]

The commit decision is:

\[
\Lambda_t = \Lambda_{policy} \land \Lambda_{sandbox} \land \Lambda_{timing} \land \Lambda_{thermal}
\]

When enforcement is enabled:

\[
S_{t+1}=
\begin{cases}
S_{t+1}^{candidate}, & \Lambda_t=1\\
S_t, & \Lambda_t=0
\end{cases}
\]

A rejected instruction restores:

- all registers;
- the complete memory byte array;
- electronic cycle count;
- cumulative modeled energy;
- modeled junction temperature;
- previous instruction-bus state;
- electronic trace length;
- VM cycle count;
- running state;
- instruction trace length;
- Omega ledger length and logical time.

## 2. Committed execution sequence

```text
FETCH
DECODE
POLICY_CHECK
CHECKPOINT
EXECUTE
UPDATE_FLAGS
RESOLVE_NEXT_IP
SANDBOX_CHECK
ELECTRONIC_PROJECT
ELECTRONIC_LAMBDA_CHECK
OMEGA_LOG
TRACE_COMMIT
STATE_COMMIT
```

No ledger or instruction-trace entry is retained for a rejected transition.

## 3. Register bank

The runtime exposes 16 integer registers:

```text
Ξ Ψ Φ Λ Ω Θ 𝒮 Π A B C D IP SP FLAGS TMP
```

`FLAGS` currently uses:

- bit 0: zero;
- bit 1: negative.

The reflex controller is opt-in. Ordinary assembly execution does not implicitly mutate `Ψ` or `Φ`.

## 4. Memory

The default memory is 4096 bytes. Access is bounds checked.

`LOAD` and `STORE` operate on signed 64-bit little-endian values. Stores preserve the low 64 bits of the source integer.

## 5. Instruction format

The 64-bit instruction word remains:

```text
63        56 55      48 47      40 39      32 31      24 23       8 7       0
+-----------+----------+----------+----------+----------+-----------+---------+
| opcode    | reserved | dst      | src1     | src2     | imm16     | reserved|
+-----------+----------+----------+----------+----------+-----------+---------+
```

The implemented ISA is:

```text
SET MOV ADD SUB LOAD STORE CMP JMP JZ HALT JNZ MUL XOR AND OR
```

The assembler resolves labels in two passes and encodes signed 16-bit immediates using two's complement.

## 6. Omega provenance

Each committed entry contains canonical JSON-compatible data:

```text
logical_time
opcode
state
metadata
previous_hash
hash
```

Hash material excludes wall-clock time. Replay from the same initial state and bytecode therefore produces the same ledger hash sequence.

Persistent ledgers are written through a temporary file, flushed, fsynced, and atomically replaced. An unsuccessful persistence operation restores the in-memory ledger checkpoint.

## 7. Electronic provenance

The deterministic electronic backend estimates:

```text
register_bit_transitions
instruction_bus_bit_transitions
gate_toggles
dynamic_energy_j
cumulative_energy_j
dynamic_power_w
total_power_w
junction_temp_c
critical_path_ns
timing_margin_ns
lambda_accept
```

Every telemetry object is labeled `deterministic-model`. The implementation does not claim direct hardware measurement.

## 8. Interfaces

### CLI

```bash
jarvisx run program.jx
jarvisx run program.jx --ledger omega.json
jarvisx api
jarvisx web
jarvisx node
```

### API

```text
GET  /health
POST /run
```

Each HTTP request creates an isolated VM execution.

### TCP node

The node binds to `127.0.0.1` by default, limits requests to 64 KiB, applies socket timeouts, returns JSON, and creates isolated executions.

## 9. Verification

The v0.2 test suite covers:

- signed immediate and label resolution;
- arithmetic semantics;
- loops and conditional branching;
- memory store/load round trips;
- deterministic ledger replay;
- JSON-safe persistent provenance;
- timing violation detection;
- bounded electronic traces;
- electronic checkpoint restoration;
- whole-cycle rollback.

## 10. Explicit non-claims

Jarvis-X v0.2 does not yet implement:

- a frontier language or vision model;
- natural-language planning;
- external tool adapters;
- browser or desktop control;
- semantic policy evaluation;
- multi-user authentication or tenancy;
- distributed consensus or scheduling;
- measured RTL, FPGA, GPU, CPU, or silicon telemetry;
- SOTA agent benchmark performance.

The implemented result is a reliable deterministic kernel onto which those layers can be added without surrendering transactionality, provenance, or policy authority.
