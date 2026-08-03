# Canonical Transactional VM Core

## Status

**Integration candidate.**

This document defines the transaction boundary implemented by the canonical `CodexVM` branch. It narrows the system-wide Jarvis-X operational law into one executable instruction-cycle contract.

The governing rule is:

```text
checkpoint authoritative state
→ decode and authorize
→ enforce the prospective cycle budget
→ execute the instruction
→ apply explicitly enabled reflex correction
→ construct the canonical post-state
→ persist the ledger entry
→ record the trace
→ commit
```

If any operation after the checkpoint fails, the authoritative checkpoint is restored and the VM stops.

## Authoritative state covered by one instruction transaction

A checkpoint contains:

- the complete register file;
- the complete VM memory image;
- the committed cycle count;
- the running state;
- the Omega-ledger length;
- the trace length.

The loaded program is immutable during an instruction cycle and therefore does not require a per-step copy.

The transaction state is:

\[
S_t=(R_t,M_t,c_t,q_t,J_t,T_t),
\]

where:

- \(R_t\) is the register file;
- \(M_t\) is memory;
- \(c_t\) is the committed cycle count;
- \(q_t\) is the run-state flag;
- \(J_t\) is the ledger prefix;
- \(T_t\) is the trace prefix.

For a decoded and authorized instruction \(i_t\), execution proposes:

\[
\widetilde S_{t+1}=F(S_t,i_t).
\]

The candidate commits only if execution, resource enforcement, provenance persistence and tracing all succeed:

\[
S_{t+1}=
\begin{cases}
\widetilde S_{t+1}, & V(\widetilde S_{t+1})=\mathrm{pass},\\
S_t, & V(\widetilde S_{t+1})=\mathrm{fail}.
\end{cases}
\]

A failed transaction stops execution after restoration. The caller must explicitly decide whether to inspect, repair, reload or discard the machine.

## Cycle-budget semantics

The sandbox validates the prospective committed cycle count before instruction execution:

\[
c' = c_t + 1,
\qquad
c' \le c_{\max}.
\]

This prevents an instruction beyond the configured budget from mutating state and then requiring avoidable rollback.

Exactly `max_cycles` instructions may commit. The next attempted instruction fails closed and preserves the last committed checkpoint.

## Reflex ordering

Reflex correction remains disabled by default.

When enabled, it is part of the same instruction transaction and runs before the canonical snapshot is written to the ledger and trace:

```text
execute instruction
→ optional reflex stabilization
→ advance IP and cycle count
→ snapshot
→ ledger
→ trace
```

Therefore the recorded post-state is the state that is actually authoritative after the instruction. Reflex changes are not hidden from provenance.

This does not turn reflex correction into autonomous authority. It remains an explicit constructor option and is bounded by the same rollback boundary.

## Invalid instruction behavior

The executor recognizes the canonical instruction surface:

- `SET` (`0x01`);
- `ADD` (`0x03`);
- `SUB` (`0x04`);
- `HALT` (`0x0A`).

Unknown opcodes raise an error. Invalid register indices raise an error. Neither case emits a committed receipt or advances the instruction pointer.

This replaces silent continuation with fail-closed behavior.

## Memory contract

Memory now enforces:

- positive allocation size;
- integer addresses and lengths;
- non-negative ranges;
- complete in-bounds reads and writes;
- equal-size checkpoint restoration.

A failed access cannot use Python slice behavior to wrap negative indices or silently truncate an out-of-range operation.

## Ledger transactionality

The in-memory Omega ledger exposes prefix checkpoints. Restoration removes every entry after the checkpoint.

The persistent ledger uses the following write transaction:

```text
remember ledger prefix
→ append candidate entry in memory
→ write complete candidate ledger to a temporary file
→ flush and fsync
→ atomically replace the destination
```

If persistence fails, the in-memory candidate entry is removed and the previous on-disk file remains authoritative because replacement is atomic.

If a later VM-stage failure requires restoration after a successful ledger write, the prior ledger prefix is persisted again.

The ledger remains a tamper-evident hash chain. It is not encryption, trusted timestamping, external witnessing or deletion resistance.

## Trace transactionality

The tracer exposes prefix checkpoints and restoration. A rejected instruction cannot leave a trace entry describing state that never committed.

For every committed instruction:

\[
\text{ledger state}=
\text{trace state}=
\text{authoritative register state}.
\]

## Failure classes covered

The transaction boundary covers failures from:

- cycle-budget enforcement;
- instruction execution;
- invalid opcodes or register operands;
- reflex correction;
- ledger serialization or persistence;
- trace recording;
- checkpoint restoration validation.

A failure during rollback is elevated as `VM transaction rollback failed`, and execution stops. This is deliberately distinct from an ordinary rejected candidate because recoverability could not be established.

## Validation fixtures

Focused tests establish:

1. ordinary arithmetic remains deterministic;
2. reflex remains explicit and its final state is journalled and traced;
3. a cycle-budget rejection preserves the last committed registers, instruction pointer, cycle count, ledger and trace;
4. a receipt failure after instruction execution restores registers and memory and emits no receipt;
5. an unknown opcode fails closed without state mutation;
6. ledger prefix restoration preserves chain validity;
7. a persistent-ledger write failure removes the in-memory candidate entry.

The reconstructed focused harness passes all 15 VM and ledger tests.

Repository CI remains authoritative for:

- the complete regression suite;
- supported Python versions;
- branch coverage;
- formatting and import ordering;
- static type checking;
- package builds;
- dependency audit;
- CodeQL.

## Compatibility boundary

This change intentionally does not yet:

- expand the four-opcode canonical ISA;
- define a versioned external bytecode envelope;
- isolate hostile bytecode at the operating-system process boundary;
- roll back irreversible network, device or external-service effects;
- define machine-reset versus program-load lifecycle semantics;
- unify every research runtime under one cross-language receipt format;
- provide concurrent multi-VM transactions.

External effects must remain outside the instruction mutation boundary until they use a separate prepare/authorize/commit protocol.

## Next integration sequence

After this transaction core is accepted, the recommended order is:

1. define Canonical ISA v1 and its versioned bytecode envelope;
2. add explicit machine-reset, execution-reset and checkpoint APIs;
3. introduce one cross-component receipt schema;
4. add staged external-effect commands;
5. bind sparse and numerical candidates through typed VM adapters;
6. add end-to-end deterministic replay across Python, C++ and browser components.

## Capability statement

This implementation supports the following defensible claim:

\[
\boxed{
\text{Each canonical VM instruction either commits one complete recorded state transition or preserves the previous authoritative state.}
}
\]

It does not establish production security, general intelligence or unrestricted self-modification.
