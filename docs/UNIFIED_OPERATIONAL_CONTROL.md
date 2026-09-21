# Unified Jarvis-X Operational Control Plane

Jarvis-X has many runtime surfaces, but they must not become independent sources
of authority. The unified operational control plane provides one bounded system
entry point:

```bash
jarvisx-system status --pretty
jarvisx-system verify --pretty --output operational-receipt.json
```

## Authority chain

The operational model is:

```text
installed Jarvis-X entry points
        |
        v
structural registry + canonical module resolution
        |
        v
SystemRuntime bounded request
        |
        v
capability projection -> isolated VM -> ledger verification
        |
        v
atomic commit | reject | rollback
        |
        v
Dr Moagi sparse 3D OS bounded cycle
        |
        v
ingest -> bit-plane -> inward fold -> codec/distiller
        |
        v
fixed-point/resource verification -> exact transport verification
        |
        v
atomic sparse-state commit
        |
        v
single machine-readable operational receipt
```

The control plane does **not** execute arbitrary host commands, grant new
capabilities, mutate source code, or convert non-authoritative visual/media
surfaces into authoritative state.

## What `status` proves

`status` is a structural deployment check. It discovers every installed console
entry point whose name begins with `jarvisx`, resolves its Python module, and
checks the canonical authority modules required by the system.

This catches packaging drift such as a declared CLI whose module was not shipped
in the wheel.

## What `verify` proves

`verify` includes the structural checks and then executes two bounded runtime
transactions.

### Canonical VM transaction

The probe assembles and executes a deterministic bytecode program through
`SystemRuntime`. A passing receipt requires:

- explicit `vm.execute` capability;
- bounded bytecode validation;
- isolated VM execution;
- VM ledger verification;
- deterministic result verification;
- system audit append;
- committed state hash;
- complete runtime verification.

The smoke program computes `Ω = 7 + 11` and requires the committed value to be
exactly `18`.

### Sparse 3D operating-system transaction

The probe boots an in-memory `DrMoagiOSKernel`, loads the deterministic 3D demo
field, and executes one full bounded cycle. A passing receipt requires:

- the candidate cycle to commit;
- non-empty resulting sparse state;
- exact transport bytes;
- transport/state/theta/journal hashes;
- a valid hash-chain journal.

The probe uses `state_dir=None`, so CI does not persist runtime state.

## CI authority graph

The primary CI workflow now follows:

```text
Quality gate -----\
Python 3.10 -------\
Python 3.11 --------\
Python 3.12 ---------> Unified operational gate -> Build package
Python 3.13 --------/
Dependency audit --/
```

Packaging therefore occurs only after the complete supported Python matrix,
quality checks, dependency audit, and runtime operational receipt have passed.

The built wheel is then smoke-tested again with:

```bash
jarvisx-system status
```

This verifies that the operational entry point survives packaging rather than
existing only in the source checkout.

## Receipt semantics

The receipt is JSON and intentionally separates structural availability from
executed evidence:

- `entrypoints`: all installed Jarvis-X console surfaces and whether their
  modules are present;
- `canonical_modules`: authority-bearing modules required for operation;
- `core_transaction`: deterministic VM commit evidence;
- `sparse_3d_cycle`: bounded 3D transaction evidence;
- `ok`: the final fail-closed operational result.

A green receipt means those stated checks executed successfully. It does not
claim that every possible workload, optional accelerator, external provider, or
hardware target was exercised.
