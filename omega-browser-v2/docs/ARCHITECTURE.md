# Moagi Ω Browser V2 — Architecture

## Trust boundary

The Java browser kernel is trusted. Web content is not. Native engine hosts are
separate replaceable components and should run in operating-system sandboxes.

```text
User intent
   ↓
BrowserCommand
   ↓
Transaction journal: CREATED → VALIDATING → AUTHORIZED → EXECUTING
   ↓
Capability broker / origin policy
   ↓
EngineSession
   ↓
Semantic snapshot + frame surface + engine events
   ↓
Transaction journal: COMMITTED / REJECTED / FAILED
```

## Implemented in this alpha

- Engine-neutral Java interfaces
- Capability-scored engine selection
- Origin-bound capability tokens
- Append-only transaction state journal
- Browser command validation and commit/failure semantics
- Engine event forwarding through Java Flow publishers
- CPU and native frame-surface contracts
- Normalized semantic-scene snapshots
- Process-supervisor primitive for external engine hosts
- Deterministic mock engine and spatial inspector UI
- Self-test covering commit, capability grant and rejection paths

## Native engine integration contracts

### Chromium/JCEF

A JCEF adapter should implement `Engine.BrowserEngine` and translate CEF browser,
render, life-span, request and display callbacks into the engine-neutral event
stream. Off-screen rendering should publish native shared surfaces when the
platform permits it, falling back to `CpuFrame` only for compatibility.

### Servo

Servo should run behind a small Rust/C ABI host process. The Java kernel should
control the host through a versioned IPC protocol. FFM can be used for a local
control shim on JDK 22+, but renderer execution should remain out of process.

## Transaction invariant

A command may affect browser state only when:

```text
Valid(command)
AND Authorized(origin, capability)
AND SessionAlive
AND EngineHealthy
AND ParentRevisionCurrent
```

The demonstrator currently serializes commands per transaction execution but
has not yet implemented optimistic revision conflict retries.

## Security non-claims

The mock engine is not an HTML/CSS/JavaScript engine and does not browse the
public web. The project is a verifiable kernel alpha and adapter foundation.
Production security still requires OS sandbox profiles, audited IPC, site
isolation inside the compatibility engine, signed update infrastructure,
certificate UI, storage partitioning, fuzzing and Web Platform Tests.
