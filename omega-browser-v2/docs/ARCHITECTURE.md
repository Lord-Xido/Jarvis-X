# Moagi Ω Browser V2 — Architecture

## Trust boundary

The Java browser kernel is trusted. Web content is not. Native engine hosts are
separate replaceable components and should run in operating-system sandboxes.

```text
User intent
   ↓
Per-session serial command chain
   ↓
BrowserCommand + session parent revision
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

Commands submitted to different sessions may execute concurrently. Commands
inside one session are appended to one completion chain, so engine state cannot
be mutated out of submission order.

## Implemented in this alpha

- Engine-neutral Java interfaces
- Capability-scored engine selection
- Origin-bound, time-bound capability tokens
- Request-correlated engine capability decisions
- Single-use capability consumption before authority is released
- Session-local ordered transaction execution and revision chains
- Append-only transaction state journal
- Browser command validation and commit/failure semantics
- Engine event forwarding through Java Flow publishers
- CPU and native frame-surface contracts
- Normalized semantic-scene snapshots
- Restartable process supervision with continuous output draining
- Deterministic mock engine and spatial inspector UI
- Self-test covering ordering, revision isolation, capability consumption,
  rejection, journal persistence, viewport validation, and launch cleanup

## Capability mediation

A renderer cannot grant itself authority. It emits a request containing a unique
`requestId`, active origin, capability, and rationale. The kernel then:

1. verifies that the request origin equals the session's current origin;
2. asks the central `CapabilityBroker` for a scoped token;
3. validates and consumes a single-use token;
4. returns an explicit granted or denied `CapabilityResolution` to the engine;
5. publishes the decision for audit and UI presentation.

Unknown request ids and mismatched origins are denied. An adapter must not begin
a privileged operation until the matching resolution has been received.

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
AND ParentRevision == CurrentSessionRevision
AND PreviousSessionCommandCompleted
```

A successful command advances its session revision exactly once. Rejected and
failed commands do not advance it. Stale parent revisions are rejected rather
than silently committed. Optimistic conflict retry is not yet implemented.

## Process supervision invariant

The supervisor owns one process identity per configured engine host. Initial
launch failure removes the registration so a clean retry is possible. Combined
stdout/stderr is continuously drained to prevent a native engine from blocking
on a full operating-system pipe. Restart is bounded and cannot occur after the
entry has been stopped or removed.

## Security non-claims

The mock engine is not an HTML/CSS/JavaScript engine and does not browse the
public web. The project is a verifiable kernel alpha and adapter foundation.
Production security still requires OS sandbox profiles, audited IPC, site
isolation inside the compatibility engine, signed update infrastructure,
certificate UI, storage partitioning, fuzzing and Web Platform Tests.
