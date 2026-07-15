# Moagi Ω Browser V2 — Architecture

## Trust boundary

The Java browser kernel is trusted. Web content is not. Native engine hosts are
separate replaceable components and should run in operating-system sandboxes.

```text
User intent
   ├── state command ──> per-session serial transaction chain
   │                        ↓
   │                 parent-revision validation
   │                        ↓
   │                 engine session mutation
   │
   └── STOP ─────────> journaled cancellation control lane
                            ↓
                     interrupt in-flight engine work

Both paths
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

Commands submitted to different sessions may execute concurrently. State
commands inside one session are appended to one completion chain, so engine
state cannot be mutated out of submission order. `STOP` is deliberately routed
outside that chain; otherwise it could never cancel the operation blocking the
head of the queue. A successful stop is journaled but does not advance document
revision.

## Implemented in this alpha

- Engine-neutral Java interfaces
- Capability-scored engine selection
- Origin-bound, time-bound capability tokens
- Request-correlated engine capability decisions
- Single-use capability consumption before authority is released
- Session-local ordered transaction execution and revision chains
- Journaled out-of-band cancellation control lane
- Append-only transaction state journal
- Browser command validation and commit/failure semantics
- Engine event forwarding through Java Flow publishers
- CPU and native frame-surface contracts
- Normalized semantic-scene snapshots
- Restartable process supervision with continuous output draining
- Supervised JCEF compatibility adapter contract
- Versioned JCEF host negotiation and decoded host-event protocol
- Deterministic mock engine and spatial inspector UI
- Self-tests covering ordering, revision isolation, cancellation, capability
  consumption, JCEF event translation, rejection, journal persistence, viewport
  validation, crash propagation, shutdown, and process-launch cleanup

## Capability mediation

A renderer cannot grant itself authority. It emits a request containing a unique
`requestId`, active origin, capability, and rationale. The kernel then:

1. verifies that the request origin equals the session's current origin;
2. asks the central `CapabilityBroker` for a scoped token;
3. validates and consumes a single-use token;
4. returns an explicit granted or denied `CapabilityResolution` to the engine;
5. publishes the decision for audit and UI presentation.

Unknown request IDs and mismatched origins are denied. An adapter must not begin
a privileged operation until the matching resolution has been received.

## Native engine integration contracts

### Chromium/JCEF

`JcefBrowserEngine` implements the Java adapter boundary while keeping JCEF and
CEF classes out of the kernel JAR. It negotiates `JcefHostProtocol.VERSION` with
a separately packaged host and translates decoded host messages into
engine-neutral events.

The adapter supports:

- supervised process launch and teardown;
- handshake and command deadlines;
- create, navigate, reload, stop, semantic action, snapshot, and close;
- frame and semantic-snapshot delivery;
- request-correlated capability mediation;
- recoverable renderer-crash propagation;
- explicit capability declarations from the host.

The current repository does not bundle a Chromium binary or an IPC wire codec.
The platform host must use authenticated local IPC, preferably Unix-domain
sockets or Windows named pipes. Off-screen rendering should publish native shared
surfaces only where verified, falling back to `CpuFrame` for compatibility.

### Servo

Servo should run behind a small Rust/C ABI host process. The Java kernel should
control the host through the same versioned IPC principles. FFM can be used for
a local control shim on JDK 22+, but renderer execution should remain out of
process.

## Transaction invariant

A state command may affect browser state only when:

```text
Valid(command)
AND Authorized(origin, capability)
AND SessionAlive
AND EngineHealthy
AND ParentRevision == CurrentSessionRevision
AND PreviousSessionStateCommandCompleted
```

A successful state command advances its session revision exactly once. Rejected,
failed, and cancelled state commands do not advance it. Stale parent revisions
are rejected rather than silently committed. Optimistic conflict retry is not
yet implemented.

A stop command follows a separate invariant:

```text
Valid(STOP)
AND SessionAlive
AND EngineHealthy
AND JournaledControlExecution
AND NoDocumentRevisionAdvance
```

## Process supervision invariant

The supervisor owns one process identity per configured engine host. Initial
launch failure removes the registration so a clean retry is possible. Combined
stdout/stderr is continuously drained to prevent a native engine from blocking
on a full operating-system pipe. Restart is bounded and cannot occur after the
entry has been stopped or removed. Failed JCEF protocol negotiation closes the
transport and removes the supervised host.

## Security non-claims

The mock engine and in-memory JCEF contract transport are not HTML/CSS/JavaScript
engines and do not browse the public web. The project now contains a verified
Java compatibility-adapter boundary, not a bundled production Chromium runtime.
Production security still requires signed native packages, OS sandbox profiles,
audited IPC codecs, verified site isolation, certificate UI, storage
partitioning, fuzzing, and Web Platform Tests.
