# External Engine Host Protocol — Draft 1

Transport: local authenticated IPC (Unix domain socket / Windows named pipe).
Do not expose the protocol on a public TCP listener.

Every envelope contains:

```text
magic = MOAGI_OMEGA
protocolVersion = 1
sessionId
messageSequence
messageType
payloadLength
payload
mac
```

Every state-changing command carries one kernel-issued operation context:

```text
transactionId
parentRevision
deadline
```

The same `transactionId` is returned on every navigation, status, frame, and
semantic callback caused by that operation. A committed document callback also
carries `documentRevision`; callbacks emitted before commit use no committed
revision.

Required messages:

```text
HELLO(engineId, engineVersion, capabilities, protocolVersion)
CREATE_SESSION(configuration)
RECOVER_SESSION(configuration, currentUri, currentOrigin, documentRevision)
NAVIGATE(uri, transactionId, parentRevision, deadline)
RELOAD(transactionId, parentRevision, deadline)
STOP(controlTransactionId, observedRevision, deadline)
STOP_ACK(controlTransactionId, cancelledTransactionId)
SEMANTIC_ACTION(nodeId, action, arguments, transactionId, parentRevision, deadline)
FRAME_NATIVE(transactionId, documentRevision, handleType, handle, dimensions, damage)
FRAME_CPU(transactionId, documentRevision, pixelFormat, dimensions, damage, bytes)
SEMANTIC_SNAPSHOT(transactionId, revision, frameSequence, nodes)
CAPABILITY_REQUEST(requestId, origin, capability, rationale)
CAPABILITY_RESOLUTION(requestId, capability, origin, granted, tokenId, detail)
NAVIGATION_STARTED(uri, transactionId)
NAVIGATION_COMMITTED(uri, origin, title, transactionId, documentRevision)
NAVIGATION_FAILED(uri, reason, transactionId)
STATUS(transactionId, documentRevision, message)
HEARTBEAT(monotonicTime)
CRASH(reason, recoverable)
CLOSE_SESSION
```

Rules:

1. Renderer messages never grant capabilities.
2. Every capability request carries a unique request ID and the active origin.
3. A privileged operation may begin only after the matching kernel resolution.
4. A granted single-use token is validated and consumed by the kernel before the
   resolution is released to the engine host.
5. Capability requests whose origin differs from the current session origin are
   denied.
6. State commands for one session are processed in submission order and carry
   the transaction ID, parent session revision, and deadline issued by the kernel.
7. Every callback caused by a state command carries the same transaction ID.
8. A successful state command advances the session revision once; rejected,
   failed, and cancelled state commands do not advance it.
9. `STOP` is a journaled control-lane command. The host must process it while a
   state command is in flight and must not wait for that command to finish.
10. Successful `STOP` acknowledgement does not advance document revision and
    identifies the transaction it cancelled.
11. Protocol negotiation must complete before `CREATE_SESSION`; a version
    mismatch closes the transport and tears down the supervised process.
12. After a supervised restart, the adapter must negotiate again, verify that
    host identity and capability declarations have not changed, send
    `RECOVER_SESSION` for every live session, and remain unhealthy until all
    recoveries succeed.
13. A stale transport generation cannot publish callbacks after replacement.
14. Hosts advertise only capabilities verified by the packaged CEF
    configuration. Unknown or unverified capabilities default to false.
15. Frame handles carry explicit lifetime and release messages.
16. Message sizes and node counts are bounded before allocation.
17. Unknown message types close the channel.
18. The host process is disposable and restartable by `ProcessSupervisor`.
19. Engine stdout/stderr must be drained or redirected so renderer logging cannot
    block protocol progress.
