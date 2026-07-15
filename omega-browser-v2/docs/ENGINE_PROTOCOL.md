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

Required messages:

```text
HELLO(engineId, engineVersion, capabilities, protocolVersion)
CREATE_SESSION(configuration)
NAVIGATE(uri, transactionId, parentRevision)
RELOAD(transactionId, parentRevision)
STOP(controlTransactionId, observedRevision)
STOP_ACK(controlTransactionId, cancelledOperationId)
SEMANTIC_ACTION(nodeId, action, arguments, transactionId, parentRevision)
FRAME_NATIVE(handleType, handle, dimensions, damage)
FRAME_CPU(pixelFormat, dimensions, damage, bytes)
SEMANTIC_SNAPSHOT(revision, frameSequence, nodes)
CAPABILITY_REQUEST(requestId, origin, capability, rationale)
CAPABILITY_RESOLUTION(requestId, capability, origin, granted, tokenId, detail)
NAVIGATION_STARTED(uri, operationId)
NAVIGATION_COMMITTED(uri, origin, title, operationId)
NAVIGATION_FAILED(uri, reason, operationId)
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
   the parent session revision observed by the kernel.
7. A successful state command advances the session revision once; rejected,
   failed, and cancelled state commands do not advance it.
8. `STOP` is a journaled control-lane command. The host must process it while a
   state command is in flight and must not wait for that command to finish.
9. Successful `STOP` acknowledgement does not advance document revision.
10. Protocol negotiation must complete before `CREATE_SESSION`; a version
    mismatch closes the transport and tears down the supervised process.
11. Hosts advertise only capabilities verified by the packaged CEF
    configuration. Unknown or unverified capabilities default to false.
12. Frame handles carry explicit lifetime and release messages.
13. Message sizes and node counts are bounded before allocation.
14. Unknown message types close the channel.
15. The host process is disposable and restartable by `ProcessSupervisor`.
16. Engine stdout/stderr must be drained or redirected so renderer logging cannot
    block protocol progress.
