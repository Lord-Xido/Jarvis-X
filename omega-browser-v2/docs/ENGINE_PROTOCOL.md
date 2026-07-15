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
HELLO(engineId, capabilities)
CREATE_SESSION(configuration)
NAVIGATE(uri, transactionId, parentRevision)
RELOAD(transactionId, parentRevision)
STOP(transactionId, parentRevision)
SEMANTIC_ACTION(nodeId, action, arguments, transactionId, parentRevision)
FRAME_NATIVE(handleType, handle, dimensions, damage)
FRAME_CPU(pixelFormat, dimensions, damage, bytes)
SEMANTIC_SNAPSHOT(revision, frameSequence, nodes)
CAPABILITY_REQUEST(requestId, origin, capability, rationale)
CAPABILITY_RESOLUTION(requestId, capability, origin, granted, tokenId, detail)
NAVIGATION_COMMITTED(uri, origin, title)
NAVIGATION_FAILED(uri, reason)
HEARTBEAT(monotonicTime)
CRASH(reason, recoverable)
CLOSE_SESSION
```

Rules:

1. Renderer messages never grant capabilities.
2. Every capability request carries a unique request id and the active origin.
3. A privileged operation may begin only after the matching kernel resolution.
4. A granted single-use token is validated and consumed by the kernel before the
   resolution is released to the engine host.
5. Capability requests whose origin differs from the current session origin are
   denied.
6. Commands for one session are processed in submission order and carry the
   parent session revision observed by the kernel.
7. A successful command advances the session revision once; rejected and failed
   commands do not advance it.
8. Frame handles carry explicit lifetime and release messages.
9. Message sizes and node counts are bounded before allocation.
10. Unknown message types close the channel.
11. The host process is disposable and restartable by `ProcessSupervisor`.
12. Engine stdout/stderr must be drained or redirected so renderer logging cannot
    block protocol progress.
