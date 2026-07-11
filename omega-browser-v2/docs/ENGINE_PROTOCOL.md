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
NAVIGATE(uri, transactionId)
RELOAD(transactionId)
STOP(transactionId)
SEMANTIC_ACTION(nodeId, action, arguments, transactionId)
FRAME_NATIVE(handleType, handle, dimensions, damage)
FRAME_CPU(pixelFormat, dimensions, damage, bytes)
SEMANTIC_SNAPSHOT(revision, frameSequence, nodes)
CAPABILITY_REQUEST(origin, capability, rationale)
NAVIGATION_COMMITTED(uri, origin, title)
NAVIGATION_FAILED(uri, reason)
HEARTBEAT(monotonicTime)
CRASH(reason, recoverable)
CLOSE_SESSION
```

Rules:

1. Renderer messages never grant capabilities.
2. Every capability grant is minted by the Java kernel and scoped to an origin.
3. Frame handles carry explicit lifetime and release messages.
4. Message sizes and node counts are bounded before allocation.
5. Unknown message types close the channel.
6. The host process is disposable and restartable by `ProcessSupervisor`.
