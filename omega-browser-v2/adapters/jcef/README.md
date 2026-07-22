# JCEF compatibility adapter

The Java-side compatibility adapter is implemented under
`src/main/java/com/moagi/omega/adapter/jcef/`.

It deliberately contains no JCEF or Chromium classes. Instead, it connects the
trusted browser kernel to a separately packaged, supervised native host through
a versioned and authenticated local transport.

## Implemented Java boundary

- `JcefBrowserEngine` implements `Engine.BrowserEngine`.
- Its nested session implementation provides navigate, reload, stop, semantic
  actions, snapshots, capability resolution, and deterministic shutdown.
- `JcefHostProtocol` defines protocol negotiation and decoded host events.
- Host callbacks are translated into engine-neutral `Engine.Event` records.
- Permission requests carry request IDs and are returned to the host only after
  kernel mediation.
- Process launch can be attached to `ProcessSupervisor`; failed negotiation
  tears down the process registration.
- Command and handshake timeouts prevent an unresponsive native host from
  blocking the trusted kernel indefinitely.
- Recoverable renderer crashes are surfaced without marking the entire host
  adapter unhealthy.

## Packaging boundary

CEF/JCEF binaries, native libraries, sandbox helpers, locales, resources, and
the IPC wire-codec implementation must be packaged separately from the kernel
JAR. The native package must publish:

1. platform and architecture;
2. JCEF and Chromium version;
3. source and binary provenance;
4. checksums and signatures;
5. sandbox configuration;
6. supported protocol version;
7. verified capability declarations.

The host must not advertise native shared surfaces, site isolation, WebGPU, or
other security/performance capabilities unless they are verified by its packaged
configuration.

## Contract validation

`JcefAdapterSelfTest` runs in Java 21 CI using an in-memory host transport. It
verifies:

- protocol negotiation and capability declaration;
- session creation and shutdown;
- navigation commit, frame delivery, and semantic snapshots;
- reload;
- out-of-band cancellation of an in-flight navigation;
- origin-bound capability mediation and single-use token consumption;
- recoverable renderer-crash propagation.

This validates the Java adapter mechanics without bundling or falsely claiming
a production Chromium runtime. The remaining delivery is the platform-specific
JCEF host binary and audited Unix-domain-socket / Windows-named-pipe codec.
