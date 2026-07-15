# Moagi Ω Browser V2 Operations

## Runtime contract

The repository currently ships the verified Java 21 control-plane kernel and a
deterministic mock engine. Native Chromium/JCEF and Servo processes are adapters,
not trusted kernel components.

Commands are ordered per browser session. Each committed command advances only
that session's revision. Commands in different sessions may run concurrently.

## Local validation

```bash
cd omega-browser-v2
./test.sh
./build.sh
java -jar dist/moagi-omega-browser-v2.jar
```

The self-test must pass before packaging. It verifies:

- semantic snapshot generation and origin normalization;
- deterministic per-session transaction ordering;
- session-local revision isolation;
- semantic-action commit behavior;
- explicit and engine-originated capability decisions;
- single-use capability-token consumption;
- forbidden URI-scheme rejection;
- positive viewport validation;
- append-only journal persistence;
- native process launch-failure cleanup.

## CI contract

`omega-browser-v2-ci.yml` performs the following on every browser-subsystem pull
request and push:

1. installs Temurin JDK 21;
2. runs the kernel self-test with assertions enabled;
3. compiles the production sources;
4. packages the executable JAR;
5. records a SHA-256 digest;
6. uploads the JAR and checksum as a workflow artifact.

## Release contract

Tag a commit with a name matching:

```text
omega-browser-v2-vMAJOR.MINOR.PATCH
```

The release workflow rebuilds from source, reruns tests, emits checksums, uploads
the artifact, and creates a GitHub release using the tag.

## Security boundary

- Browser commands pass through `BrowserKernel` transaction validation.
- Commands for one session execute through a single ordered completion chain.
- Session revisions are independent and stale parent revisions are rejected.
- Capabilities are origin-bound and time-bound.
- Renderer capability requests carry request ids and receive explicit kernel
  resolutions.
- Single-use capability tokens are consumed by the kernel before approval is
  released to the renderer.
- Native engines must execute out of process.
- Native process output is continuously drained to prevent pipe saturation.
- Renderer failures must be surfaced as engine events rather than terminating
  the kernel.
- No engine adapter may bypass the capability broker or transaction journal.

## Adapter readiness gates

A real engine adapter is mergeable only when it provides:

- process launch, output handling, and supervised bounded restart;
- versioned IPC negotiation;
- navigation and cancellation;
- ordered transaction acknowledgement with session revision metadata;
- semantic snapshot delivery;
- frame-surface delivery;
- request-correlated capability forwarding and resolution handling;
- crash isolation;
- deterministic adapter integration tests.

## Rollback

The subsystem is isolated under `omega-browser-v2/`. Reverting its merge commit
removes the Java browser runtime without changing the existing Jarvis-X Python/C
runtime.
