# Moagi Ω Browser V2 — Kernel Alpha

A runnable engine-independent browser-kernel prototype written in Java 21.

This project is the first implementation slice of the beyond-SOTA architecture:
it proves the browser control plane before coupling it to Chromium/JCEF or Servo.

## Run the demonstrator

Linux/macOS:

```bash
./build.sh
java -jar dist/moagi-omega-browser-v2.jar
```

Windows:

```bat
build.bat
java -jar dist\moagi-omega-browser-v2.jar
```

## Run the self-test

```bash
./test.sh
```

The self-test verifies:

- semantic snapshot generation;
- origin normalization;
- semantic-action transaction commit;
- origin-bound capability approval;
- forbidden URI-scheme rejection;
- append-only transaction journal output.

## What the UI demonstrates

- Engine-independent navigation commands
- Deterministic frame delivery
- Normalized semantic-scene inspection
- Spatial depth overlays
- Transaction state transitions
- Explicit capability prompts
- Private-session partition identity
- Engine selection and event routing

## What it is not yet

The bundled mock engine does not parse or execute arbitrary web pages. Native
Chromium/JCEF and Servo binaries are intentionally not bundled. The interfaces,
supervisor and protocol documents are the integration foundation for those
engines.

## Requirements

- JDK 21 or newer
- No external Java dependencies

## Project map

```text
api/                 Stable engine-neutral records and interfaces
core/                Browser kernel, policy, journal and supervisor
engine/mock/         Deterministic test renderer
ui/                  Swing control surface and semantic inspector
docs/                Architecture and external engine IPC contract
native/servo-host/   Draft C ABI for a future Rust Servo host
adapters/jcef/        Chromium/JCEF adapter requirements
```

## Repository operations

When hosted inside Jarvis-X, the subsystem is validated by
`.github/workflows/omega-browser-v2-ci.yml`. Tag releases as
`omega-browser-v2-vMAJOR.MINOR.PATCH` to trigger deterministic rebuild, test,
checksum generation, artifact upload, and GitHub release publication.

See [`docs/OPERATIONS.md`](docs/OPERATIONS.md) for the runtime, adapter,
release, security, and rollback contracts.
