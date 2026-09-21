# Jarvis-X System Operationalisation

This document defines the repository-level operational boundary for Jarvis-X.

## Canonical operational command

After installing the package, the single entrypoint is:

```bash
jarvisx-operationalize doctor
jarvisx-operationalize smoke
jarvisx-operationalize serve --host 0.0.0.0 --port 10000
```

`doctor` verifies that the canonical Python control-plane surfaces are importable.
`smoke` performs bounded deterministic execution across the transactional VM,
Dr Moagi OS kernel, verification journal, and FastAPI control plane. It does not
claim external deployment, hardware acceleration, or unmeasured performance.
`serve` launches the authoritative bounded Dr Moagi OS HTTP control plane.

## Container deployment

The repository root `Dockerfile` is the canonical production image:

```bash
docker build -t jarvisx .
docker run --rm -p 10000:10000 jarvisx
```

For persistent state and hardened local orchestration:

```bash
docker compose -f compose.dr-moagi-os.yml up --build
```

The service exposes `/healthz` on port 10000 and runs as an unprivileged user.
The Compose profile drops Linux capabilities, enables no-new-privileges, makes
the root filesystem read-only, and persists only the explicit state volume.

## Entire-system CI receipt

`.github/workflows/system-operational.yml` is the repository-wide operational
gate. It verifies representative supported surfaces:

1. Python transactional runtime, sparse 3D OS, and HTTP control plane.
2. JavaScript runtime-fabric contract and deterministic application tests.
3. Java platform compilation and assertions.
4. .NET graphics-codec build and round-trip self-test.
5. C++ runtime configuration, compilation, and CTest suite.
6. Root production-image build and live HTTP health check.

The final `Entire system operational` job succeeds only when all six surfaces
succeed. Language-specific workflows remain authoritative for deeper matrices
such as sanitizers, Windows builds, CodeQL, empirical validation, and specialized
engine tests.

## Authority boundary

Operationalisation means the checked software paths are executable,
deterministic where specified, bounded by their configured resource/capability
policies, and verified by CI. It does not convert logical address spaces into
physical allocation, make experimental engines authoritative, establish
scientific validity, or substantiate performance beyond measured receipts.

The governing runtime invariant remains:

```text
propose -> bound -> execute provisionally -> verify -> commit | rollback -> audit
```
