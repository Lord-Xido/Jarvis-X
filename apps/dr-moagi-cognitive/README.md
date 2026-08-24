# Dr Moagi DM-vΩΞ⁺ 3D Cognitive Web Application

This app is the production-oriented browser control plane for the Dr Moagi stack. It deliberately separates 3D visualization, measured browser telemetry, local bounded control state, and verified firmware execution.

## Runtime model

```text
Ψ = sampled current 3D state
Φ = deterministic inward/outward transform
Λ = local validation/promotion gate
Ω = exponentially weighted fixed-point residual memory
Θ = bounded control parameters
```

The local invariant remains:

```text
PROVISIONAL != AUTHORITATIVE
```

## What makes this a web application rather than a demo

- deterministic Three.js state field with point inspection and nested Ψ/Φ/Λ/Ω/Θ visual layers;
- measured rolling FPS, normalized state residual and Ω residual-memory telemetry;
- adaptive renderer with LOW/MEDIUM/HIGH profiles and AUTO hysteresis;
- signed Q16.48 display of authoritative control values;
- bounded command surface and fail-closed parameter compiler;
- persistent browser session state and command history;
- validated session import/export and PNG scene snapshots;
- browser-local SHA-256 journal for session-level traceability;
- explicit backend connection state with health polling and request timeouts;
- firmware `/healthz`, `/status`, `/manifest`, `/verify`, `/boot`, `/run` commands;
- installable PWA shell with a service worker and offline reuse of previously cached static/vendor assets;
- responsive desktop/mobile control-plane layout.

The browser-local journal is not a replacement for the externally anchored firmware trace ledger. The browser does not receive signing/encryption keys and cannot bypass verified boot.

## Local command surface

```text
inward
outward
coupling 1.8
beta 1.1
density 30
/status
/pause
/resume
/reset
/snapshot
/export
```

Firmware commands after an explicit connection:

```text
/connect https://firmware-api.example
/health
/manifest
/verify
/boot
/run 4
/disconnect
```

## Bounded parameter IDE

Only these assignments are executable:

```text
mode = INWARD | OUTWARD
coupling = 0..4
beta = 0.5..2
density = 0..100
```

Arbitrary JavaScript/Python and unsupported identifiers are rejected.

## Firmware browser access

The firmware service remains closed to cross-origin browser access by default. To authorize a deployed web origin, configure the server-side allowlist:

```bash
export JARVISX_FIRMWARE_CORS_ORIGINS="https://<owner>.github.io"
```

Use the exact trusted origin in production. Credentials remain disabled in the CORS middleware and signing/encryption keys stay server-side.

## Test

```bash
node --test apps/dr-moagi-cognitive/test_core.mjs
```

The test suite covers fixed-point/Q16.48 math, bounded parsing/compilation, deterministic transforms, residual memory, backend URL validation, adaptive quality decisions, convergence logic, bounded history and versioned session snapshots.

## Static deployment

The shared `Jarvis-X Runtime Pages` workflow publishes this app under:

```text
/dr-moagi-cognitive/
```

The Pages gate validates the numerical core and smoke-checks the browser module, PWA manifest and service worker before deployment.
