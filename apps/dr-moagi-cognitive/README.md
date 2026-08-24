# Dr Moagi Cognitive Control Plane

This browser runtime turns the supplied DM-vΩΞ⁺ HTML concept into a bounded control plane that separates visual metaphor from measured runtime state.

## Operational mapping

```text
Ψ = current particle state sampled from the 3D field
Φ = bounded inward/outward field transform
Λ = validation / promotion gate
Ω = exponentially weighted fixed-point residual memory
Θ = control parameters (mode, field coupling, beta shift, density)
```

The local state invariant is:

```text
PROVISIONAL != AUTHORITATIVE
```

A chat command or IDE patch is parsed into a candidate configuration, validated against explicit bounds, and only then promoted.

## Measured telemetry

The HUD reports:

- actual browser FPS over a rolling measurement window;
- normalized state residual `ΔΨ = ||Ψ(t+1)-Ψ(t)|| / (||Ψ(t)|| + ε)` from sampled particle positions;
- an `Ω`-style exponentially weighted residual memory;
- explicit Q16.48 register encodings;
- a dimensionless saturation ratio;
- local promotion-gate status.

The field-coupling value is a dimensionless visualization/control gain. It is **not** reported in TV/m or any other physical EM unit because the browser is not measuring an electromagnetic field.

## Bounded parameter IDE

The IDE intentionally does not execute arbitrary Python or JavaScript. It accepts only:

```text
mode = INWARD | OUTWARD
coupling = 0..4
beta = 0.5..2
density = 0..100
```

Unsupported identifiers and out-of-range values fail closed.

## Command surface

Local commands include:

```text
inward
outward
coupling 1.8
beta 1.1
density 30
/status
```

An optional verified-firmware API can be configured in the Registers tab or with:

```text
/connect https://firmware-api.example
```

When connected, these commands proxy to the existing firmware service:

```text
/verify
/boot
/run 4
/status
```

No Gemini/OpenAI/API credential is embedded in the static page. Voice output uses the browser `speechSynthesis` API when enabled.

### Cross-origin note

A hosted static page can call a firmware API only when that API is reachable over a browser-compatible origin and explicitly allows the page's origin. The control plane does not bypass CORS, TLS, or verified-boot requirements.

## Tests

```bash
node --test apps/dr-moagi-cognitive/test_core.mjs
```

The tests cover Q16.48 conversion, command parsing, bounded patch compilation, residual measurement, residual memory, deterministic field transforms, and dimensionless saturation semantics.
