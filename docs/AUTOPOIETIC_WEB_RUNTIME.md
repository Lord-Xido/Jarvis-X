# Experimental 3D Auto-Poietic Web Runtime

## Status

This component is an **experimental observability and bounded-adaptation laboratory**. It is not part of the authoritative `CodexVM` execution path and does not rewrite Jarvis-X source code, policy, bytecode, or governance.

## Integration

The runtime is packaged at:

```text
src/jarvisx/web/autopoietic_runtime.html
```

Mount it in an existing FastAPI application:

```python
from fastapi import FastAPI
from jarvisx.autopoietic_web import mount_autopoietic_runtime

app = FastAPI()
mount_autopoietic_runtime(app)
```

The default endpoints are:

```text
/research/autopoietic-runtime
/research/autopoietic-runtime/manifest
```

The root route is deliberately rejected so an experimental runtime cannot replace the host application entrypoint accidentally.

## Browser bridge

The page exposes a bounded host API:

```javascript
const runtime = window.JarvisXAutopoieticRuntime;

runtime.run();
runtime.pause();
runtime.step();
runtime.seal();
runtime.setInward(true);
runtime.setAutoEvolve(false);
runtime.setModel({ learningRate: 0.02 });
console.log(runtime.snapshot());
```

State changes are emitted as browser events:

```javascript
window.addEventListener("jarvisx:autopoietic-state", event => {
  console.log(event.detail.reason, event.detail.snapshot);
});
```

Protocol identifier:

```text
jarvisx.autopoietic-web.v1
```

## Mechanical boundary

The browser engine may mutate only its bounded numerical model parameters. A candidate change follows:

```text
observe → encode → predict → residual → Ω update → project into Λ → verify → commit/rollback
```

It does not receive filesystem, shell, GitHub, bytecode-policy, or canonical VM mutation privileges.

The mathematical expression editor is still a research feature. It uses a restricted character/token filter before dynamic JavaScript function construction and must not be treated as a strong security sandbox for hostile input.

## Persistence and reproducibility

Committed browser generations and edited source tabs are stored in browser local storage when available. This persistence is local to the browser profile and is not the authoritative Jarvis-X ledger.

For scientific evaluation, export snapshots through the host bridge and record the browser/runtime version, seed policy, workload, frame-independent simulation step, and complete parameter manifest.
