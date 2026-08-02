# DMVX-1 Operational 1000³ Matrix Runtime

This static browser application operationalizes the bounded matrix extension in `reference/dmvx_1/MATRIX_1000KB3.md`.

## Implemented

- deterministic sparse logical domain `M = {0, ..., 999}³` with at most 4,096 resident cells;
- procedural observed field and 64-value latent encoder;
- Q16.16 latent quantization;
- decoder, reconstruction RMSE, finite/bounds/budget validation;
- candidate isolation with atomic commit or rollback;
- leaky residual memory;
- inward active-set reduction;
- measured codec latency, throughput, frame latency, resident bytes, and logical compression ratio;
- versioned binary ROM image with bounds checks and CRC-32;
- automatic local checkpointing, manual loading, and `.bin` export;
- SHA-256 receipt chaining in browser storage;
- deterministic 3D rendering of the committed state.

## Capability boundary

The application is a reference runtime and visualizer. It does not claim physical electromagnetic bytecode, zero latency, a physically resident billion-cell tensor, universal zero-loss reconstruction, or measured `1000^n` acceleration.

## Local use

Serve this directory with any static HTTP server, then open `index.html`. ES modules do not run reliably from `file://` URLs.

## Tests

```bash
node --test test_core.mjs
```
