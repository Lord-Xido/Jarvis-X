# Multiparallel VM Reference Visualization

This directory contains an interactive Three.js visualization of the proposed Dr. Moagi multiparallel bytecode and virtual-machine architecture.

## Status

The page is a **visual reference**, not an implementation of the declared VM.

Implemented:

- a Three.js scene with a recursive tensor core;
- 2,400 instanced swarm proxies;
- camera orbit and zoom controls;
- procedural spherical swarm motion;
- illustrative 32-bit instruction telemetry;
- measured frame-time and FPS telemetry.

Not implemented:

- a 32-bit bytecode interpreter;
- `SharedArrayBuffer`-backed VM state;
- Web Worker scheduling or 1,024 physical cores;
- `Atomics` barriers;
- WebAssembly SIMD execution;
- self-modifying or runtime-patched executable code;
- transactional candidate-code verification;
- zero-copy CPU-to-GPU buffer aliasing.

The logical count and opcode displays are architectural targets. The rendered scene materializes 2,400 proxies.

## Run locally

Serve the repository through an HTTP server and open `index.html`. The demo loads Three.js r128 from a CDN, so network access is required.

```bash
python -m http.server 8000
```

Then navigate to:

```text
http://localhost:8000/examples/web/multiparallel-vm/
```

## Controls

- Mouse or touch drag: orbit the camera.
- Mouse wheel: zoom.

## Related Jarvis-X work

This visualization should remain decoupled from runtime claims until it is integrated with:

- the 32-bit ISA specification in PR #16;
- the transactional VM kernel in PR #17;
- deterministic ROM compilation in PR #31;
- spatial propose–verify–commit mechanics in PR #33.

See [`architecture.md`](architecture.md) for the implementation boundary and proposed evolution path.
