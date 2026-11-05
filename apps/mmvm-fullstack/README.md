# Jarvis-X MMVM Full-Stack Runtime

This application operationalizes the Jarvis-X auto-encoding/decoding architecture as a bounded full-stack runtime.

## Operational path

```text
Input bytes
  -> lossless ZLIB packet + 128-D latent Z
  -> Xi iterative refinement
  -> Lambda checksum / finite / bounds / resource projection
  -> exact reconstruction
  -> sparse 3D object allocation
  -> optional text/image/audio/video/3D decoder
  -> Omega event persistence
  -> transactional commit
```

The logical memory lattice is exactly `100000^3 = 10^15` byte-addressable voxels, equivalent to 1,000,000 decimal GB. SQLite stores only resident objects, artifacts and event telemetry.

## Run

From this directory:

```bash
docker compose up --build
```

Open <http://localhost:8080>.

The browser surface connects to the kernel over HTTP/WebSocket and renders real runtime telemetry through a WebGL2 multi-pass bloom pipeline:

```text
Xi-dot / error / Omega / Lambda
  -> scene framebuffer
  -> bright-pass
  -> ping-pong Gaussian blur
  -> bloom composite
  -> tone mapping
  -> viewport
```

## API

- `GET /health` — liveness and scheduler status
- `GET /api/status` — kernel, Omega, Lambda and memory telemetry
- `POST /api/submit` — queue text or base64 binary input and optional generation target
- `POST /api/cycle` — execute one queued task synchronously
- `GET /api/tasks` — recent in-process tasks
- `GET /api/objects/{object_id}` — sparse memory object metadata
- `GET /api/artifacts/{artifact_id}` — generated artifact bytes
- `WS /ws/telemetry` — live kernel telemetry

Supported local generation targets are `text`, `image` (SVG), `audio` (WAV), `video` (H.264 MP4 through FFmpeg), and `3d` (voxel JSON).

This is an application/runtime microkernel, not CPU virtualization or a general-purpose host operating system. It deliberately preserves the deterministic core boundary and keeps generated/research state subordinate to Lambda validation and transactional commit.
