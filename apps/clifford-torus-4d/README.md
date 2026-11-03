# Clifford Torus 4D — Live C++ → Browser Viewer

This is a **local demonstration** using the C++17 geometry engine from PR #452 and the uploaded Three.js visualizer concept. It is not an ANN/LLM, learned autoencoder, cryptographic runtime, or physical rotor simulation.

## Build

From repository root (CMake 3.16+, C++17 compiler, Node.js 20+):

```sh
cmake -S cpp_runtime -B build/clifford-torus-live \
  -DJARVISX_BUILD_CLIFFORD_TORUS_4D=ON \
  -DJARVISX_BUILD_GL_VISUALIZER=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/clifford-torus-live --target jarvisx-clifford-torus4d --parallel 2
ctest --test-dir build/clifford-torus-live -R clifford-torus-4d --output-on-failure
node --test apps/clifford-torus-4d/test.mjs
node apps/clifford-torus-4d/server.mjs
```

Open **http://127.0.0.1:8787**. A browser with WebGL and CDN access to Three.js is required. To point to another binary, set `CLIFFORD_BIN=/absolute/path/to/jarvisx-clifford-torus4d`. Set `PORT=8788` to change the port. The server deliberately binds only `127.0.0.1`.

## Stream contract

An HTTP `GET /events?time_scale=1000000` starts a bounded C++ child process for that client and streams **Server-Sent Events**. A frame has protocol identifier `jx-clifford-frame-v1`, frame number, simulated seconds, scale, `36 × 36` vertex grid, radii, rotation angles, C++ measured **geometry-compute** seconds and vertices/second, max 4D norm-squared error, matrix orthogonality error, minimum perspective denominator, and a flattened `[x,y,z,...]` array.

- The server accepts a scale in `[1, 2,000,000]`; the engine validates its own numeric parameters.
- Stream grid is limited to 64 per side, frames to 1,000,000, and pacing to at most 1,000 milliseconds per frame.
- The browser uses one retained `BufferGeometry`, one `LineSegments` object, and dynamic position-buffer updates rather than rebuilding 72 line objects per frame.
- Pause closes the SSE stream and kills its C++ child; resume opens a new stream at frame zero. Changing speed restarts the stream.
- Each client has its own process; at most two concurrent clients are admitted. Client disconnects terminate the process.
- `/health` reports whether the configured C++ executable exists.

### Mathematical model

```text
p(u,v) = (R1 cos u, R1 sin u, R2 cos v, R2 sin v)
p'     = R_YW(beta) R_XW(alpha) p         (ZW disabled by default)
q      = projection_radius * (p'.x,p'.y,p'.z) / (projection_radius - p'.w)
```

The projection radius must be strictly greater than `sqrt(R1²+R2²)`. This mapping is a pole-safe **perspective projection** of the 4D torus, not the classical unit-S³ stereographic projection.

The C++ engine measures its geometry-compute time for each frame and transmits the result. It **does not** measure overall application latency, network throughput, or GPU compute. Browser FPS is separately measured by animation frames. The simulated-time multiplier only changes rotation phase per update; it cannot substantiate a computation speedup. The original illustrative spin, energy and stability gauges have been replaced with calculated invariants.

## Run direct backend protocol

```sh
./build/clifford-torus-live/jarvisx-clifford-torus4d \
  --stream-json --frames 3 --grid 8 --time-scale 1000000 --pace-ms 0
```

Each output line is one valid JSON object. Node's SSE bridge wraps these lines with `data:` framing; no external C++ JSON, HTTP or WebSocket libraries are needed.

## Security and limitations

For **local development only**. The Node server spawns a C++ child with fixed arguments and validated speed inputs, uses no shell, bounds simultaneous streams and line sizes, and exposes no public listener. Do not deploy it directly to a public address without authentication, resource controls, and reverse-proxy hardening. The Three.js scripts are currently loaded from third-party CDNs. The browser and Node integration tests do not establish GPU performance or end-to-end frame timing; production performance assessment must collect real render, serialization, transport and compute timings on the target hardware.
