# JARVIS-X 3D World Web Engine (v1.0)

A functioning self-hosted 3D voxel codec and visualization web app. It includes a Python standard-library HTTP server, C++20 portable native codec, 3D Mandelbulb/torus/sphere/wave generators, SHA-256 integrity validation, 2×2×2 spatial coarse latent grids, lossless XOR residuals, bounded recursive 6-neighbor diffusion and an interactive WebGL2 (Canvas fallback) browser viewer.

**Scope:** Algorithmic spatial compression, not a trained neural network, photorealistic model generator, internet crawler, or trillion-parameter system. The engine can be cloud-hosted with Docker but has not been publicly deployed. Use HTTPS ingress and authentication before exposing it on the open internet.

## Run locally

Python 3.10+ required. No pip dependencies.

```bash
python3 server.py
# Open http://127.0.0.1:8080
```

To change port: `PORT=8090 python3 server.py`. To listen on the VM's network: `HOST=0.0.0.0 PORT=8080 python3 server.py` (deploy only behind a hardened HTTPS reverse proxy with authentication and rate limiting).

## Run as a cloud container

```bash
docker build -t jarvisx-3d-web .
docker run --rm -p 127.0.0.1:8080:8080 jarvisx-3d-web
```

Use a TLS reverse proxy such as Caddy/Nginx or managed ingress to serve the app at a public hostname. **This demo does not include built-in authentication, rate limiting, multi-tenant isolation, or durable storage.** A production deployment must add those protections. Downloads are cached in memory for up to 48 artifacts and disappear after server restart.

## End-to-end mathematical operation

For a uint8 input cube `X[x,y,z]`, dimension `D` (even 16..64):

1. **Encode:** `C[x//2,y//2,z//2] = round(mean(X in each 2×2×2 block))` (integer rounded).
2. **Spatial predictor:** `P[x,y,z] = C[x//2,y//2,z//2]`.
3. **Residual:** lossless `R = X XOR P`. Preview (lossy) omits R.
4. **Compress:** `JX3D = header || zlib(C || R)`.
5. **Decode:** lossless `X' = P XOR R = X`; preview `X' = P`.
6. **Verify:** SHA-256 on reconstructed bytes and MSE against that cycle's input.
7. **Recurse:** `X_(t+1) = round((1-alpha) X' + alpha mean_6_neighbors(X'))`; repeat 1..8 times.

The diffusion update intentionally alters the *next* input volume; the lossless codec is exact for each individual round. SHA-256 is an **integrity checksum**, not proof of authorship, authenticity, or cryptographic key control.

### JX3D v1 packet layout

52-byte big-endian header followed by zlib body:

| Byte offset | Length | Field |
|---:|---:|---|
| 0 | 4 | ASCII `JX3D` |
| 4 | 1 | version = 1 |
| 5 | 1 | mode: 0 lossless, 1 preview |
| 6 | 6 | X,Y,Z dimensions, 3×uint16 big endian |
| 12 | 4 | uint32 coarse grid byte length |
| 16 | 4 | uint32 zlib body byte length |
| 20 | 32 | SHA-256 of reconstructed voxel bytes |
| 52 | rest | zlib(compressed coarse + XOR residuals if lossless) |

Voxel indexing: `(z*dim+y)*dim+x` (x fastest). Cubic 16..64 even dimensions; byte uint8 per voxel; packet input <= 1,500,000 bytes. The codec validates dimensions, decompression length, trailing compressed stream data, and digest.

## HTTP API

| Request | Function |
|---|---|
| `GET /api/health` | Runtime status |
| `POST /api/simulate` | JSON simulation request; returns ledger, decoded volume as base64 and packet ID |
| `POST /api/encode` | Raw uint8 voxel bytes. Headers: `X-JX3D-Dimension: 32`, `X-JX3D-Mode: lossless` |
| `POST /api/decode` | Raw `.jx3d` packet body; returns verified reconstructed volume |
| `GET /api/download/{id}` | Download generated `.jx3d` artifact |

Sample JSON for `/api/simulate`:

```json
{"dimension":32,"geometry":"mandelbulb","seed":7,"cycles":4,"alpha":0.2,"mode":"lossless"}
```

## C++20 native encoder and decoder

Requires compiler, CMake, zlib development libraries, and OpenSSL libcrypto development libraries.

```bash
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/jx3d encode volume.raw 32 lossless output.jx3d
./build/jx3d verify output.jx3d
./build/jx3d decode output.jx3d recovered.raw
```

C++20-generated packets are decodable by Python and vice versa; verified in cross-language testing.

## Validate

```bash
python3 -m unittest discover -s tests -v
node --check static/app.js  # optional syntax check
```

For cloud production: add identity/auth, quotas and rate limiting, object storage, metrics, TLS, workload isolation, GPU support for learned model integration, and deployment-specific monitoring.
