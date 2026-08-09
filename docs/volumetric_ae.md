# Jarvis-X 3D Volumetric Auto-Encoding/Decoding Runtime

This module operationalizes the 6,400-GiB 3D tensor concept as a **sparse virtual address space**. It does not allocate 6.4 TiB of RAM. Only active payload chunks are materialized.

## Numerical substrate

The default storage cell is 32 bits, matching Q16.16 representation.

- Logical capacity: `6400 * 1024^3 = 6,871,947,673,600` bytes
- Bytes per Q16.16 cell: `4`
- Logical cells: `1,717,986,918,400`
- Minimum cubic side containing those cells: `11,977`
- Allocation mode: sparse virtual

The virtual field is addressed by mapping each active chunk onto a linear cell index and then into `(x, y, z)` coordinates.

## What is operational

`Universal3DAutoEncoder` performs a real reversible transform:

1. split payload bytes into chunks;
2. map each chunk onto the virtual 3D field;
3. compress each chunk with zlib;
4. record chunk size, coordinate, and SHA-256 metadata;
5. serialize a self-contained `.jx3d` latent artifact;
6. decode every chunk;
7. verify per-chunk hashes and the end-to-end payload hash;
8. reject corruption or malformed artifacts.

The implementation is an operational storage/transport codec, not a trained neural network. Compression ratio is measured from the actual output artifact rather than fixed to a simulated value.

## CLI

After installing the package:

```bash
jarvisx-volumetric metrics
jarvisx-volumetric selftest
jarvisx-volumetric encode input.bin state.jx3d
jarvisx-volumetric decode state.jx3d restored.bin
```

The final command verifies that the reconstructed bytes match the hash committed in the artifact.

## HTTP API and live 3D dashboard

Start the service:

```bash
jarvisx-volumetric-api
```

or:

```bash
uvicorn jarvisx.volumetric_api:app --host 0.0.0.0 --port 8090
```

Open `http://127.0.0.1:8090/` in a browser. The packaged Three.js dashboard is served by the API process itself. It continuously performs real encode -> artifact -> decode -> verification cycles and renders measured telemetry from those executions.

Dashboard telemetry includes:

- sparse virtual capacity and logical cube dimensions;
- measured compression ratio from the generated artifact;
- actual artifact byte size and chunk count;
- encode/decode latency;
- committed payload SHA-256;
- end-to-end bit-exact verification state.

The 3D node lattice is a visualization of runtime activity; it does not pretend that all 1.7 trillion logical cells are resident GPU objects.

Endpoints:

- `GET /` — live Three.js operational dashboard
- `GET /health`
- `GET /v1/volumetric/metrics`
- `POST /v1/volumetric/encode` with raw request bytes
- `POST /v1/volumetric/decode` with raw `.jx3d` artifact bytes
- `POST /v1/volumetric/cycle` with raw request bytes for a complete verified encode/decode cycle

The encode endpoint returns the artifact as Base64 plus an execution receipt. The decode endpoint returns the reconstructed bytes and verification headers. The cycle endpoint intentionally omits the artifact body and returns encode/decode receipts plus a final `verified` boolean, making it appropriate for live telemetry.

## Verification invariants

The runtime enforces:

```text
SHA256(decoded_payload) == committed_payload_sha256
len(decoded_payload)    == committed_payload_bytes
SHA256(decoded_chunk_i) == committed_chunk_sha256_i
```

A corrupt or truncated artifact raises `ArtifactError` and does not produce reconstructed output.

## Dashboard provenance

The dashboard is derived from the Universal 3D Bit ANN Matrix visualization and preserves its interactive Three.js lattice, drag rotation, zoom, auto-cycle control, and substrate log presentation. Its former randomized compression/fidelity telemetry has been replaced with values returned by the operational API.
