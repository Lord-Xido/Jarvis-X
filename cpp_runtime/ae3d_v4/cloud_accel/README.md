# JARVIS-X AE3D Cloud Acceleration v5.0

**Status:** locally verified batched CPU codec and authenticated HTTP server. CUDA code path is implemented, but **CUDA throughput and exact cross-device parity have not been measured** in this CPU-only environment. No cloud instance has been deployed. **1,000× acceleration is a target, not a measured property.**

This is a **source-only, additive extension** of the already merged [JARVIS-X v4 C++17 autoencoder](../README.md). It reuses the exact `JXMOD4` model format and implements compatible `JXVOL4`/`JXPKT4` files without modifying the v4 quine or previously recorded archives.

## Operational geometry

For each `4×4×4` U8 voxel block:

```
V[z,y,x] -> 64 U8 bytes -> FP64 batched [B,64]
  -> tanh(W1*X+b1) [B,32]
  -> tanh(W2*h1+b2) [B,12]
  -> q=round(127*z) int8 [B,12]
  -> D(q/127) [B,64] -> round(255*sigmoid(...)) U8
  -> reshape / unflatten back to V[z,y,x]
```

In exact mode, the encoder retains `residual = original XOR predicted` with v4-compatible raw/RLE coding and recovers `original = prediction XOR residual`. Actual storage may be **larger** than the input in exact mode. The portable format stores only 12 quantized latent bytes per 64-voxel block in lossy mode, plus archive metadata. The shared model has **5,004 real weights/biases**; the `10^36` positions are a virtually addressable coordinate domain, **not** trained physical weights.

### Why this is faster than the original loop

The original scalar C++ `encode_q` calls a full `encode_forward`, which also runs the decoder despite only the 12-dimensional latent being needed. The v5 encoder eliminates that unnecessary decoding work and batches matrix multiplications through PyTorch. This improvement requires no change to the archived model or the C++ v4 source. Batch size is bounded; default 4,096 blocks.

### Supported modes and parity caveat

- `cpu`: FP64 PyTorch / BLAS with v4-compatible half-integer ties-away-from-zero rounding (encoder and decoder) using trained v4 parameters. Verified exact `JXPKT4` *byte equality* with native C++ on tested 16³ and random 8³/16³ inputs, plus supplied 32³ and 64³ synthetic volumes. Byte identity for every possible input and platform is **not** mathematically proven.
- `cuda`: same FP64 Torch code and device selection. **Requires NVIDIA CUDA runtime, compatible GPU and PyTorch CUDA build. Not executed in this environment.** On GPU, floating-point reduction/rounding can change a last byte; assess cross-device C++ parity before using CUDA-generated archives for preservation-critical workloads.
- Training and model version management remain in C++ v4. v5 is a **batched inference/codec service**, not an autonomously retraining neural platform.

## Local CPU build and tests

From the repository root:

```bash
cmake -S cpp_runtime/ae3d_v4 -B build/jx-ae3d-v4 -DCMAKE_BUILD_TYPE=Release
cmake --build build/jx-ae3d-v4 --parallel 2
ctest --test-dir build/jx-ae3d-v4 --output-on-failure
python -m pip install -r cpp_runtime/ae3d_v4/cloud_accel/requirements.txt pytest httpx PyYAML
JX_CPP_BIN="$PWD/build/jx-ae3d-v4/jxae3d" \
  python -m pytest -q cpp_runtime/ae3d_v4/cloud_accel/tests/test_accel.py
```

Generate a checkpoint (do **not** commit private/trained checkpoints):

```bash
./build/jx-ae3d-v4/jxae3d generate /tmp/demo.jxvol 32 9
./build/jx-ae3d-v4/jxae3d train /tmp/demo.jxvol /tmp/model.jxmodel 15 0.004
```

## Authenticated cloud HTTP service

From `cpp_runtime/ae3d_v4/cloud_accel/`:

```bash
export JX_MODEL_PATH=/tmp/model.jxmodel
export JX_API_KEY="$(python -c 'import secrets;print(secrets.token_hex(32))')"
export JX_ACCEL_DEVICE=cpu  # change to cuda only on a provisioned GPU host
uvicorn service:app --host 127.0.0.1 --port 8080 --workers 1
```

Raw binary payloads, no base64 inflation:

| Method | Endpoint | Request | Response |
|---|---|---|---|
| GET | `/healthz` | none | health/model SHA-256/device |
| POST | `/v1/encode?exact=true&ox=0&oy=0&oz=0` | `JXVOL4` bytes | `JXPKT4` bytes |
| POST | `/v1/decode` | `JXPKT4` bytes | `JXVOL4` bytes |
| POST | `/v1/query?x=1&y=2&z=3` | `JXPKT4` bytes | JSON predicted/decoded byte |

All POST operations require `Authorization: Bearer <JX_API_KEY>`. The server refuses weak/missing API keys, limits body size (default 16 MiB), bounds dimensions and virtual origin, checks CRC32 and model fingerprint, and does not accept arbitrary filesystem paths or run external commands. **For internet exposure, put the service behind TLS, ingress authentication/rate limits, network policy and per-tenant isolation.** No public endpoint is created automatically.

Sample request (with the same exported key):

```bash
curl -fsS -H "Authorization: Bearer $JX_API_KEY" \
  --data-binary @/tmp/demo.jxvol \
  "http://127.0.0.1:8080/v1/encode?exact=true" > /tmp/demo.jxpkt
curl -fsS -H "Authorization: Bearer $JX_API_KEY" \
  --data-binary @/tmp/demo.jxpkt \
  "http://127.0.0.1:8080/v1/decode" > /tmp/restored.jxvol
cmp /tmp/demo.jxvol /tmp/restored.jxvol
```

## Container templates (not deployed or build-tested here)

From repository root, build CPU image with:

```bash
docker build -f cpp_runtime/ae3d_v4/cloud_accel/Dockerfile.cpu cpp_runtime/ae3d_v4
```

`compose.yaml` is relative to the `cloud_accel` folder. Set `JX_MODEL_FILE` to an **absolute** read-only checkpoint path and `JX_API_KEY` to a random secret. The CPU service binds localhost only. `Dockerfile.cuda` is an **unverified GPU container template** requiring compatible NVIDIA drivers and container runtime. `k8s-example.yaml` (CPU) and `k8s-gpu-example.yaml` (CUDA GPU request) are **placeholders** for approved image digest, secret and model PVC; it is not a deployable production configuration until those are supplied. Do not publish the API without TLS and load testing.

## Reproducible benchmark

```bash
g++ -std=c++17 -O3 -march=native \
  cpp_runtime/ae3d_v4/cloud_accel/bench_cpp_v4.cpp -o /tmp/bench_cpp_v4
python cpp_runtime/ae3d_v4/cloud_accel/bench.py \
  --model /tmp/model.jxmodel --volume /tmp/demo.jxvol \
  --cpp "$PWD/build/jx-ae3d-v4/jxae3d" \
  --cpp-kernel /tmp/bench_cpp_v4 \
  --threads 1 --repetitions 7 --device cpu --out /tmp/jx_bench.json
```

The C++ **kernel benchmark** explicitly measures the original v4 `encode_q` loop in-process; the Torch kernel benchmark measures batched `encode_blocks` in-process. Both exclude model loading and disk I/O. Report the dimensions and repeat counts; do **not** extrapolate results to a 1,000× gain or to GPU speed.

### Local CPU measurements (5 November 2026)

Device: x86-64 local host, Torch 2.10.0+cpu, one PyTorch thread, C++ `-O3 -march=native`, 7 iterations (median, rerun after half-tie rounding fix):

| Volume | Blocks | Original v4 in-process encoder | v5 batched Torch in-process encoder | Approx. speedup |
|---|---:|---:|---:|---:|
| `32³` | 512 | 1.98 ms | 0.247 ms | 8.0× |
| `64³` | 4,096 | 13.54 ms | 2.95 ms | 4.6× |

Kernel timing is sensitive to CPU load, BLAS scheduling and batch size; previous runs differed. These tests are *not* independent cluster benchmarks. Other hosts and batch sizes will differ. For exact decoded bytes on the tested data, observed discrepancy was zero. The v5 `--device cuda` path was **not benchmarked** because no CUDA device exists in this environment.

## Safety and limits

- 64 MiB voxel size cap for on-disk format; cloud request cap defaults to 16 MiB.
- Raw XOR residuals are required to contain exactly one byte per voxel; malformed residual lengths are rejected before decoding or querying. Adversarial half-integer rounding and residual-length cases are included in the regression suite.
- CRC32 is not authentication. FNV model fingerprints are not digital signatures. Do not use these to establish security against adversarial mutation.
- Model files should be immutable and sourced from trusted storage. API keys never belong in Git history.
- A committed source change requires updating this adapter's parity tests if v4 format or quantization semantics change.
- Avoid treating a single-user synthetic training benchmark as a general-purpose compression or graphics generation benchmark.

## One-shot verification of the packaged local examples

From the extracted `JARVISX_AE3D_Cloud_v5` directory, using a Python environment with the dependencies installed:

```bash
g++ -std=c++17 -O3 src/jx_ae3d.cpp -o jxae3d_v4
g++ -std=c++17 -O3 cloud_accel/bench_cpp_v4.cpp -o bench_cpp_v4
JX_CPP_BIN="$PWD/jxae3d_v4" python -m pytest -q cloud_accel/tests/test_accel.py
python cloud_accel/bench.py --model examples/synthetic_demo.jxmodel \
  --volume examples/synthetic_64.jxvol --cpp "$PWD/jxae3d_v4" \
  --cpp-kernel "$PWD/bench_cpp_v4" --repetitions 7 --threads 1 --out benchmark_local.json
```

The demo model and volume are synthetic and bundled **only** in the user-downloadable ZIP; they are **not** part of the public GitHub PR.
