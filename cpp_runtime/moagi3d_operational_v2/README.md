# Dr Moagi 3D Operational Engine v2.0

**Status:** C++17 lossless 3D spatial predictive codec, deterministic bounded 3D iterative field, 16-value latent *summary*, CLI and offline volumetric viewer. Built and tested on Linux x86_64 (2026-11-04).

**Reference baseline**: the user's v1.0 source from 2026-11-04, preserved separately outside the public GitHub module to avoid disclosing the original source. Its SHA-256 is `4aa29bef45e2f0a72a5d41ed386fe3b22faef0aaa23e28ab988f2af08d545bf5`. The original does **not** compile due to broken string literals, and its float projection is not invertible. v2.0 is a separate implementation, **not** a patch to v1.0.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
python3 tests/test_cli.py build/moagi3d
```

Or: `g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic src/moagi3d.cpp -o moagi3d`

## Start immediately

```bash
./build/moagi3d demo demo/operational
./build/moagi3d inspect demo/operational.m3d
./build/moagi3d decode demo/operational.m3d demo/recovered.raw
cmp demo/operational.raw demo/recovered.raw
./build/moagi3d field demo/operational.m3d demo/field.csv
```

Open `demo/volume_viewer.html` in a browser to inspect the exported 3D field. It is a fully offline viewer created from the demo dataset; it does not run the C++ codec itself.

Encode an arbitrary **raw tensor** with known shape:

```bash
./build/moagi3d encode my_volume.raw my_volume.m3d --dims 64x64x64 --channels 1 --modality 4 --steps 8
./build/moagi3d decode my_volume.m3d restored_volume.raw
cmp my_volume.raw restored_volume.raw
```

The raw input must be in **x-fastest, then y, then z, channel-last** order: `index=(((z*H+y)*W+x)*C + c)`. File length must equal `W*H*D*C` bytes. Each sample is an unsigned 8-bit integer.

## Mathematically operational steps

1. **Spatial causal prediction** (same channel): for voxel `(x,y,z)`, use only existing negative-direction neighbors `X[x-1,y,z,c]`, `X[x,y-1,z,c]`, `X[x,y,z-1,c]`, ignoring missing neighbors. Their integer mean is `P`, or zero if no neighbors exist.
2. **Lossless residual**: `e[x,y,z,c] = X[x,y,z,c] XOR P`. Optionally run-length encode zero runs; choose the *smaller* of raw residual and zero-RLE. A 128-byte header stores shape, modality tag, iteration count, and non-keyed checksums.
3. **Reconstruct causally**: in the same x-fastest, then y, then z order, compute `P` from **previously reconstructed** neighbors and `Xhat = e XOR P`. Induction over voxel order proves `Xhat==X` exactly if the archive is intact.
4. **3D field**: each voxel starts with the channel-average `U[v] = floor(sum_c(257 * X[v,c]) / C)`, so `0<=U<=65535`. Initialize `F_0=U`. Apply `steps` iterations: `F_(t+1)[v] = floor((35*F_t[v] + 45*floor(sum_6(F_t[neighbor])/6) + 20*U[v])/100)`. Absent neighbors contribute zero. This is a real six-face 3D lattice coupling in deterministic integer arithmetic. `0<=F_t<=65535`; the ideal unrounded map has `L_inf` Lipschitz factor at most 0.80 with fixed `U`. Integer truncation means exact Banach contraction is **not** claimed for the discrete implementation.
5. **16-value spatial summary**: split voxels into index classes `v mod 16`, store mean `F[v]` per class (zero for empty classes). This is **telemetry**, not sufficient to reconstruct the volume.
6. **Integrity checks**: the decoder recomputes CRC-32 of the original bytes, a CRC-32 of the 3D field serialized as little-endian 32-bit values, the 16-value summary, and the header CRC-32; mismatches reject a corrupt archive. These CRCs are not cryptographic authentication, do not provide confidentiality, and do not prove ownership.

### Archive format (`MOAGI3D2`, v2)

| Byte offset | Length | Content |
| --- | ---: | --- |
| 0 | 8 | ASCII `MOAGI3D2` |
| 8 | 16 | `W,H,D,C` (4 × unsigned 32-bit LE) |
| 24 | 12 | `modality,steps,codec` (3 × unsigned 32-bit LE) |
| 36 | 16 | `raw_bytes,payload_bytes` (2 × unsigned 64-bit LE) |
| 52 | 8 | `input_crc32,field_crc32` (2 × unsigned 32-bit LE) |
| 60 | 64 | `latent16` (16 × unsigned 32-bit LE) |
| 124 | 4 | CRC-32 of header bytes 0..123 |
| 128 | variable | raw XOR residual (`codec=0`) or zero-RLE (`codec=1`) |

For zero-RLE, a command byte in `0..127` means the following `command+1` bytes are literals. A command byte in `128..255` means `(command&127)+1` zeros without following bytes.

### CLI commands

`encode`, `decode`, `inspect`, `field`, `demo`, `selftest`. `--modality` is a **metadata tag only**: `0=text, 1=image, 2=audio, 3=video, 4=3D`. The engine handles raw byte tensors, **not** JPEG/PNG, MP3/WAV, MP4, or PDF container parsing. Media formats must first be decoded to a raw tensor elsewhere. This v2.0 engine is **not a trained ANN or VQ-VAE**, and does not run in the cloud without a separate deployment.

## Limits and tradeoffs

- Maximum input tensor: **64 MiB**, dimensions `1..4096`, channels `1..16`, `0..64` spatial iterations, modality `0..4`; single-process in-memory execution.
- Compression ratio depends on spatial predictability. Some data expands because of the 128-byte header. Zero-RLE is chosen only when it is smaller than raw residual.
- The two-sided six-neighbor field is an auxiliary measurement; compression itself uses a separate *causal three-neighbor* predictor to guarantee exact decoding.
- Archive corruption detection uses CRC-32 rather than an authenticated encryption scheme. Do not trust it against an active attacker.
- Floating-point training, backpropagation, auto-tuning beyond choosing raw/RLE, GPU compute, distributed cloud processing, format-native media ingestion, cryptographic signatures, access control, and self-modifying quines are **not** implemented.
- Code metadata intentionally credits the original author; claims of IP ownership and confidentiality need independent legal/security processes.

## Demonstration run

Example `16x16x16`, `C=1`, `8` field iterations on a generated volume:

- Input: **4,096 bytes**
- RLE payload: **2,034 bytes**
- Total archive: **2,162 bytes** including 128-byte header
- Reconstruction: **exact**, as verified by `cmp`
- Archive reduction: **47.22%** for this particular synthetic input, not a general performance guarantee
- Field summary values and checksums are printed by the CLI.

The test suite includes 45 C++ in-memory round trips and 24 Python black-box command-line round trips, alongside checks that tampering and invalid headers are rejected. Inspect `tests/` and the dedicated GitHub Actions workflow. The original release manifest and prebuilt binary are not part of this source-only integration.

## Optional authenticated encryption (OpenSSL 3.x)

When OpenSSL headers and the crypto library are available, **CMake automatically enables** `seal` and `unseal` via AES-256-GCM. The engine uses a fresh 96-bit nonce from OpenSSL's cryptographic RNG, a 128-bit authentication tag, and a 256-bit key. The envelope is `MOAGISE2` + 12-byte nonce + 16-byte GCM tag + authenticated ciphertext. Magic + nonce are authenticated as additional data.

```bash
# Generate a fresh secret local key. Do NOT commit, upload, or embed this key.
openssl rand -out local_secret_key.bin 32
./build/moagi3d seal demo/operational.m3d protected.m3ds local_secret_key.bin
./build/moagi3d unseal protected.m3ds reopened.m3d local_secret_key.bin
./build/moagi3d decode reopened.m3d recovered.raw
cmp demo/operational.raw recovered.raw
```

**Important:** `seal` uses a shared symmetric key and is **not** a digital authorship signature, license enforcement, or legal IP registration. Key management, secure key deletion and user access control are outside the demo. Key files need restrictive OS permissions. Ciphertext and authentication tags are not evidence of ownership. Do not distribute the secret key in the example package.

To test this additional layer:

```bash
python3 tests/test_secure.py build/moagi3d
```

If building without OpenSSL using the plain `g++` command, the encryption commands are **absent**. For a manual OpenSSL build, add `-DMOAGI_ENABLE_OPENSSL=1 -lcrypto` after supplying the proper include/link paths.
## Jarvis-X repository integration

This module is **isolated** in `cpp_runtime/moagi3d_operational_v2/`; it does not modify Jarvis-X APIs, deployed services, authorization boundaries, or existing codec wire formats. It deliberately does **not** publish the locked original v1.0 source, release binaries, demo raw payloads, or encryption keys. Only the verified v2 reference source, CMake build, tests, and offline viewer are included. The GitHub Actions workflow is path-scoped; CI results must be checked on this pull request before any merge.
