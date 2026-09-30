# Streaming INR VM under a 1 GiB process ceiling

This runtime turns the conceptual ultra-resolution 3D tensor domain into a coordinate-query machine. The domain is never represented as a dense voxel array. Queries are normalized, Fourier-projected, passed through compact implicit neural representation (INR) layers, fused with bounded temporal state, entropy-gated, and returned as feature vectors.

## Memory contract

The original 400 MB + 400 MB + 233 MB sketch totals 1,033 MB before VM stacks, allocator metadata, code pages, and host/runtime overhead. This implementation therefore uses a strict binary-memory contract:

| Arena | Ceiling |
|---|---:|
| Active INR / Fourier weights | 384 MiB |
| Working activations / query cache | 352 MiB |
| Omega temporal state | 192 MiB |
| VM bytecode / metadata | 32 MiB |
| **Managed subtotal** | **960 MiB** |
| Host/runtime headroom | **64 MiB** |
| **Process design ceiling** | **1024 MiB** |

BudgetTracker rejects arena overcommit. The 1024 MiB figure is a design ceiling; exact process RSS also depends on the C++ runtime, allocator, loader, OS, and optional backends, so production deployment should additionally enforce an OS/container memory limit.

## Virtual geometry

The default axis is 8,000,000 samples (8000K interpreted literally as eight million positions per axis). Its conceptual volume is enormous, but only queried coordinates are evaluated:

    q=(x,y,z,t)
      -> normalized q in [0,1]^3
      -> Bq
      -> sin(Bq)
      -> MLP
      -> Omega fusion
      -> entropy gate
      -> f(q)

This is materialization-free voxel evaluation, not a claim that the machine physically stores or iterates the full domain.

## ISA

| Opcode | Mnemonic | Semantics |
|---|---|---|
| 0x01 | LD_COORD | Normalize the spatial query into [0,1]^3. |
| 0x02 | FF_TRANS | Multiply by the Fourier feature matrix. |
| 0x03 | SIN_ACT | Apply elementwise sine activation. |
| 0x04 | MAT_MUL | Apply one dense INR layer. |
| 0x05 | HOLO_FUSE | Mix current features with bounded Omega temporal state. |
| 0x06 | BND_CHECK | Compute normalized information entropy and gate low-information output. |
| 0x07 | RET_VAL | Emit the resolved feature vector. |
| 0xFF | HALT | Terminate the current query program. |

Default bytecode:

    LD_COORD
    FF_TRANS
    SIN_ACT
    MAT_MUL
    SIN_ACT
    MAT_MUL
    HOLO_FUSE
    BND_CHECK
    RET_VAL
    HALT

## Kinetic interpretation

A spatial-temporal query is the kinetic trigger. The VM does not sweep the full field. Each active query follows a path through the implicit field:

    q_t -> phi_B(q_t) -> h1 -> h2 -> h2 (+) Omega_t -> f_t

The Omega store is a bounded circular phase-history buffer. The current reference kernel keeps that ring resident and deterministic; an asynchronous cold-state sink can be attached as a separate persistence/backend layer without changing the ISA.

## Build

From the repository root:

    cmake -S cpp_runtime -B build/cpp-runtime
    cmake --build build/cpp-runtime --target jarvisx-streaming-inr-vm jarvisx-streaming-inr-vm-tests
    ctest --test-dir build/cpp-runtime -R streaming-inr-vm --output-on-failure

Example:

    ./build/cpp-runtime/jarvisx-streaming-inr-vm --queries 256 --axis 8000000 --cutoff 0.05

The runtime reports the virtual axis, number of evaluated queries, number emitted after entropy gating, managed bytes claimed, and a deterministic output checksum.
