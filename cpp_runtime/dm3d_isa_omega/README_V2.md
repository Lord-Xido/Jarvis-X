# DM-3D-ISA-Ω v2 — self-hosted sparse geometric streaming VM

**Experimental reference software, not a production cloud service or an industrial safety controller.**

The v2 executable expands the original C++20 32-bit bytecode VM with a **fixed, verified 10-opcode pipeline**. The pipeline interpreter dispatches CPU worker jobs. **Every tile is processed through the original DM3D bytecode VM microprogram** before serialization, so the v2 orchestration uses the v1 VM rather than bypassing it.

## Build and test

```bash
cmake -S cpp_runtime/dm3d_isa_omega -B build/dm3d-omega -DCMAKE_BUILD_TYPE=Release
cmake --build build/dm3d-omega -j
ctest --test-dir build/dm3d-omega --output-on-failure
build/dm3d-omega/dm3d_pipeline --e2e /tmp/dm3d-omega-example
```

Requires C++20, GCC or Clang, CMake >=3.18, POSIX Bash and Python 3 for the integration test. CUDA is **optional** and disabled automatically when no CUDA compiler is installed. Run `dm3d_pipeline --cuda-check` to determine whether a usable GPU is available and verify the block-average CUDA kernel against the CPU reference. No CUDA results are claimed without a CUDA-equipped machine.

## CLI

```
dm3d_pipeline --self-test-v2
dm3d_pipeline --e2e EMPTY_OR_NEW_OUTPUT_DIRECTORY
dm3d_pipeline --stream-encode INPUT OUTPUT.dm3ds2 [CPU_THREADS_1_TO_16]
dm3d_pipeline --stream-decode INPUT.dm3ds2 OUTPUT [CPU_THREADS_1_TO_16]
dm3d_pipeline --render INPUT_32768.raw OUTPUT.ppm
dm3d_pipeline --tune
dm3d_pipeline --cuda-check
```

For one-shot end-to-end execution, `--e2e` generates a reproducible 236,493-byte dataset, writes a **40-byte pipeline bytecode program**, re-parses it, and executes the following instructions:

| Opcode | Mnemonic | Behavior |
|---|---|---|
| `0x80` | TRAIN | Learn linear 8->1->8 autoencoder parameters with SGD, quantize Q16.16 |
| `0x81` | TUNE | Compare learned codec kernels and verify byte-for-byte parity; select fastest sampled candidate |
| `0x82` | DISPATCH | Configure bounded CPU worker pool / logical shards |
| `0x83` | STREAM.PACK | Read source in 32³ tiles; execute original 32-bit ISA microprogram on each tile; store verified frames |
| `0x84` | STREAM.UNPACK | Decode frames, validate SHA-256 tile and stream checksums and ordered hash chains |
| `0x85` | STORE.COMMIT | Write sparse tile packets to disk, evict cached data, reopen store and verify all loaded bytes |
| `0x86` | RENDER | Native software ray-cast the first volume tile to portable RGB PPM |
| `0x87` | POLICY.CHECK | Test toy refrigeration command policy in a **sandbox only**; no device connection |
| `0x88` | VERIFY | Validate source == reconstructed file and all hash-chain checkpoints |
| `0x00` | HALT | Terminate pipeline |

The 10 instructions are serialized into little-endian 32-bit words. **Out-of-order or modified pipeline opcode sequences are rejected**, rather than granting arbitrary external I/O powers to untrusted bytecode.

## Streaming file format

`DM3DST02` stream: 8-byte magic, 4-byte version, 8-byte source length, 8-byte tile count, 32-byte source digest, 100-byte fixed-point model, 4-byte logical-shard count; then bounded packets `[uint32 LE length][DM3DTIL2 tile]` followed by aggregate and per-shard SHA-256 checkpoint chains. Each tile packet stores the Morton tile index, actual byte length, codec mode, lengths, original SHA-256 and payload. Encoding strategies: trained latent + XOR residual, latent + zero-RLE residual, original raw bytes, all-zero tile, constant-valued tile, and raw zero-RLE. The encoder keeps whichever payload is smallest. A compressed size below source size is **not guaranteed**, and no claim of state-of-the-art ANN compression is made.

Writes are staged using `.partial` files and published only after verified completion. The decoder rejects bad lengths, invalid tile ordering, extra data, corrupted hashes and malformed compressed streams. **SHA-256 detects accidental or unsophisticated modification; it is not a digital signature and does not authenticate a malicious producer.**

The learned codec is intentionally small. It demonstrates gradient-based weight training and deterministic decoding using stored weights, not modern deep learning, foundation-model inference or training at massive scale.

## Storage and threading

32×32×32 voxel tiles are mapped into Morton-keyed sparse addresses. `DiskWorld` persists encoded tiles in a directory with a model manifest, uses bounded least-recently-used resident storage and verifies cold reads and model consistency. Files are streamed with bounded concurrent batches. Worker shard IDs are logical partitions inside one process; **there is no actual multi-node transport or cloud scheduler**.

## Graphical and safety scope

The renderer integrates voxel opacity along volume rays with local gradient shading. It outputs PPM, not a photorealistic neural renderer. A separate optional CUDA implementation covers only v1's block-average kernel; it is not a GPU-based implementation of the learned codec. The physical-control policy is a demonstration of checking synthetic temperature commands: **do not connect this VM to industrial actuators, clinical cold chains, or live firmware**.

## Validation

The regression suite checks original v1 VM backward compatibility, real assembly-source quine identity, new trainable model, equivalent encode kernels, binary file round-trip for empty and non-tile-aligned inputs, 1-thread vs 4-thread byte-identical archive output, corrupt packet rejection with no output file left behind, cache eviction / cold restart, rendering, sandbox rejection, and v2 opcode sequencing.

**Not implemented or verified:** real GPU training/inference, deployed multi-node networking, arbitrary live device control, cryptographic signing, hardened security boundaries, autonomous source-code modification, production-grade ANN, 3D photorealism, or 10^36-cell physical allocation. A successful CPU test is not evidence that these capabilities exist.

Original `dm3d_isa` CLI remains unchanged and backward compatible.
