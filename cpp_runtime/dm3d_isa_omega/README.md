# Ξ⃝ DM-3D-ISA-Ω — C++20 Sparse Geometric Bytecode Reference VM

A **working reference interpreter** for a proposed 3D geometric bytecode ISA: sparse voxel memory, 120-bit Morton addresses, deterministic latent block encoding, a six-neighbor stencil, exact XOR residual recovery, SHA-256 capsules and hash-chained checkpoints. A separately testable **source quine** emits its own canonical assembly source and reassembles into precisely the same binary instructions.

## Scope and verification

- Tested under Linux with GCC 12+ compatible C++20 and CMake/CTest.
- 32-bit base instruction format: `OP8 | DST5 | A5 | B5 | FLAGS5 | EXT4`; all words serialized in little-endian order. An instruction can include 0–15 more 32-bit words.
- `COORD3` carries X/Y/Z as six 32-bit extension words (two words per 40-bit coordinate). `MORTON3` interleaves into a 120-bit address stored in a 128-bit register.
- Sparse memory: `32³ = 32768` bytes per resident voxel tile. `MORTON(address) >> 15` identifies a tile; lower 15 bits identify its voxel. 40-bit axes make room for up to 2⁴⁰ coordinates per axis, **not** fully allocated data.
- Autoencoder: each 2×2×2 input block becomes one 8-bit averaged latent value: `32768 -> 4096` bytes per tile. Its decoder is deliberately lossy; an XOR residual restores **every original byte**. Optional zero-run RLE is only retained if it reduces residual size.
- Integrity: SHA-256 capsule digest and sequential hash-chain checkpoints. These detect corruption relative to a trusted reference, but are **not** signatures or hardware roots of trust.
- Source quine: `STR.LIT`, `FORMAT.EMIT`, `HALT` are assembled as real instructions. The program prints its own source by substitution of a quoted string, **without reading its code file**; recompilation of the emitted source yields identical bytecode.
- Security: branches cannot target immediate extension words; bounded instruction fuel and sparse tile limits; strict capsule lengths and SHA-256 checks; no network, remote-access or hardware-control opcodes.

**Not included:** an LLM, learned ANN training, accelerated GPU kernels, hardware ISA implementation, digital signatures, arbitrarily large physical storage, compiler for all mnemonics, industrial control enforcement, or any claim of 1000× speedup. The latent transform is a fixed block-average predictor with spatial stencil. It is a meaningful executable scaffold for replacing this with trained models later.

## Compile and run

```bash
g++ -std=c++20 -O2 -Wall -Wextra -Wpedantic dm3d_isa.cpp -o dm3d_isa
./dm3d_isa --self-test
./dm3d_isa --demo --out demo
./dm3d_isa --quine
./dm3d_isa --bench 32
./dm3d_isa --encode demo.recovered.raw rebuilt.dm3d
./dm3d_isa --decode rebuilt.dm3d reconstructed.raw
cmp demo.recovered.raw reconstructed.raw
```

Alternatively:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The code uses the GCC/Clang extension `__uint128_t`; MSVC needs an alternative two-limb 128-bit integer implementation. No external packages are required beyond a C++20 toolchain.

## Binary instruction example

```text
MORTON3 A2, C1
OP=0x21, DST=2, A=1, B=0, FLAGS=0, EXT=0
32-bit word: 0x21104000
Little-endian bytes: 00 40 10 21
```

Implemented opcodes:

| Hex | Mnemonic | Operation |
| --- | --- | --- |
| 00 | HALT | Halt execution |
| 20 | COORD3 | Load three 40-bit coordinate components from six extension words |
| 21 | MORTON3 | Encode coordinate into 120-bit Morton address |
| 22 | NEIGHBOR6 | Produce six face-adjacent coordinate registers; trap at global boundaries |
| 24 | TILE.FIND | Look up resident tile without allocating |
| 25 | TILE.ALLOC | Demand-allocate a zero-initialized tile |
| 26 | VOX.LOAD | Read addressed voxel from a resident tile |
| 27 | VOX.STORE | Write addressed voxel within that tile |
| 28 | HALO.EXCHANGE | Construct 16² faces from resident neighbors or clamped boundaries |
| 40 | LAT.ENC | Encode a 32³ tile into 16³ block averages |
| 41 | LAT.VQ | Scalar nearest-step quantization, 16 values spaced 17 apart |
| 43 | LAT.STENCIL6 | Deterministic integer-weighted six-neighbor propagation |
| 45 | LAT.DEC | Expand each latent mean into its original 2³ block |
| 46 | RES.XOR | Compute bytewise residual between original and predictor |
| 47 | RECON.XOR | Reconstruct exactly with the residual |
| 60 | ASSERT.EQ | Trap if two tile buffers differ |
| 61 | HASH | Compute SHA-256, place upper 64 digest bits in scalar register |
| 62 | CHECKPOINT | Verify digest prefix and append SHA-256 hash-chained record |
| 63 | BR.MISSING | Branch by relative word offset when tile handle is null |
| e0 | STR.LIT | Load inline string literal from instruction extension words |
| e1 | FORMAT.EMIT | Substitute `{Q}` in template with its quoted literal, emit string |

Other previously discussed opcodes, including GPU GEMM and local attention, **are not implemented in this reference VM**. Bytecode generation uses a C++ `Builder` API; the textual assembler currently supports the quine subset only.

## Main execution pipeline

```asm
COORD3        C1, 17, 18, 19
MORTON3       A2, C1
TILE.FIND     T0, A2
BR.MISSING    T0, done
LAT.ENC       Z0, T0
HALO.EXCHANGE H1, Z0
LAT.STENCIL6  Z2, Z0, H1
LAT.DEC       T1, Z2
RES.XOR       T2, T0, T1
RECON.XOR     T3, T1, T2
ASSERT.EQ     T3, T0
HASH          R0, T3
CHECKPOINT    T3, R0
done:
HALT
```

Note that this is *illustrative assembly notation*. The actual demo program is built into the reference VM using the `Builder` API, including a patched relative branch. Note also `HALO.EXCHANGE` produces an **H** halo handle, not a latent vector: this is a typed variation on the original paper ISA.

## Self-reproducing assembly source

```asm
STR.LIT S0, "STR.LIT S0, {Q}\nFORMAT.EMIT S0\nHALT\n"
FORMAT.EMIT S0
HALT
```

`FORMAT.EMIT` substitutes `{Q}` with the correctly quoted contents of `S0`. The resulting text is **identical** to the three-line source above. `--quine` verifies both text identity and identical compiled bytecode. No opcode reads its own code address.

## .dm3d capsule format (version 1)

| Offset (bytes) | Size | Meaning |
| ---: | ---: | --- |
| 0 | 8 | ASCII `DM3DCAP1` |
| 8 | 4 | Little-endian version (1) |
| 12 | 16 | Morton tile key in little-endian 128-bit form |
| 28 | 1 | Residual codec, 0 = raw, 1 = zero-run RLE |
| 29 | 4 | Latent payload length, always 4096 |
| 33 | 4 | Residual payload length |
| 37 | 32 | SHA-256 hash of original 32768-byte tile |
| 69 | 4096 | Averaged latent bytes |
| 4165 | variable | Exact residual bytes, raw or RLE |

RLE token with high bit set encodes a zero run of length `(token & 127)+1`. Otherwise the token encodes that many subsequent literal bytes. Decoder rejects malformed lengths and unexpected trailing bytes. This version is **not** an encrypted format.

## Testing and safety limitations

`--self-test` checks SHA-256 known answers, Morton forward/inverse mapping, bytecode word endianness, exact XOR residual recovery, capsule corruption rejection, zero compression, sparse allocation, missing-tile branch behavior, VM state transitions, checkpoint determinism, voxel load/store, source quine round-trip, and rejection of branches into data words.

No device I/O exists in the VM. It is intended for **offline computation, education, and defensive simulation** and must not be connected directly to industrial machinery or clinical cold-chain equipment. For production, additional work is required on signed binaries, capability checks, sandbox isolation, resource quotas, fuzzing, failure handling, and formal verification.

## Initial local demonstration

- 32 active tiles, 1 MiB total materialized data.
- About **83.8 MiB/s** original-byte throughput in one local VM execution (11.9 ms, including interpreter work and checkpoint checks). This is not a reproducible hardware performance guarantee.
- One structured demonstration tile generated an 11,029-byte capsule from 32,768 original bytes. This is input dependent; some inputs expand.
- CTest: all three tests passed.
