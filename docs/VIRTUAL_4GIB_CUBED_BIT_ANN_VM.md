# Jarvis-X Ω³: Sparse Virtual 4GiB × 4GiB × 4GiB Bit-Iteration ANN VM

**Status:** bounded C++17 **software reference**, not a physical 4GiB³ device, analog NAND compute demonstration or general-purpose neural network.

## Scope and exact numbers

Treat "4 GiB per axis" as **2³² virtual voxel positions**, not 4 GiB of physically allocated RAM per axis:

- Coordinate domain: \(\Omega=\{0,\ldots,2^{32}-1\}^3\).
- Virtual voxels: \(2^{96}\). With one bit per voxel, dense packed storage would be **\(2^{93}\) bytes**, approximately 9.90 × 10²⁷ bytes.
- Virtual address: 96-bit interleaved Morton/Z key stored as three 32-bit words. Address extrema are tested without 128-bit compiler extensions.
- Tile geometry: \(32^3=32,768\) bits = **4,096 bytes** each.
- Tiles per axis: \(2^{27}\); full virtual tile count: \(2^{81}\).
- **Only nonzero active tiles are stored**, with a hard limit of 256 resident tiles (CLI caps initial pages at 128). Unallocated tiles read as zero.

This **does not** mean the entire virtual space has been traversed, compressed, reconstructed or allocated. A dense 4 GiB NAND chip is not sufficient to materialize the virtual space.

## Exact inward/outward folding

For every 2×2×2 neighbourhood at level \(l\), select its first bit \(a\) and retain seven XOR details:

\[
a=b_0,\quad d_i=b_i\oplus a\ (1\le i\le7).
\]

At decode, \(b_0=a\) and \(b_i=a\oplus d_i\), so all eight input bits are recovered without numerical approximation.

Apply this recursively:

\[
32^3\to16^3\to8^3\to4^3\to2^3\to1^3,
\]

then unfold in reverse. The anchor and details preserve **exactly 32,768 bits** per tile. This is a reversible transform, not inherently compression. The transformed stream is optionally zero-run encoded and used only when smaller.

## ANN predictor

A tiny **6→8→1** supervised MLP is trained on three synthetic local 3D tile patterns. Its six binary input features come from prior x, y, z and three two-axis diagonal neighbours, all causal in z-y-x scan order. Training uses float gradients; inference uses serialized **fixed-point integer weights and ReLU hidden activations**.

For each voxel:

\[
r_i=x_i\oplus P_\theta(x_{<i}), \qquad
\hat{x}_i=r_i\oplus P_\theta(\hat{x}_{<i}).
\]

Given unchanged predictor parameters, decoder initialization and exact residual, \(\hat{x}_i=x_i\) by induction over voxel order. This is **ANN-guided predictive entropy coding**, *not* an architecture that learns to reconstruct arbitrary data from a smaller latent code without residuals.

## Codec choices and persistence

For each nonzero tile choose the smallest *payload* among:

1. Bit-packed raw tile.
2. Raw tile with zero-run RLE.
3. ANN XOR prediction residual with zero-run RLE.
4. Five-level inward-transformed bitstream with zero-run RLE.

Raw fallback ensures the selected tile payload never exceeds 4,096 bytes. Snapshot metadata, the predictor, indices and checksums cause additional storage overhead.

The binary snapshot includes version, tile geometry, serialized quantized MLP model, model identifier, sorted Morton96 keys, codec modes, lengths, payloads and FNV-1a 64-bit checksums. It rejects invalid tile bounds, corrupt checksums, unknown modes, truncated data, duplicate addresses, trailing bytes and mismatched ANN weights. **FNV-1a is a noncryptographic corruption detector, not integrity authentication or encryption.**

Iteration reads and decodes an active tile, deterministically changes one bit, re-encodes, decodes again and compares **every bit**. This checks exact reconstruction only on tiles actually visited.

## Build and run

~~~bash
cmake -S cpp_runtime -B build/cpp-runtime -DJARVISX_BUILD_GL_VISUALIZER=OFF
cmake --build build/cpp-runtime --parallel --target jarvisx-virtual3d-bit-ann
ctest --test-dir build/cpp-runtime --output-on-failure -R virtual3d-bit-ann-reference

./build/cpp-runtime/jarvisx-virtual3d-bit-ann --iterations 1000 \
  --active-tiles 5 --training-epochs 2 --save state.jx3d

./build/cpp-runtime/jarvisx-virtual3d-bit-ann --load state.jx3d
~~~

Supported CLI: --self-test; --iterations 0..10000; --active-tiles 1..128;
--training-epochs 0..20; --seed 0..2³²-1; --save FILE; --load FILE.

## Locally verified

On the local test environment:

- GCC C++17 -Wall/-Wextra/-Wpedantic/-Wconversion/-Wshadow/-Werror compilation passed.
- Clang C++17 with ASan + UBSan self-test passed.
- CMake/CTest self-test passed.
- A 1,000-iteration test visited and compared **32,768,000 bit positions** across repeated 32³ tile reconstructions; runtime approximately **0.92 seconds** in this one optimized local test, including predictor training.
- After the 1,000 iterations, five active tiles used **11,199 bytes of tile payload**, compared with 20,480 bytes for the same five tiles uncompressed. This excludes model/checkpoint/record metadata.
- The checkpoint was saved and successfully reloaded in a separate process.

Tests use synthetic data; no generalization, universal compressibility, performance SLA, physical NAND efficiency, actual 4GiB memory allocation or hardware energy claims are established.

## Relationship to VNAND PR #499

This is a **separate logical sparse volume implementation**. PR #499 models a different proposed two-plane 3-bit VNAND autoencoder with 28×28 input; the present 2⁹⁶-bit virtual addressing scheme is not a physical extension of that die. Future device-backed work would require a measurable physical memory allocation, specified controller, ECC, NAND read/program/erase mechanics, and per-operation energy accounting.
