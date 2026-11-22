# JX64 AVX2/FMA Volumetric Transformer — v3

Versioned performance/correctness reference. No existing v2 source is overwritten.

## What it actually executes

- A **64×64×64** float32 volumetric field: **262,144** spatial scalar values, not an 8→3→8 autoencoder.
- Original field retained independently; encoder uses a rational approximation to tanh with fixed gain/bias.
- Odd-stride modular index permutation (bijective modulo 2^18) using AVX2 gather.
- Eight default elementwise recurrent feedback iterations (not spatial iterations).
- One 6-neighbour spatial diffusion pass with replicated/clamped boundaries.
- One 27-point decoder or a separable three-pass mathematical equivalent.
- Reconstruction MSE versus *original*, FNV-1a hashes over floating-point bit patterns, and **256-bin empirical** Shannon entropy over the propagated field.

**Not a lossless codec, and not a trained ANN.** `original` is preserved for evaluating loss, but the decoded value differs from it. A true lossless implementation would preserve quantized residual data. `entropy_bits` is a quantized histogram estimate; it is not entropy of an unnormalized float vector.

## SIMD corrections compared with the supplied draft

1. `_mm256_load_ps(src+c)` with x=1 is 4-byte offset, not guaranteed 32-byte aligned. Unaligned-safe `_mm256_loadu_ps` used for sliding neighbourhood loads.
2. Vector blocks cover x=1..56; scalar x=0 and x=57..63 completes the row. The original x<=54 loop did not process six interior columns x=57..62.
3. All faces, edges, and corners are evaluated using *clamped* neighbour coordinates (no stale buffer values).
4. AVX2 and scalar modes use separate floating-point instruction paths. They agree within tested numerical tolerance but **not bitwise**; FMA and reduction order change rounding.
5. Reentrant state reset, aligned `operator[]` implementation, actual MSE versus original, empirical entropy histogram and full-field output hash.
6. The decoder does not accidentally leave its entire first z=0 plane as the seeded data; the output hash covers all 262,144 values rather than only summing the first 1,024.
7. `std::tanh` is **not** the implemented function: saturation at ±3 plus rational `x(27+x²)/(27+9x²)` is an approximation.

## Build / test

Requires Linux or another x86-64 toolchain with **AVX2 and FMA** CPU feature support; this code **does not run on the ARM-based Apple A17 Pro**. To run on iPhone, port to Metal or ARM NEON.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
./build/jx64avx2 scalar27 11
./build/jx64avx2 simd27 11
./build/jx64avx2 separable 11
```

Reproducing across machines may yield slightly different floating-point bit patterns. Treat only CPU-ISA-normalized execution and explicitly specified operation ordering as candidates for strict bitwise reproducibility.

## Tested numerical results

| Mode | Median 11 runs | MSE against original |
|:--|--:|--:|
| Scalar 27-point | 8.380 ms | 0.381690949372 |
| AVX2 + FMA 27-point | 3.455 ms | 0.381690950083 |
| AVX2 + FMA + separable decoder | 3.699 ms | 0.381690957689 |

All stages are timed together, on Intel Xeon Platinum 8573C in the provided CPU environment. No CPU pinning, turbo or cold-cache controls, so these are indicative, not sustained guarantees. The AVX2 implementation achieved approximately 2.43× speedup over the scalar implementation in this specific benchmark. The scalar reference and SIMD decoders had maximum absolute difference 1.78814e−7 over 20 seeded cases (5,242,880 voxel comparisons). Separable max deviation 2.38419e−7. Full-frame boundary/constant preservation, modular permutation, reentrant execution and address sanitizer + UBSan tested.

## Security / safety boundary

No external files or networks accessed. Arithmetic is bounded by fixed geometry. This is a performance reference, not an AVX2 bitstream format or a cryptographically authenticated archive.
