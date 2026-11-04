# Jarvis-X PERMEATE 3D ANN Tile Codec — C++20 v1.1

This is an **isolated, opt-in research component** for the native runtime. It is a reconstructed and extended derivative of the supplied 23-page engineering dossier's C++20 neural tile codec. It does **not** replace existing Jarvis-X codecs or change their authorization / commit gates.

## Executed features

- 1,000 × 1,000 × 1,000 logical *byte voxels*, addressed as 100³ possible 10³-voxel tiles (one million possible tiles, **not** 10^15 tiles).
- A 221-parameter shared trainable 3D convolutional autoencoder with 500 uint8 latents per 1000-byte tile.
- Lossy quantized inference, or *exact* lossless reconstruction using XOR residual if predictions are bit-identical at both ends.
- Six disjoint faces of consecutive Chebyshev shells, with boundary clipping and unique visits.
- Real zero-tile omission, sorted sparse record emission in the existing .jxa format, model fingerprint, CRC32.
- Complete million-address **enumeration-only** test. ANN operations occur only on nonzero materialized tiles.

The PERMEATE command generates **deterministic synthetic** tiles, not externally loaded 3D volumes. It does not diffuse information between tiles, retrain during permeation, implement fixed-point Q16.16, or encrypt data. CRC32 and FNV-1a are *not* cryptographic authentication. The sparse store uses an in-memory std::map, not an out-of-core processor. MAC statistics are analytical arithmetic, not a hardware performance counter. Dense one-billion-voxel runtime has not been benchmarked.

## Build and verify

```sh
cmake -S cpp_runtime/permeate3d -B build/permeate3d -DCMAKE_BUILD_TYPE=Release
cmake --build build/permeate3d --parallel
ctest --test-dir build/permeate3d --output-on-failure
```

The root cpp_runtime/CMakeLists.txt supports optional -DJARVISX_BUILD_PERMEATE_3D=ON and a jarvisx-permeate3d target without upgrading other C++17 targets.

## CLI

```sh
./build/permeate3d/jarvisx-permeate3d selftest
./build/permeate3d/jarvisx-permeate3d train /tmp/model.jxm 500
./build/permeate3d/jarvisx-permeate3d permeate-selftest /tmp/model.jxm /tmp/self.jxa
./build/permeate3d/jarvisx-permeate3d permeate /tmp/model.jxm /tmp/wave.jxa 50 50 50 2 lossless 7
./build/permeate3d/jarvisx-permeate3d permeate-verify /tmp/model.jxm /tmp/wave.jxa
./build/permeate3d/jarvisx-permeate3d probe /tmp/model.jxm /tmp/wave.jxa 500 500 500
```

permeate takes MODEL OUT.jxa CX CY CZ R lossless|lossy [ZERO_MOD]. Coordinates are tile indices 0–99. Optional ZERO_MOD=n blanks tiles whose linear tile ID is divisible by n. The writer refuses overwrite and uses a .partial temporary path.

## Reconstruction identity and limitations

Define Q=quantize(encode(Model,X)), P=predict(Model,Q), R=X XOR P. Then P XOR R = X for matching predictions. The residual is side information, so there is **no universal compression guarantee**. Quantized inference without residual is lossy.

The shell traversal changes tile **order**, not cross-tile coupling or recurrent dynamics. Future work: real 3D input, out-of-core scheduling, authenticated encryption, 64-bit indexing with overflow-safe statistics, portable fixed-point decoding and measured dense throughput.

Source provenance: PERMEATE v1.1 reconstructed 2026-11-04; source SHA-256 007c69d690d41abf27c1bb9f6951fc66c1f8caffa46481dd351e448b7c63daaf. This is *not* byte-identical to the 510-line source SHA printed in the dossier.
