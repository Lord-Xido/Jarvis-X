# DM10Cube — 10 MiB³ Sparse Self-Codec Runtime

This package publishes the Dr. Moagi 10 MiB × 10 MiB × 10 MiB logical 3D runtime in a repository-native reproducible form.

## Representation

The logical coordinate domain is:

```text
[0, 10,485,760)³
= 1,152,921,504,606,846,976,000 logical cells
```

The domain is sparse. The runtime initially materializes 65,536 active voxels and validates every transaction against a fixed 16,384-voxel evaluation lattice. It never allocates the dense logical cube.

## Implemented mechanics

- 4 × 4 × 4 latent-only spatial codec;
- 64 Q16.16 latent values;
- decoder receives latent state and coordinates only;
- mean-baseline-relative reconstruction validation;
- transactional inward reduction with rollback;
- CRC-32 and SHA-256 source-image verification;
- strict source and ROM allocation limits;
- round-trip verification before persistence;
- SHA-256 receipt chaining;
- optional PLY active-field export;
- exact self-source reconstruction.

## Rebuild

```bash
python3 reference/dm10cube/generate_dm10cube.py \
  --output build/dm10cube_10MiB.cpp \
  --print-sha256

g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic \
  build/dm10cube_10MiB.cpp -o build/dm10cube

build/dm10cube \
  --source build/dm10cube_10MiB.cpp \
  --rom build/dm10cube.rom \
  --reconstructed build/reconstructed.cpp \
  --ply build/active_field.ply \
  --steps 4 --inward-every 1 --self-test

cmp build/dm10cube_10MiB.cpp build/reconstructed.cpp
```

## Repository transport

The 10 MiB source is generated from an embedded compressed canonical C++ core plus a deterministic inert comment reservoir. CI builds and publishes the full source, ROM, PLY field, and Linux x64 binary as workflow artifacts. This avoids committing large generated blobs while preserving reproducibility.

## Capability boundary

This is a bounded reference runtime. The 10 MiB cube is a virtual address domain, not a dense physical allocation. The system does not claim physical electromagnetic bytecode, zero latency, unlimited compression, or superphysical acceleration.
