# JARVIS-X EM-aware 3D C++17 runtime

Executable 3D autoencoder, toroidal inward refinement, Omega observer, CTR verification, and **assumption-based electromagnetic switching/transfer energy proxy**. This is not an electromagnetic field solver, chip-level emulator, or physical power measurement.

## Architecture

1. Generate a bounded 3D volume `X` from a deterministic synthetic waveform.
2. Encode via 2×2×2 block average to target latent `T` (8:1 element count reduction).
3. Refine latent `Z` from zero using a toroidal six-neighbor recurrence coupled to numerical feedback memory `Omega`: `e=Z-T`, `w=Omega-T`, `e_next=gamma*(0.75*e+0.20*avg_6(e)+0.05*w)`, and `w_next=0.35*w+0.65*e_next`. For `0<gamma<1`, this is contractive in the **joint** infinity norm (up to floating-point rounding).
4. Maintain `Omega` as a coupled feedback state and verify convergence of both latent and memory.
5. Decode by block replication. In default `exact residual` mode, carry per-voxel float residual `X-D(Z)` as a side-channel and reconstruct within floating-point tolerance. **This is not a compressing codec; 1 latent float + 8 residual floats per input block is larger than 8 floats.** With `--no-residual`, reconstruction is lossy.
6. CTR checks latent convergence and reconstructed MSE. Export a JSON execution receipt and optional binary PGM middle-Z slice.

## Build and execute

Inside the repository:

```sh
cmake -S cpp_runtime -B build/em-aware -DJARVISX_BUILD_GL_VISUALIZER=OFF
cmake --build build/em-aware --target jarvisx-em-aware-3d
ctest --test-dir build/em-aware -R em-aware-3d --output-on-failure
./build/em-aware/em_aware/jarvisx-em-aware-3d --edge 16 --iterations 48 \
  --receipt em-aware.json --slice em-aware.pgm
```

Standalone: `c++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow cpp_runtime/em_aware/main.cpp -o jarvisx-em-aware-3d`.

## Estimated energy, never empirical hardware telemetry

The meter counts **abstract algorithm-level** floating operations and bytes. It does not measure true instructions, cache misses, transistor switching, or memory physical transfers. Defaults are **user-adjustable illustrative assumptions**, not calibrated silicon constants:

- `logic_J = FLOPs * toggles_per_op * effective_capacitance_F * Vdd^2`
- `memory_J = modeled_bytes * memory_pJ_per_byte * 1e-12`
- `total_J = logic_J + memory_J`
- `modeled_seconds = FLOPs / (assumed_GFlop/s * 1e9) + bytes / (assumed_GB/s * 1e9)`

The simple time model assumes serialization and ignores memory/cache overlap; physical power requires device characterization and metrology. Actual host wall time is separately measured. No Maxwell field simulation, device thermal simulation, inference training, or empirical acceleration claim is made.

`--voltage`, `--cap-ff`, `--toggles-per-op`, `--mem-pj-per-byte`, `--gflops`, and `--gbps` tune assumptions. `--help` lists bounds.
