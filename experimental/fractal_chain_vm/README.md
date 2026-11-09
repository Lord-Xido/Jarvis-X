# JARVIS-X Ω³ Fractal Chain Bytecode VM (experimental)

Isolated implementation of the supplied 3D recursive Mandelbulb bytecode chain. This module does **not** change the existing fractal_cloud_v2 runtime.

## Key corrections

- Preserve the blended 3D state from OP_BLEND into the following level; do not overwrite it at the next OP_MANDELBULB dispatch.
- Use a stable latent projection basis across successive levels so latent recoil is meaningful.
- Keep voxel recoil velocity inside the runtime object.
- Trace all dispatches, including conditional jumps, using input/output state hashes and measured instruction durations.
- Save versioned little-endian 12-byte instructions and volume data. Abort on unsupported opcodes.
- Check dimension and recursion bounds before allocating volumes.

## Scope

The latent projection is deterministic but untrained, so a small latent payload is not evidence of reconstructive fidelity or useful compression. The power-evolution policy is a heuristic, **not** MSE minimisation or autonomous learning. This is an experimental CPU VM, not a 1000 GB/s implementation or GPU service.

## Build

```sh
cmake -S experimental/fractal_chain_vm -B /tmp/jx-fvm -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/jx-fvm -j2
ctest --test-dir /tmp/jx-fvm --output-on-failure
/tmp/jx-fvm/jx_fractal_chain --dim 32 --levels 5 --iterations 32 --prefix output
```

Outputs: `output.jxbc` (versioned little-endian bytecode), `output.jx3v` (versioned little-endian float32 3D field), and `output.csv` (instruction-by-instruction trace). Host/compiler floating point variation may change bit patterns; measured durations are intentionally excluded from reproducibility comparisons.

## Verification

Three levels execute 31 instruction dispatches (10 per level and one HALT), and the retained blended field becomes the next level's permeation input. Regression tests cover trace continuity, byte formats, local replay parity, and invalid dimension rejection.

No production, self-optimisation, learned ANN, encrypted ISA, or throughput claims.
## Fixed-reference quality experiment

`jx_fractal_quality` measures five recursion depths against the **same** initial Mandelbulb field after four permeation passes, and against a constant-0.5 reconstruction baseline. The CSV reports fixed-reference MSE, baseline MSE, relative error, each runtime's own final-step MSE, and decoded-volume hash. The fixed-reference and internal MSE are different metrics because the runtime evolves its volume. Neither a pass nor a small latent payload establishes monotonic quality improvement, trainability, or a useful codec. The quality probe is a deterministic *measurement gate*, not an optimisation success criterion.

### Quality-evidence gate

The quality experiment now also compares against the **optimal constant-value predictor** (the reference field's mean), not merely a constant-0.5 baseline. Its five-level CSV reports reference MSE, baseline-relative MSE, decoder standard deviation and PSNR. `quality_report.py` checks metric arithmetic, notes whether improvements are monotonic, and emits machine-readable JSON. GitHub Actions prints each level's actual measured results and stores CSV/JSON as 14-day artifacts for both GCC and Clang.

A green CI result confirms that the measurement pipeline runs correctly. It does **not** establish useful compression, trained autoencoding, or improved MSE. The source is a synthetic Mandelbulb volume. Compare candidate optimizers on the same input and include bit cost before claims of a production codec.

### Adaptive spatial codec mode

The bytecode VM retains its default `--codec projection` mode and adds an opt-in, non-neural `--codec spatial` backend. At every ENCODE it evaluates 8-bit block-mean candidates with edge lengths 2, 4 and 8 (when they divide D). The winner minimises the *input-reconstructible* local rate–distortion objective:

`J(B) = MSE(volume, decode(encode_B(volume))) + lambda * (serialized_bytes_B / input_float32_bytes)`.

Run: `jx_fractal_chain --dim 16 --levels 5 --iterations 12 --codec spatial --rd-lambda 0.02 --prefix spatial_demo`. This retains bytecode stages, volumetric recoil, feedback and per-instruction trace while recording selected block size and encoded bytes. The `EVOLVE` parameter-power heuristic is disabled for this mode; adaptation is the codec choice, not a learned power update. `spatial_demo.jx3b` contains versioned shape, block size, quantised bytes and an FNV-32 accidental-corruption checksum (not cryptographic integrity). `spatial_demo.jx3v` contains the VM's final recoil-adjusted output.

On a **synthetic** D=16 Mandelbulb fixed reference (12 fractal iterations, 4 diffusion passes), the independently tested block-mean codec reported GCC MSE 0.0112095 at block=2 (536 total encoded bytes, 30.57:1 versus 16,384 raw float32 bytes); block=4 MSE 0.0223263 (88 bytes, 186.18:1); block=8 MSE 0.0645307 (32 bytes, 512:1). The original untrained decoder had MSE 0.160285 at five recursion levels; the optimal constant predictor had MSE 0.0653658. GCC and Clang results agreed closely. These figures describe a fixed synthetic test and include a 24-byte archive header; they do not establish general real-data performance, ANN training, production-scale throughput, or a 1000× speedup.
