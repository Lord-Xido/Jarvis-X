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
