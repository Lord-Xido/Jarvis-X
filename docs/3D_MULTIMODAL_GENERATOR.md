# Dr. Moagi 3D Multimodal Generator

## Scope

`DrMoagi-Multimodal-3D` evolves the bounded Jarvis-X 3D autoencoder into a dependency-free reference composition layer for text, image, audio, video, volumetric and generic numeric media. It reuses the existing `Autoencoder3D` numerical kernel rather than replacing it.

The runtime is intentionally explicit about its capability boundary: it is a deterministic research generator and cross-modal latent transport engine. The default weights are not a pretrained foundation model, so prompt-conditioned outputs are procedural until modality models are trained on aligned data. It does not claim semantic text-to-image/video quality without such training.

## Architecture

Every supported modality is mapped into a bounded scalar 3D field

```text
X_m in [-1,1]^(1 x N x N x N)
```

and encoded by a modality-specific 3D autoencoder

```text
Z_m = E_m(X_m)
```

with common latent geometry

```text
Z_m in [-1,1]^(C x N/2 x N/2 x N/2).
```

Because every model shares the same latent tensor shape, a source latent can be decoded through another modality head:

```text
Y_n = D_n(Z_m).
```

This is the engine's cross-modal transport primitive. It is shape-compatible by construction; semantic alignment must be learned from paired or otherwise aligned training data.

## Prompt-conditioned generation

A deterministic prompt field `P(prompt, seed)` is generated in the same latent geometry. Conditional translation mixes encoded source state and prompt state:

```text
Z* = alpha E_m(X_m) + (1-alpha) P(prompt, seed)
Y_n = D_n(Z*)
```

where `alpha` is `--mix` in `[0,1]`.

Unconditional generation bypasses source encoding:

```text
Y_n = D_n(P(prompt, seed)).
```

Prompt seeding uses a stable FNV-style text hash, xorshift state evolution and bounded sinusoidal phase injection. This gives deterministic replay for the same configuration, seed, prompt and platform.

## Modality adapters

- **text**: UTF-8 bytes are folded through the 3D lattice with a depth positional phase; generated values map to printable ASCII in this reference backend.
- **image**: scalar/RGB-like sample packets are channel-averaged, nearest-resampled in `x/y`, and replicated through the input volume; output is a grayscale PGM slice.
- **audio**: waveform samples are resampled over the 3D lattice; output is mono PCM16 WAV.
- **video**: frame index maps to the `z` axis and spatial coordinates map to `x/y`; output is a deterministic PGM frame sequence plus manifest.
- **volume3d**: 3D samples are nearest-resampled directly into the model field; positive generated voxels can be exported as an OBJ point cloud.
- **generic**: arbitrary normalized samples are folded through the lattice and emitted as raw `float32`.

The current CLI treats non-text input files as raw bytes normalized to `[-1,1]`. PNG/JPEG/MP4/MP3 container decoding is deliberately outside the dependency-free core; production deployments should attach codec adapters before constructing `MediaPacket` values.

## Build

```bash
cmake -S cpp_runtime -B build/cpp-runtime -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp-runtime --parallel
ctest --test-dir build/cpp-runtime --output-on-failure
```

The new executable is emitted as:

```text
DrMoagi-Multimodal-3D
```

## Examples

Text to image-volume projection:

```bash
./build/cpp-runtime/DrMoagi-Multimodal-3D \
  --source text \
  --target image \
  --text "recursive geometric intelligence" \
  --prompt "high spatial coherence" \
  --mix 0.8 \
  --edge 8 \
  --channels 4 \
  --output-dir .jarvisx-mm3d
```

Text to waveform:

```bash
./build/cpp-runtime/DrMoagi-Multimodal-3D \
  --source text \
  --target audio \
  --text "Jarvis X" \
  --prompt "smooth harmonic pulse" \
  --output-dir .jarvisx-audio
```

Prompt-only volumetric generation:

```bash
./build/cpp-runtime/DrMoagi-Multimodal-3D \
  --source text \
  --target volume3d \
  --prompt "symmetric shell" \
  --unconditional \
  --output-dir .jarvisx-volume
```

Source-adaptation steps remain available through the inherited autoencoder training path:

```bash
./build/cpp-runtime/DrMoagi-Multimodal-3D \
  --source audio \
  --target video \
  --input waveform.raw \
  --train-steps 16 \
  --prompt "temporal field" \
  --output-dir .jarvisx-video
```

## Output contract

Each run writes `generation.csv` plus one target-specific artifact:

- text: `generated.txt`
- image: `generated.pgm`
- audio: `generated.wav`
- video: `frames/frame-NNNN.pgm` and `generated-video.txt`
- volume3d: `generated.obj`
- generic: `generated.f32`

Telemetry records source/target modality, latent/output element counts, RMS energy, conditioning mix and final adaptation MSE.

## Identity and attractor semantics

The module preserves the architectural distinction established by the 3D locked formulation:

```text
representation: D_m(E_m(X)) approximates X after training
control/physical convergence: separate subsystem
```

The generator is a representation and synthesis layer. It does not infer that a reconstructed physical state is repaired, nor does it directly command safety-critical actuators.

## Evolution path

This reference layer is designed so production capability can be added without changing the core contract:

1. attach real codec adapters (PNG/JPEG, WAV/FLAC, MP4/WebM, tokenizers);
2. replace independent modality training with aligned contrastive/cycle objectives;
3. add temporal latent prediction for video/audio continuation;
4. add diffusion/flow or autoregressive latent priors behind `prompt_latent`;
5. add GPU tensor backends while retaining the bounded CPU reference implementation;
6. checkpoint modality models plus shared latent alignment state;
7. benchmark reconstruction, cross-modal retrieval/alignment and generation quality separately.

This keeps the execution order: working -> robust -> portable -> elegant -> advanced.
