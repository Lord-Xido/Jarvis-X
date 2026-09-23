# Trainable MM3D multimodal engine

`jarvisx.mm3d_engine` is the optional PyTorch backend for the Jarvis-X 3D
multimodal architecture. It complements the deterministic
`dr_moagi_multimodal_loop` runtime; it does not replace that bounded reference
implementation.

## Operational state

For modality `m`, the encoder maps an input into a common cubic field:

```text
V_m = E_m(X_m),    V_m in R^(B x C x G x G x G)
```

Text, source code, RGB images, mono audio, and RGB video therefore share one
latent geometry. Learned modality scores are normalized with softmax and fused:

```text
alpha_m = softmax(q_m)
Z_0 = Phi_fuse(sum_m alpha_m * (V_m + e_m))
```

The latent field is refined with a stable bounded six-neighbour diffusion term
and learned residual transform:

```text
Z'      = Z + alpha * Laplacian_6(Z)
Z_next  = Z' + sigmoid(g(Z')) * f_theta(Z')
```

Global voxel attention then operates on the `G^3` spatial tokens. Optional
Omega memory mixes the current state with the previous recurrent state. The
same final state is decoded through text, code, image, audio, and video heads.

The executable invariant is:

```text
Encode -> Fuse -> Diffuse -> Refine -> Attend -> Remember -> Decode -> Train
```

This is a trainable neural reference architecture. Random weights are not AGI,
and conceptual scale is not measured physical performance.

## Installation

Install Jarvis-X with the optional backend:

```bash
python -m pip install -e '.[mm3d]'
```

The extra installs PyTorch, NumPy, and Pillow without making those heavy
packages mandatory for the base deterministic VM.

## Run

Run one forward generation pass:

```bash
jarvisx-mm3d --out mm3d-output
```

Run a bounded training smoke step before generation:

```bash
jarvisx-mm3d --steps 1 --out mm3d-output
```

The command writes:

- `generated_image.png`
- `generated_audio.wav`
- `generated_video.gif`
- `generated_text.txt`
- `generated_code.txt`
- `telemetry.json`

Telemetry records the actual device, parameter count, latent shape, latent
statistics, and learned modality weights for that run.

## Default geometry

The default shared state is:

```text
B x 64 x 6 x 6 x 6
```

or 216 spatial latent sites with 64 channels per site. The default model has
roughly 4.5 million trainable parameters; the exact count is reported at
runtime because code or configuration changes can alter it.

## Verification

`tests/test_mm3d_engine.py` uses a reduced configuration to verify:

- a constant field has zero six-neighbour Laplacian;
- all five modalities reach the shared latent field;
- generated output tensors have the declared shapes;
- modality fusion weights sum to one;
- the aggregate loss is finite;
- gradients propagate through the engine.

The tests automatically skip when the optional MM3D dependencies are not
installed, preserving the base Jarvis-X installation boundary.
