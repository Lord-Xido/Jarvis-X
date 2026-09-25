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
Encode -> Fuse -> Diffuse -> Refine -> Attend -> Remember
       -> Echo -> Decide -> Render3D -> Decode -> Train
```

## Echo-through resolvent

After recurrent memory, the runtime now permeates the latent field through a
bounded echo resolver. The conceptual full-system operator is

```text
M = Pi o F o D o A_theta o E
E_omega = sum_(n=0)^infinity omega^n M^n = (I - omega M)^(-1)
```

when the weighted operator is contractive. The executable MM3D backend does
not claim to evaluate that infinite nonlinear whole-engine composition.
Instead, it uses a finite, auditable latent surrogate:

```text
M_echo(Z) = (1 - beta) Z + beta Avg3D(Z)
Z_echo    = sum_(n=0)^K omega^n M_echo^n(Z)
```

where replicated-boundary 3x3x3 averaging and a convex blend make
`M_echo` non-expansive in the sup norm. Therefore the corresponding infinite
weighted series is bounded for `0 < omega < 1`. Runtime execution truncates
at `K = echo_depth` and reports both

```text
echo_weight_sum = sum_(n=0)^K omega^n
echo_tail_factor = omega^(K+1) / (1 - omega)
```

so the unresolved geometric-series tail is explicit. The echoed field becomes
the state used by the decision, renderer, and modality decoders, and when
stateful execution is enabled Omega memory retains that echoed state for the
next cycle.

This is the operational meaning of "echo through": previous/refined spatial
structure is not merely stored alongside the current field; it is propagated
back through the shared latent geometry with diminishing weights.

## Permeated 3D output state

The explicit end-to-end output equation is implemented as

```text
Psi_t = R_3D(D_phi(Z_t), softmax(W_o pool(Z_t)))

Z_t = T_theta(X_in,t (+) E_env,t (+) Omega_t)
```

where `(+)` denotes learned multimodal fusion rather than arithmetic addition.
The optional environment input is a tensor with shape

```text
[B, environment_channels, D, H, W]
```

and is encoded into the same `B x C x G x G x G` latent lattice as the
text, code, image, audio, and video modalities.

The decision branch computes

```text
a_t = W_o pool(Z_t)
p_t = softmax(a_t)
```

and reports both `action_logits` and `action_probs`. The 3D render branch
projects `p_t` back into latent-channel space, conditions the volumetric field,
predicts RGB plus density, and performs front-to-back alpha compositing along
the depth axis. The resulting differentiable projection is returned as
`rendered_3d`.

This keeps softmax in the action/probability branch instead of forcing the
entire geometric latent state through a probability simplex. Geometry therefore
remains available to the decoders and renderer while the decision head receives
a normalized probability distribution.

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
- `rendered_3d.png`
- `telemetry.json`

Telemetry records the actual device, parameter count, latent shape, latent
statistics, echo-series weight sum and tail factor, and learned modality
weights for that run.

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
- all six modalities can reach the shared latent field;
- the finite echo resolver matches the geometric series on a constant field;
- generated output tensors have the declared shapes;
- echo diagnostics are finite and positive under the default bounded config;
- modality fusion weights sum to one;
- the aggregate loss is finite;
- gradients propagate through the engine.

The tests automatically skip when the optional MM3D dependencies are not
installed, preserving the base Jarvis-X installation boundary.
