# Inward 3D Multimodal Media ANN

**Status:** research runtime  
**Branch:** `feature/inward-multimodal-swarm3d`

## Purpose

This layer turns the existing inward multimodal 3D runtime into a small trainable
multimedia ANN. It keeps the logical `8192^3` space virtual and sparse; it does
not allocate a dense `8192 x 8192 x 8192` tensor.

The operational chain is:

```text
multimodal bytes/tensors
    -> modality encoding
    -> shared state (3D coordination r + high-D feature h)
    -> trainable autoencoder
    -> graph/swarm coupling
    -> inward E(D(z)) correction
    -> memory coupling
    -> converged 3D consensus latent
    -> RGB frame / PCM audio surfaces
    -> external media container if desired
```

The 3D coordinate is a control/coordination chart. Rich semantic/media content
remains in the feature vector; three coordinates are not claimed to preserve
unrestricted semantics.

## Trainable autoencoder

For input feature matrix `X`, the reference NumPy MLP implements

```text
h1 = tanh(X W1 + b1)
z  = tanh(h1 W2 + b2)
h3 = tanh(z W3 + b3)
y  = tanh(h3 W4 + b4)
```

and trains with explicit backpropagation against mean squared reconstruction
error.

## Inward operator

For each modality particle `p=(r,h,m)`, the autoencoder generates a reconstructed
feature and latent coordinate. The local inward target is

```text
Phi_r(p) = q r + (1-q) r_AE
Phi_h(p) = normalize(q h + (1-q) h_AE)
```

with `0 < q < 1`.

The runtime measures

```text
R_phi = mean ||Phi(p_i) - p_i||
```

as an explicit self-consistency residual.

## 3D graph coordination

Cross-particle graph weights use shared-feature cosine similarity and chart
distance:

```text
A_ij = softmax_j(feature_gain * cos(h_i,h_j)
                 - geometry_gain * ||r_i-r_j||^2)
```

The recurrent state update blends:

- current state,
- task anchor,
- graph consensus,
- inward autoencoder target,
- bounded memory state.

The gain sum is constrained below one before the step is accepted.

## Multimedia surfaces

After inward refinement, all modalities share a consensus 3D latent

```text
r* = mean_i r_i
```

The research runtime exposes:

- `generate_rgb_frame(r*, phase, size)` -> `uint8[H,W,3]`
- `generate_pcm_audio(r*, duration, sample_rate)` -> `int16[N]`

Both are conditioned on the same converged 3D state. This establishes a concrete
cross-media coordination path without coupling the core library to a particular
video container or operating-system process launcher.

MP4 packaging is intentionally external. A caller may pass the returned RGB
frames and PCM samples to FFmpeg, PyAV, OpenCV, or another container backend.
The core remains deterministic and testable without those dependencies.

## Bounded self-optimization

`bounded_optimize()` mutates only declared numerical runtime parameters. Every
candidate is run in shadow evaluation. Promotion requires:

```text
candidate_score > incumbent_score * 1.0025
candidate_fixed_point_residual <= incumbent_fixed_point_residual
finite(candidate_score)
```

Source code, permissions, deployment configuration, and hardware control are not
mutated by this optimizer.

## Local prototype result

The standalone prototype used to design this module produced a real MP4 after
external FFmpeg packaging. In that run:

```text
logical cells              549,755,813,888
initial AE reconstruction  0.0436708
final AE reconstruction    0.00321634
initial fixed-point error   0.550212
final fixed-point error     0.117584
consensus virtual cell      (4555, 4905, 3929)
```

These are prototype measurements, not universal performance guarantees.
Repository CI remains authoritative for the committed implementation.

## Verification

Focused tests cover:

1. autoencoder training reduces reconstruction loss;
2. inward dynamics reduce fixed-point residual on the deterministic fixture;
3. RGB and PCM surfaces are generated from the same 3D consensus latent;
4. bounded optimization cannot promote a fixed-point regression.

## Boundary

This is a classical numerical ANN/runtime. The 3D manifold is virtual. Media
frames and audio are numerical arrays. No claim is made that the semantic state
is a literal electromagnetic or physical-spacetime field.
