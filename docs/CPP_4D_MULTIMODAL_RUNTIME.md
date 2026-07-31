# Bounded 4D Multimodal C++ Runtime

## Classification

This subsystem is a deterministic C++17 research runtime that composes four bounded 3D autoencoders with discrete latent-history memory, transactional candidate updates and an optional OpenGL/GLUT visualizer.

The four current modalities are represented by deterministic synthetic fixtures:

- **visual** — a solid 3D sphere field;
- **audio** — a windowed three-axis sinusoidal field;
- **text** — a token-bit and positional field derived from a fixed phrase;
- **generic** — deterministic bounded noise.

These fixtures exercise modality-specific state and scheduling. They are not image, video, audio or language codecs and do not ingest arbitrary media files in the current reference implementation.

## 1. State geometry

For each modality `m`, the runtime maintains a single-channel cubic input

```text
X_m ∈ [-1,1]^(1×N×N×N)
```

and an independent trainable 3D autoencoder

```text
Z_m = E_(theta_m)(X_m)
X_hat_m = D_(phi_m)(Z_m).
```

The latent geometry is

```text
Z_m ∈ [-1,1]^(C×N/2×N/2×N/2).
```

The additional dimension is a bounded discrete temporal history of latent tensors:

```text
H_m(t) = [Z_m(t), Z_m(t-1), ..., Z_m(t-T+1)].
```

Therefore the term **4D** means a 3D latent field indexed through discrete runtime time. The implementation does not claim a continuous physical fourth dimension or a learned 4D convolutional kernel.

## 2. Temporal fusion

The decoder receives an exponentially weighted temporal latent:

```text
Z_bar_m(t)
  = sum_(tau=0)^(T-1) lambda^tau Z_m(t-tau)
    / sum_(tau=0)^(T-1) lambda^tau,
```

where

```text
0 <= lambda < 1.
```

`lambda = 0` uses only the newest latent tensor. Larger values retain more history. The history depth is bounded to `[1,64]`.

Temporal coherence is reported as

```text
coherence_m
  = 1 - clamp(RMS(Z_m(t) - Z_m(t-1)), 0, 1).
```

This is a diagnostic similarity score, not a probability or formal stability proof.

## 3. Transactional self-optimization

At each adaptive cycle, the scheduler selects one modality. Its priority is based on current reconstruction MSE plus a small deterministic rejection-pressure term. The purpose is to allocate updates to the modality currently reconstructing least accurately rather than train every model blindly in fixed order.

For the selected modality, the runtime creates an isolated copy of the model and executes a bounded number `K` of SGD steps:

```text
(theta', phi') = SGD_K(theta, phi; X_m).
```

The candidate is admitted only when

```text
L_m(theta', phi') <= L_m(theta, phi) + epsilon,
```

where `epsilon` is the configured acceptance tolerance.

The transition is therefore

```text
candidate -> evaluate -> commit or rollback.
```

A rejected candidate never replaces the authoritative model. Commit and rollback counts are exposed in telemetry and checkpoint state.

This is bounded parameter optimization. The runtime does not rewrite arbitrary C++ code, mutate native instructions, establish consciousness or perform unrestricted autonomous self-modification.

## 4. Quantized inference

Training uses continuous latent activations. Optional final inference maps each activation through the canonical signed three-bit domain

```text
Q3 = {-4,-3,-2,-1,0,1,2,3}
```

and dequantizes it back into `[-1,1]` before temporal fusion and decoding.

No straight-through estimator or false differentiability claim is used. Candidate training remains continuous.

## 5. Headless runtime

Build all C++ targets:

```bash
cmake -S cpp_runtime -B build/cpp-runtime -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp-runtime --parallel
ctest --test-dir build/cpp-runtime --output-on-failure
```

Run the adaptive multimodal engine:

```bash
./build/cpp-runtime/jarvisx-multimedia4d \
  --edge 8 \
  --channels 4 \
  --temporal-depth 8 \
  --proposal-steps 2 \
  --cycles 120 \
  --learning-rate 0.03 \
  --temporal-decay 0.72 \
  --quantized \
  --output-dir .jarvisx-multimedia4d
```

The output directory contains:

```text
multimedia4d.csv
visual-input.obj
visual-latent.obj
visual-reconstruction.obj
audio-input.obj
audio-latent.obj
audio-reconstruction.obj
text-input.obj
text-latent.obj
text-reconstruction.obj
generic-input.obj
generic-latent.obj
generic-reconstruction.obj
checkpoint/
```

The OBJ files are voxel point clouds, not watertight meshes.

## 6. Checkpoint contract

A checkpoint persists:

- one `JX3D` autoencoder model per modality;
- accepted and rejected transaction counts;
- temporal latent histories using round-trip-safe decimal float precision;
- temporal configuration, cycle count and Q3 inference state;
- the most recent gradient telemetry value.

Loading validates the checkpoint version, modality order, tensor dimensions and temporal configuration before replacing runtime state.

For the same compiler/platform, configuration and checkpoint, replay is deterministic. Cross-platform bit identity is not claimed because standard floating-point and transcendental implementations can differ.

## 7. OpenGL visualizer

When OpenGL and GLUT/freeglut are available, CMake also builds:

```bash
./build/cpp-runtime/jarvisx-multimedia4d-gl \
  --edge 16 \
  --channels 4 \
  --temporal-depth 8 \
  --proposal-steps 2 \
  --learning-rate 0.015
```

The rendered pipeline is directly backed by runtime tensors:

- left cube — selected modality input `X_m`;
- centre temporal stack — recent latent tensors `H_m(t)`;
- right cube — decoded temporal reconstruction `X_hat_m`;
- gold overlay — absolute residual `|X_m-X_hat_m|`;
- forward particles — encoder and decoder data flow;
- return particles — residual feedback.

Controls:

```text
1..4       visual, audio, text, generic
SPACE      pause/resume adaptive transactions
Q          toggle floating/Q3 latent inference
F          follow the scheduler-selected modality
R          toggle camera auto-rotation
S / L      save/load the visualizer checkpoint
+ / -      increase/decrease rendered voxel density
mouse drag rotate camera
ESC        exit
```

The frame-budget controller changes only visualization density, temporal layers displayed and training cadence. Wall-clock timing is not used to accept or reject model candidates, preserving deterministic numerical selection.

## 8. Validation

The regression suite checks:

- distinct deterministic modality fixtures;
- deterministic scheduler and transaction replay;
- bounded temporal-history depth;
- no admitted candidate exceeding the configured MSE tolerance;
- valid signed three-bit latent levels;
- checkpoint restoration of models, counters, histories and aggregate metrics.

## 9. Complexity boundary

Let input edge be `N`, latent channels `C`, kernel volume `K=27`, temporal depth `T`, modalities `M=4`, and proposal steps `P`.

Approximate per-cycle work for the selected modality is

```text
O(P C K N^3)
```

plus temporal fusion

```text
O(T C N^3 / 8).
```

All tensors and temporal histories are dense and deliberately bounded. This is a correctness and visualization reference, not a production multimedia model, GPU runtime or large-volume performance claim.
