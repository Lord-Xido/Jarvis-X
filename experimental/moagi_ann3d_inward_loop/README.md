# Dr Moagi 3D ANN — inward-loop bounded C++ reference

**Provenance.** Matladi Maxwell Moagi's C++ `MATLADI MAXWELL MOAGI ANN SYSTEM` concept supplied 2026-11-16. This is a revised research implementation, not the exact original source code, and is not a production or intelligence claim. The original design specified a 32×32×32 binary lattice; dense 32,768→1,024→256→1,024→32,768 autoencoder; fifty feedback cycles; six-neighbor toroidal flips; and a *virtual* `1e24 flips/ns` target.

## What changed, and why

* **Complete gradients:** unlike the supplied sample, which only updates `dec2` (not `enc2` despite its comment), all four layers in this implementation receive actual backpropagated SGD gradients. The output uses stable sigmoid + binary cross entropy (BCE), with the mathematically correct preactivation delta `(prediction - target)/64`.
* **Shared local patches:** a 32³ lattice is partitioned into 512 nonoverlapping 4³ patches. One shared 64→32→8→32→64 MLP processes every patch. The original dense model would have ~67.6 million trainable weights, ~270 MB of float32 weight storage alone, and high CPU costs. The patch model has ~4,608 weights plus biases; the 8:1 ratio is dimensionality reduction of activations, **not** proven lossless compression or an on-disk byte ratio. Patches do **not** learn across-patch boundaries.
* **Explicit task semantics:** supervised **synthetic denoising pretraining** with a known clean target precedes a 50-step inward inference loop. That inference loop feeds decoded-and-thresholded results back as the next state and applies local XOR flips; it does not keep optimizing weights on unknown truth. Therefore the experiment does not establish autonomous self-improvement or learned physical flip dynamics.
* **Evidence:** reports BCE vs synthetic reference, input/reconstruction/next-state bit errors, latent energy, attempted toggles, FNV-1a state hashes and local wall time. `16` flip centers × `7` toggles = `112` attempted bit toggles per cycle; overlapping toggles can cancel. `1e24 flips/ns` is unmeasured, unimplemented and not a legitimate hardware performance figure.
* **No misleading randomness assumption:** the fixed reference field comprises four spatial motifs. A random iid 32³ input generally cannot be reconstructed without loss from an 8-coordinate local bottleneck.

## Build and test

```sh
cmake -S experimental/moagi_ann3d_inward_loop -B build/moagi-ann3d -DCMAKE_BUILD_TYPE=Release
cmake --build build/moagi-ann3d --parallel
ctest --test-dir build/moagi-ann3d --output-on-failure
./build/moagi-ann3d/moagi_ann3d
```

Standalone alternative (when in this directory):

```sh
c++ -std=c++17 -O2 -Wall -Wextra -Wpedantic moagi_ann3d.cpp -o moagi_ann3d
./moagi_ann3d --self-test
./moagi_ann3d
```

Self-test covers 3D toroidal addressing, the XOR flip involution, exact patch index mapping, and a four-layer learning reduction on a fixed example.

## Execution law

```text
reference 32³ → clean 4³ training patches
             → inject 8% synthetic independent corruption
             → Encoder 64→32→8 (train all layers)
             → Decoder 8→32→64 (BCE targets: clean patches)
             → repeat for 30 training epochs

initial disturbed 32³ state
    → per-patch shared Encoder → latent8 → Decoder → threshold
    → reconstruction 32³
    → 16 (center + 6 neighbor) periodic XOR perturbations
    → next 32³ input (closed-loop recurrence, 50 cycles)
    → record true reference errors + hashes
```

The fixed clean target serves only as supervised training data and an evaluation oracle. It is **not** used for online correction inside the 50-step inference loop.

### Mathematical distinctions

* Encoder–decoder: `z = E_theta(x)`, `p = sigmoid(D_theta(z))`.
* Supervised BCE training: `theta <- theta - eta grad_theta BCE(p, x_clean)` with `x = corrupt(x_clean)`.
* Inward closed loop: `x_(t+1) = F_7(threshold(D_theta(E_theta(x_t))))` where `F_7` consists of discrete neighbor XOR toggles.
* Any apparent improvement must be assessed against *measured* error and proper held-out tests. This demonstration uses training motifs also present in the evaluation field and is **not** generalization evidence.
* No CPU/GPU/quantum physical operation at `10^24` flips per nanosecond is demonstrated. There is no claim of 3D geometric manifolds beyond an explicitly indexed lattice and local patch encoding.

## Compatibility

New isolated educational/research module. Does not change Jarvis-X's authoritative VM, codecs, installed application, workflows, deployment or existing interfaces. Review before any integration/promotion.
