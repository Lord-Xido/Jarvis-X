# Dr Moagi Mandelbulb 3D ANN Quine — verified C++20 reference

This is a **bounded executable reference**, not a production codec, generative foundation model, or autonomous self-improving machine.

## Source of truth

- [Standalone C++20 multimodal engine](../cpp_runtime/mandelquine3d/dm_mandelquine.cpp)
- [Independent inward mathematical verifier](../cpp_runtime/mandelquine3d/dm_inward_verifier.cpp)
- [Reproduction and CLI](../cpp_runtime/mandelquine3d/README.md)
- [Mathematics and measured limits](../cpp_runtime/mandelquine3d/VERIFICATION_REPORT.md)
- [Offline 3D latent-field viewer](../cpp_runtime/mandelquine3d/3D_inward_verification_interactive.html)

This is additive to the existing [cloud fractal runtime specification](DR_MOAGI_CLOUD_FRACTAL_RUNTIME.md) and the [3D recursive autocoder equation](DR_MOAGI_3D_RECURSIVE_AUTOCODEC_MASTER_EQUATION.md); it does not replace their broader designs.

## Governing equations

Canonical target: Ψ* = F(Ψ*) = (R ∘ D ∘ M ∘ E)(Ψ*).

The implemented, **frozen-base** 3D latent update at voxel i is:

```text
T(Z)[i] = 0.68*Z[i] + 0.32*tanh(Z0[i] + 0.18*mean_six_neighbors(Z,i) + 0.08*phi[i])
```

The Mandelbulb 8th-power escape-time field phi is bounded, not a signed distance field. With fixed geometry phi and frozen initial base Z0, the max-norm Lipschitz bound is 0.68 + 0.32*0.18 = **0.7376**, establishing contraction of that subsystem only.

**Exact archive decoding** depends on residual side-information: `prediction XOR residual = original`. This validates byte preservation, **not compression**. The tested archives expand about 22–34 times.

## Reproduction

```sh
cd cpp_runtime/mandelquine3d
bash run_all.sh
```

GitHub Actions runs the same script with both `g++` and `clang++`, recording audit trace artifacts. The standalone verifier checks numerical gradients, geometry, recurrence, fixed-point residual, Quine reproduction, text/PPM/PCM16 WAV roundtrips, and tampering behavior.

## Known security limitation

The archive does not authenticate its residual. Deliberate bit flips can alter a decoded file without detection. Do not use this container for trusted storage, encrypted ROM, or security-critical media until authenticated integrity (for example, an AEAD container) and adversarial validation are implemented.

The ANN comprises 269 trainable parameters, is not trained end-to-end through the recurrent spatial field, and does not establish semantic understanding or neural lossless compression.
