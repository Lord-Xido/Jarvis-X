# Dr Moagi Permeated Unified-Field Engine

A dense differentiable **single-field equilibrium backend** for Jarvis-X. It complements the merged sparse `1000 x 1000` multiparallel runtime and the shell-based `dr-moagi-volumetric-torch` backend.

The central state is one registered tensor

\[
U \in \mathbb{R}^{B\times C\times D\times H\times W},
\qquad C=C_{outer}+C_{mid}+C_{inner},
\]

and the same permeation operator is applied repeatedly until the bounded fixed-point iteration reaches its iteration budget or convergence tolerance.

## Operational cycle

```text
image / video / audio / text-code-sensor embeddings
                         |
                         v
                 MultimodalProjector
                         |
                         v
                    unified field U0
                         |
             +-----------+-----------+
             |                       |
       local Conv3D             global FFT path
             |                       |
             +----> channel/core <---+
                         |
                         v
                 PermeationOperator T
                         |
                damped fixed point
            U <- U + lambda * clip(T(U)-U)
                         |
                         v
                        U*
              +----------+----------+
              |          |          |
           outer        mid       inner
              |
              +--> reconstruction stress dL/dU*
                         |
              +----------+----------+
              |                     |
          warp network           kinetic flow
              |                     |
         warped U*              evolved U*
              |                     |
              +----------+----------+
                         |
              reconstruction + equilibrium loss
                         |
                         v
                 bounded optimizer step
                         |
                         v
                transactional field commit
```

## What this revision fixes

The supplied prototype had several shape and state-contract problems that prevented end-to-end execution:

1. `torch.fft.rfftn` preserves `D,H` and compresses only the last real axis, so its frequency volume is `[D,H,W//2+1]`, not `[D//2+1,H//2+1,W//2+1]`. The implementation uses the correct geometry.
2. A `[B,16,D,H,W]` outer stress tensor cannot be expanded into `[B,112,D,H,W]` because 16 is not a singleton dimension. Stress is instead computed directly with respect to the unified field.
3. The original ODE state was the 16-channel outer prediction while its derivative was expanded to 112 channels. The kinetic state and force now both live in unified-field space.
4. The original training target was a 112-channel unified field while the reconstruction head emitted 16 channels. Outer and full-field objectives are now explicit and shape-checked.
5. The field is now an `nn.Module` with registered buffers; `.to(device)` and `state_dict()` semantics are correct.
6. Warping is non-mutating while autograd is active and contributes to reconstruction loss before a persistent state commit.
7. The fixed-point solver reports an explicit residual and supports a convergence tolerance.
8. The spectral operator is low-rank rather than allocating an impractical dense `C x C x D x H x Wf` kernel.

## Global mixing semantics

The FFT path provides a **global spatial receptive field within one numerical iteration**. “Instantaneous global mixing” refers to computational dependency across the bounded tensor, not instantaneous physical propagation.

## Multimodal projection

The projector supports:

- 2D images;
- 3D video tensors (`time` treated as input volumetric depth before resampling);
- audio feature vectors;
- generic vectors for text, code, sensors, or other embeddings.

Image lifting and vector projection are factorized to avoid the very large fully connected layer implied by directly mapping an audio vector to all `112 x 16^3` voxels.

## Verification

Focused CPU tests cover:

- zero-flow warp identity;
- image/video/audio/feature fusion into one 3D field;
- correct differentiable rFFT geometry;
- non-mutating transactional forward passes;
- end-to-end optimization with active warp, kinetic and equilibrium losses.

Run:

```bash
cd apps/dr-moagi-permeated-unified-field
python -m pip install torch
python -m unittest -v test_engine.py
python engine.py
```

`torchdiffeq` is optional; when installed it is used for kinetic integration, otherwise the backend uses a deterministic Euler step.

## Scale boundary

This is a **dense active-tile backend**, not a dense materialization of the full million-loop runtime. The scalable architecture remains:

```text
sparse 1000x1000 scheduler
        -> select active tile
        -> dense unified-field equilibrium engine
        -> project result back to sparse transactional state
```
