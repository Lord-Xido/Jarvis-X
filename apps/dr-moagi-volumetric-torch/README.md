# Dr Moagi Volumetric Torch Backend

A differentiable dense 3D backend for the Dr Moagi architecture. It is designed to complement, not replace, the sparse `1000 x 1000` logical multiparallel runtime.

## Implemented

- Persistent registered 3D shells (`outer`, `mid`, `inner`).
- Recursive inward encode / outward decode folding.
- Multimodal projection for images, video, audio vectors and generic feature embeddings.
- Per-sample dynamic 3D hypernetwork convolution conditioned on the global core `Omega*`.
- Differentiable trilinear topological warping driven by reconstruction stress.
- Warp reconstruction loss, so the deformation network is optimized toward useful geometry rather than zero displacement.
- Kinetic integration with optional `torchdiffeq`; deterministic Euler fallback when it is absent.
- Gradient clipping and persistent shell-state updates after optimization.

## Architecture

```text
image / video / audio / feature vectors
                 |
                 v
      Multimodal3DProjectionAdapter
                 |
                 v
          outer shell [16,Cube]
                 |
          encode inward
                 v
             mid shell
                 |
                 v
            inner shell
                 |
                 v
              Omega*
                 |
         recursive residual fold
                 |
                 v
       dynamic hyper-convolution
                 |
         decode outward
                 v
       reconstructed outer field
          |              |
          |              +--> stress --> warp net --> displacement
          |                                   |
          |                                   v
          |                              warped field
          |
          +--> kinetic force --> ODE / Euler update
                 |
                 v
     unified reconstruction objective
```

## Why this revision differs from the initial prototype

The initial supplied prototype generated hypernetwork kernels but did not apply them, and computed a warped shell that was not used by the loss. This revision closes both loops:

1. generated per-sample kernels are applied with grouped 3D convolution;
2. the warped reconstruction contributes directly to the training objective;
3. shell tensors are registered PyTorch buffers so `.to(device)` and `state_dict()` semantics are correct;
4. optimization avoids mutating persistent shell state while the active autograd graph is being built;
5. `torchdiffeq` is optional rather than a hard runtime requirement.

## Run

```bash
cd apps/dr-moagi-volumetric-torch
python -m pip install torch
# optional higher-level ODE package
python -m pip install torchdiffeq
python engine.py
```

## Verify

```bash
cd apps/dr-moagi-volumetric-torch
python -m unittest -v test_engine.py
```

The focused tests cover zero-flow warp identity, multimodal projection/fusion, dynamic hypernetwork kernel application, and an end-to-end recursive optimization step.

## Relationship to the 1000 x 1000 multiparallel runtime

This module is a dense research backend for active tiles or patches. It does **not** allocate a `1000 x 1000 x depth` dense tensor. A scalable deployment should map sparse active regions from `DrMoagiMultiparallel3D` into bounded dense Torch tiles, execute this backend on those tiles, and project results back into the sparse transactional state.
