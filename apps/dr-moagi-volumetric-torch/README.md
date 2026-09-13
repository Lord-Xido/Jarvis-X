# Dr Moagi Volumetric Torch Backend

A differentiable dense 3D backend for the Dr Moagi architecture. It is designed to complement, not replace, the sparse logical multiparallel runtime.

## Implemented

- Persistent registered 3D shells (`outer`, `mid`, `inner`).
- Recursive inward encode / outward decode folding.
- Multimodal projection for images, video, audio vectors and generic feature embeddings.
- Per-sample dynamic 3D hypernetwork convolution conditioned on the global core `Omega*`.
- Differentiable trilinear topological warping driven by reconstruction stress.
- Warp reconstruction loss, so the deformation network is optimized toward useful geometry rather than zero displacement.
- Kinetic integration with optional `torchdiffeq`; deterministic Euler fallback when it is absent.
- Gradient clipping and persistent shell-state updates after optimization.
- Sparse-to-dense transactional tile bridge with convergence-controlled recursive execution.
- Backend-neutral million-fold inward hierarchy controls in `jarvisx.millionfold_inward`.

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

## Million-fold inward hierarchy

The canonical logical optimization target is now

```text
1000^3 -> 100^3 -> 10^3
```

which is a `10^6` reduction in **coarse spatial sites**:

```text
1000^3 / 10^3 = 1,000,000.
```

This is not asserted to be a literal one-million-times wall-clock speedup. The hierarchy preserves fine information as sparse residual work and refines only regions whose measured error exceeds a threshold.

```text
full logical field
      |
      v
    100^3 ---- residual R0
      |
      v
     10^3 ---- residual R1
      |
      v
 latent core Z*
      |
      v
   decode X_hat
      |
   e = X-X_hat
      |
 active-set selection
      |
 Omega -> Theta -> Pi runtime update -> recur
```

The backend split is deliberate:

1. the sparse logical runtime owns large-scale state and candidate admission;
2. `jarvisx.millionfold_inward` selects active refinement regions, evaluates convergence and gates runtime-policy changes;
3. this Torch backend executes bounded dense 3D tiles;
4. `sparse_bridge.py` projects accepted tile results back into sparse transactional state.

The complete mathematical contract is documented in `docs/architecture/millionfold-inward-3d.md`.

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
python -m unittest -v test_engine.py test_sparse_bridge.py

# repository-root hierarchy/runtime tests
python -m unittest -v tests/test_millionfold_inward.py
```

The focused tests cover zero-flow warp identity, multimodal projection/fusion, dynamic hypernetwork kernel application, an end-to-end recursive optimization step, sparse bridge semantics, the exact `1000^3 -> 100^3 -> 10^3` site ratios, active-set selection, per-region convergence, and measured runtime-policy admission.

## Relationship to the logical multiparallel runtime

This module is a dense research backend for active tiles or patches. It does **not** allocate the full canonical `1000 x 1000 x 1000` logical field densely. A scalable deployment maps sparse active regions into bounded dense Torch tiles, executes this backend on those tiles, and projects accepted results back into sparse transactional state.

The governing execution rule is therefore:

> coarse everywhere; fine only where measured error says it matters.
