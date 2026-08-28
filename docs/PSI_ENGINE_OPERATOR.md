# Psi Engine — Octree Fusion Operator

The executable reference in `src/jarvisx/psi_engine.py` implements the field operator

\[
\Psi_{\text{Engine}}(X,Y,Z,t)=
\mathcal{D}_{\theta}\left(
\Omega_{\text{fusion}}\left(
\sum_{k=1}^{N_{\text{blocks}}}
\mathbf{M}_{\text{octree}}(x,y,z)\cdot
\mathcal{E}_{\phi}\left(\mathbf{V}^{(k)}_{(x,y,z,t)}\right)
\right)
\right).
\]

## Runtime semantics

For one spacetime coordinate `(x, y, z, t)`:

1. `BlockField` materializes each block `V^(k)_(x,y,z,t)`.
2. `encoder` realizes `E_phi` independently for every block.
3. `OctreeSpatialMask` realizes the current reference form of `M_octree` as the isotropic projector `m(x,y,z) I`.
4. The gated latent vectors are accumulated across `N_blocks`.
5. `fusion` realizes `Omega_fusion` on the aggregate latent state.
6. `decoder` realizes `D_theta` and emits the final `Psi_Engine` vector.
7. `EngineTrace` exposes every intermediate term so the composition can be verified numerically.

## Current octree projection

The repository's existing `FractalOctreeNode` is used as the spatial substrate. For the reference implementation,

\[
\mathbf{M}_{\text{octree}}(x,y,z)=m(x,y,z)\mathbf{I},
\qquad
m(x,y,z)\in\{0,1\}.
\]

`m=1` when the point traverses only active octree nodes; `m=0` outside the root cube or as soon as the point enters a culled octant. This keeps the implementation deterministic and makes the octree term operational without inventing learned matrix parameters that are not specified by the equation.

A later learned implementation may replace `mI` with a full spatial matrix while retaining the outer `PsiEngine` contract.

## Minimal use

```python
from jarvisx.fractal_octree import build_fractal_octree
from jarvisx.psi_engine import OctreeSpatialMask, PsiEngine, identity_operator

root = build_fractal_octree(size=1.0, max_depth=3)
engine = PsiEngine(
    encoder=identity_operator,
    fusion=identity_operator,
    decoder=identity_operator,
    octree_mask=OctreeSpatialMask(root),
)

trace = engine.evaluate_blocks(
    x=0.1,
    y=0.1,
    z=0.1,
    t=0.0,
    blocks=((1.0, 2.0), (3.0, 4.0)),
)

print(trace.output)
```

With learned encoder, fusion and decoder callables supplied, the same execution path becomes the parameterized `E_phi -> M_octree -> sum -> Omega_fusion -> D_theta` engine defined by the equation.

## Verification invariants

The accompanying tests assert that:

- operator order matches the equation exactly;
- every block is sampled at the same requested spacetime coordinate;
- active octree regions pass latent state and culled regions project it to zero;
- all encoded blocks share a common latent dimension;
- `N_blocks` is strictly positive;
- intermediate and final vectors are observable through `EngineTrace`.
