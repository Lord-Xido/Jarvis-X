# 3D-DSMPE-Ω Reference Engine

The 3D Dr. Moagi Sparse Manifold Projection Engine is a bounded reference implementation and interactive browser visualization for adaptive 3D field encoding.

## Canonical measured objective

For a target field `Ψ`, an octree latent representation `𝒯_d` at depth `d`, and its piecewise-constant decoder `𝒟`, the implementation evaluates

```text
M_Ω(Ψ, 𝒯_d) = RMSE[Ψ, D(𝒯_d)] + γ · |𝒯_d| / |𝒯_d^full|
```

and selects

```text
d* = arg min_{1 <= d <= D} M_Ω(Ψ, 𝒯_d)
```

The first term is measured over a deterministic evaluation lattice. The second is the materialized-node ratio relative to a complete eight-way tree. The denominator is therefore

```text
|𝒯_d^full| = (8^(d+1) - 1) / 7
```

rather than `8^d`, which counts only terminal leaves.

## Reference implementation

`src/jarvisx/dsmpe_reference.py` provides:

- a deterministic displaced-torus signed-distance field;
- conservative octree collapse using a declared Lipschitz bound;
- piecewise-constant decoding through the containing leaf;
- reconstruction RMSE, materialization ratio, compression ratio and depth entropy;
- exact leaf-volume partition checking;
- deterministic SHA-256 evidence digests;
- depth search over the observed regularized objective.

Run the focused tests with:

```bash
pytest tests/test_dsmpe_reference.py
```

The tests require deterministic repeated digests, exact complete-tree counts, bounded compression, volume preservation, refinement-improved reconstruction error, minimum-loss depth selection and input validation.

## Browser visualization

Open `docs/demos/3d-dsmpe-omega.html` directly in a modern browser. It has no external runtime dependencies.

The three visualization phases are intentionally distinct:

- **Original Ψ:** sampled points near the zero-isosurface of the continuous field.
- **Latent 𝒯:** wireframe cubes for boundary leaves in the selected octree.
- **Decoded 𝒟:** decoded surface points projected from latent leaf centers using a finite-difference field gradient.

The browser computes the same objective class as the Python reference, displays the observed candidate minimum, validates tree cardinality and partition invariants, emits a deterministic model digest, and can export the current evidence record as JSON.

## Capability boundary

This implementation establishes bounded software behavior for a deterministic adaptive octree codec. It does not establish a trained neural autoencoder, semantic intelligence, physical hardware performance, lossless reconstruction, production security or superiority over neural implicit representations, sparse voxel DAGs, octrees or other geometry codecs.
