# Dr Moagi Hyper-Matrix — bounded 3D reference engine

This independent research module is adapted from the reference demonstrator described in the associated mathematical breakdown. It is **not** wired into the authoritative VM, financial systems, or any production control path. Its input is deterministic; all outputs are numerical research results.

Install as an optional extra from the repository root:

```bash
python -m pip install -e ".[hypermatrix]"
jarvisx-dr-moagi-hypermatrix --self-test
jarvisx-dr-moagi-hypermatrix --side 16 --steps 30 --outdir results
# Optional visualization:
python -m pip install matplotlib
jarvisx-dr-moagi-hypermatrix --side 16 --steps 30 --outdir results --plot
```

## Output
- `results/verification.json`: configuration and numerical verification metrics.
- `results/spatial_results.npz`: surface, pulsed coordinates, fixed-point graph state, 3D vector field, original and reconstructed voxels, and **synthetic** spectral trace. Load with `np.load(path, allow_pickle=False)`.
- `results/octree_encoded.npz`: independent compressed-file octree container. Load with `load_octree` from the module.
- `results/hypermatrix_3d.png`: graphical representation of the generated surface and 3D volume.

## Algorithm flow
1. Evaluate PDF's parametric geometry over `nu × nv` samples.
2. Add a defined radial displacement driven by Gaussian-attenuated, seeded phase offsets.
3. Query the geometry using a `scipy.spatial.cKDTree` and build row-normalized sparse graph connections.
4. Contract a graph state with `s' = (1-eta)s + eta*tanh(a*s + b*P@s + f)` under `|a|+|b|<1`.
5. Independently instantiate a **true** 3D periodic voxel vector field `psi[side,side,side,3]`.
6. Finite difference the equation `rho*psi_tt + damping*psi_t - diffusion*laplacian(psi) + coupling*div(psi outer phi) + rotation*curl(g*curl(omega)) = 0`.
7. Integrate velocity first then spatial state; require all values finite.
8. Address integers by 3D Morton interleaving; encode integer cubic data with one octree anchor and 7 residual arrays per cell group.
9. Save octree to disk, reload independently and check exact voxel equality and SHA-256 hashes.
10. Write numerical metrics and visualization.

## Scope / caveats
- The PDF supplies geometry and a threshold formula but **does not** define a complete deterministic computational engine, well-typed PDE, initial/boundary conditions, trained autoencoder or measured hardware speedup.
- Tensor rank correction, radial pulse direction, nondimensional Gaussian in pulse, finite-difference evolution, graph contraction, octree, and fixed parameter values are extensions clearly identified in source-code docstrings.
- The output is a reproducible numerical **reference prototype**, not a self-training ANN, general-purpose virtual machine, cloud service, or physical quantum processor.
- `8**L` is the total number of addressable cells at depth L, **not** a performance multiplier.
- No physical calibration or full nonlinear PDE stability proof is claimed. Use bounded grids and measure errors at each resolution.
- The spectral trace is illustrative, not the computed spectrum of an operator.
- The `discrete_force_residual_max` value is an algebraic assembly check; it does not establish convergence to an analytic PDE solution. Verification of PDE truncation/discretization errors requires separate refinement studies.