# Astra Swarm EM3D: bounded neural–geometric–electromagnetic simulation

## Status and scope

**Executable C++17 research prototype, deterministic synthetic inputs; not a trained LLM, real RF device, or physical implementation of `10^(6,000,000)` agents.** This is a companion to existing `src/jarvisx/em_sig_3d_runtime.py`; it neither replaces the existing EM signalling application nor claims that the two have the same physical model.

### Pipeline

1. Sample `N` noisy 12,288-dimensional synthetic vectors from an explicit low-rank torus-based generator.
2. Encode with an implicit orthonormal `12,288 × 128` basis `A`: `z = Aᵀx`, and Euclidean-project the first three latent coordinates onto the torus of radii `R=10`, `r=2`.
3. Interpret the toroidal positions as **neutral magnetization-current emitters**: deposit moment `M` using cloud-in-cell interpolation on a periodic cubic grid, then calculate `J = curl⁻ M`. Therefore `div⁻ J ≈ 0`. Charge `rho=0` is a declared assumption, not a dynamically tracked particle charge.
4. Integrate *normalized* Maxwell curl equations with staggered forward/backward finite differences and a leapfrog time update: `B(n+1/2) = B(n-1/2)-dt curl⁺ E(n)` and `E(n+1)=E(n)+dt(curl⁻ B(n+1/2)-J)`. This is a periodic, Yee-style mimetic stencil. The grid arrays store component indices; sampling uses a simplified common-index interpolation rather than the exact geometric offsets of full physical Yee interpolation.
5. Sample six field components at each torus position; map them into latent dimensions 3–8 through an explicit (untrained) field gain. Latent coordinates 0–2 remain on the torus.
6. Decode independently as `xhat=A z` and apply bounded feedback `x_next=x+alpha(xhat-x)`.
7. Verify max absolute divergences for `E`, `B`, and `J`, geometric torus defect, and finite values; report synthetic ground-truth MSE and electromagnetic field energy.

This is an **information/field coupling research model**, not a demonstration that numerical neural embeddings correspond to physical electrical charge. Source moments and coupling gains are numerical assumptions. The field may influence latent updates, but no gradient-descent training or autonomous source generation has been implemented.

## Build and test

```bash
g++ -O2 -std=c++17 -Wall -Wextra -Wpedantic -fopenmp \
  native/astra_em3d/astra_em3d.cpp -o astra_em3d
./astra_em3d --self-test
./astra_em3d --samples 4096 --grid 32 --steps 8 --alpha 0.7 \
  --dt 0.08 --source 8 --coupling 0.05 --threads 4 \
  --receipt artifacts/astra-em3d.json --trace artifacts/astra-em3d.csv
```

Without OpenMP: omit `-fopenmp`. `--help` lists options. The default sample is 256 so it is practical on standard machines. `N=4096` sampled 12,288-dimensional agents use approximately 384 MiB for noisy states plus basis and truth and latent buffers in this double-precision version (plus grid and runtime overhead). Source strength is rescaled by sample count. Runtime is linear in `N × 12,288` for codec updates and `grid³` for Maxwell steps. Memory/time measurements must be benchmarked on deployment hardware, not inferred from symbolic population count.

Open `native/astra_em3d/visualizer.html` directly in a browser, select the generated trace CSV to inspect toroidal positions and sampled field vectors. The built-in view before loading is a **procedural demo**, not a runtime trace.

## Mathematical invariants

- Torus surface: `(sqrt(z0²+z1²)-R)²+z2²=r²`. Small floating-point residuals are expected.
- Current: `div⁻(curl⁻ M)≈0` on periodic compatible difference grids.
- Gauss constraints in neutral medium: `div⁻ E≈0` and `div⁺ B≈0`, preserved by these matched discrete curl operators up to rounding.
- CFL: `dt <= (BOX/grid)/sqrt(3)` with normalized `c=1`, `BOX=28`. Other medium constants are not implemented.
- Energy: report `0.5 * ∑cell(|E|²+|B|²) * dx³`; with a driven current it need not decrease or remain constant. The source-free plane-wave self-test checks bounded numerical energy drift rather than claiming exact energy conservation.
- Synthetic truth MSE: `mean((x - A z_truth)²)`. The encoder/decoder never receive the truth; this benchmark nevertheless favors an exactly specified low-rank toroidal synthetic distribution and is **not** a generalization evaluation.

## Validation boundary

`--self-test` exercises vacuum stability, current divergence, source-generated fields, a propagating source-free transverse-wave case, decreasing synthetic reconstruction error, field-to-latent coupling, and rejection of a CFL-violating time step. CI also validates JSON/CSV receipts and no-source behavior.

A subsequent research step could add charge-conserving particle deposition (if modelling free electric charges), exact Yee-position interpolation, physically calibrated material properties/boundaries, learned field coupling, ablation comparisons, and training/held-out datasets. Those capabilities are not claimed here.