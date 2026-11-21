# Dr Moagi — bounded 3D ANN swarm (research reference)

## Purpose

The user-supplied micro-autoencoder swarm was corrected and packaged as an optional research subsystem alongside `dr_moagi_hypermatrix_engine.py`.

Each particle has a trainable 7→hidden→latent→hidden→7 multilayer perceptron with manual gradient descent. A separate spatial update contracts the particle positions toward their current centroid. Meta-operations prune low-energy particles and gate weak latent channels.

## Mathematical contracts

- **Virtual 3D coordinates:** `(x,y,z) in {0,...,999999}^3`, representing `10^18` *possible* cells; only the bounded active particle list is materialized.
- **Embedding:** text chunks → SHA-256 → seven normalized finite floating values (position plus four auxiliary features). One-way **hashing**, not lossless text encoding or a semantic transformer.
- **MLP:** `z = tanh(tanh(xW1+b1)W2+b2) * gate`; reconstruction `y = tanh(zW3+b3)W4+b4`. Per-particle objective: `MSE(x,y)` with real SGD updates.
- **Position contraction:** `p_i' = c + (1-g)(p_i-c)`, `c=mean_i p_i`, `0<g<1`. For a fixed particle set and exact arithmetic, all pairwise separations contract by precisely `1-g`.
- **Meta-level pruning:** heuristic quantile threshold on latent-plus-residual energy. It is not a learned importance metric, and **does not guarantee increased information quality**.
- **Stopping:** position step smaller than `1e-5`, MSE smaller than `0.015`, loss change below `1e-6`, for three consecutive checks. The status is `numerically_stationary`, **not** a mathematical proof that ANN parameters globally converged.
- **No guarantees:** this is neither a trained language model nor a reversible text codec nor a demonstrably exponentially accelerated computer.

## CLI

```bash
python -m pip install -e ".[hypermatrix]"
jarvisx-dr-moagi-swarm3d --particles 96 --minimum 28 --iterations 200 \
  --text "def contractive_map(x): return x * 0.5" \
  --out swarm.npz
python -m pytest -q -o addopts='' tests/test_dr_moagi_3d_ann_swarm.py
```

The optional `hypermatrix` extra already supplies NumPy, so no package-wide dependency was added. The generated NPZ contains bounded active positions, 3D integer addresses, latents, and loss history. Do not open untrusted NPZ files with pickle enabled.

## Example deterministic verification

For `n_particles=96`, `min_particles=28`, `seed=42`, and `max_iterations=200`, one local reference run reported 28 survivors, 156 total iterations, and final seven-component MSE around `1.03e-5`. This is a tested example, not a throughput benchmark, generalization claim, or global optimum.

The code is intentionally not linked into the authoritative VM or any transactional, banking, vehicle, or deployment execution path.
