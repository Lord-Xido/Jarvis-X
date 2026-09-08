# Dr Moagi / Jarvis-X 10^18-Cell 3D AE/AD Swarm

This app is a bounded browser reference for a virtual `1,000,000 x 1,000,000 x 1,000,000` auto-encoding/decoding swarm.

## Boundary

`10^18` is the **logical lattice cardinality**. The browser does not allocate or execute `10^18` JavaScript agents. It materializes a deterministic sparse sample and preserves the full coordinate space through exact `BigInt` addressing:

```text
n = x + 10^6 (y + 10^6 z)
```

with `0 <= x,y,z < 10^6`.

## End-to-end loop

```text
BYTES / IR
  -> virtual 3D source
  -> rigid spatial transform
  -> global-memory coupling Omega
  -> inward radial fold
  -> finite-bit latent quantization
  -> outward inverse fold
  -> inverse memory coupling
  -> inverse spatial transform
  -> reconstruction error
  -> 5 x 5 x 5 self-search
  -> selected runtime state
  -> repeat
```

For a rotated source point `p` and rotated swarm centroid `c`, the coupled state is

```text
q = (1 - omega) p + omega c
```

The inward encoder is

```text
z = q / (1 + alpha ||q||)
```

and, before quantization error, the radial inverse is

```text
r = r_folded / (1 - alpha r_folded)
```

followed by

```text
p' = (q' - omega c) / (1 - omega)
```

and the inverse rigid transform.

## Quantization

With `b` latent bits per axis, each sampled 3D agent uses `3b` latent bits. At the default `b = 8`, this is 24 bits per 3D agent versus 96 bits for three float32 coordinates, a nominal representation ratio of `4:1`. This ratio refers only to the toy coordinate payload and is not a claim about arbitrary multimedia compression.

## Self-optimization

The runtime control state is

```text
Theta = (alpha, b, omega)
```

and each optimization generation evaluates a bounded `5^3 = 125` local candidate cube. Each candidate is scored with the simulator objective

```text
J = MSE
  + 0.00012 * (3b / 96)
  + 0.002 * alpha^2
  + 0.0015 * omega^2
```

and the lowest-score candidate becomes the next center:

```text
Theta_(t+1) = argmin_{Theta in N_5^3(Theta_t)} J(Theta)
```

The browser caps candidate evaluation to a sparse subset, so optimization remains interactive.

## What the visualization means

The moving points are a deterministic sampled projection of the virtual state space. Camera rotation is presentation-only. It is distinct from the inward latent motion and from the optimizer's movement through configuration space.

## Run

Open `index.html` in a modern browser. No server-side runtime or external JavaScript dependency is required.

## Relationship to the repository

This app is the browser/visual counterpart of `src/jarvisx/inward_document_swarm.py` and the existing bounded inward self-optimization work. It extends the same architectural rule: very large spaces are represented symbolically/sparsely, while measured computation remains explicit and bounded.
