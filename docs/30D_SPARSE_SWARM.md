# Jarvis-X 30-Dimensional Sparse 3D Swarm

This subsystem implements a virtual `1000 x 1000 x 1000` lattice without allocating one billion voxels. Only active coordinates and their immediate diffusion frontier are materialised.

## State

Each active coordinate stores three vectors of configurable dimension (30 by default):

- `theta`: reaction-diffusion parameters
- `state`: local activity / hidden state
- `memory`: optional local memory weights

The global latent is the mean of active `theta` vectors. A fixed linear predictor maps `tanh(latent)` to a broadcast parameter delta.

## Dynamics

For coordinate `r`, the runtime applies:

```text
laplacian(r) = sum(theta(r + neighbour)) - 6 * theta(r)
reaction(r)  = tanh(theta(r)) * state(r)
actual(r)    = theta(r) + D * laplacian(r) + eta * reaction(r)
predicted(r) = theta(r) + W * tanh(Z) + b
residual(r)  = actual(r) - predicted(r)
```

A local deterministic descent approximation then reduces residual energy plus L2 regularisation. The implementation avoids constructing a global Jacobian, which would defeat sparse execution.

## Sparse expansion and pruning

At every iteration, candidates are the active set plus valid six-neighbour coordinates. New voxels are allocated only when diffusion or activity exceeds `epsilon`; inactive voxels are dropped.

Per-step cost is `O(N_t * d)` with a constant six-neighbour stencil, where `N_t` is the number of active voxels and `d=30` by default.

## Example

```python
from jarvisx.swarm import SparseSwarm30D

swarm = SparseSwarm30D()
theta = [0.0] * 30
theta[0] = 1.0
swarm.set_voxel((500, 500, 500), theta)

for metrics in swarm.run(100, tolerance=1e-6):
    print(metrics)
```

## Operational boundary

The virtual address space is mathematically one billion voxels. Actual memory use scales with the active set and its frontier, not with the full volume. This is a sparse simulator/runtime, not a physical allocation of `10^9 x 30` floating-point values.
