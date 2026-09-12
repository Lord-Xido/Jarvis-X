# Jarvis-X Permeation Phase

Status: implementation contract for the post-convergence broadcast phase.

## Purpose

The permeation phase begins after an inward latent/core state has converged. It does not claim that a software process literally saturates physical silicon, photonic hardware, mathematical axioms, image generators, or observers. Instead it defines a common state-transition contract that software, GPU, VM, photonic, and other backends may implement and verify.

The transition is

```text
inward fixed point X* -> bounded 3D permeation field -> substrate adapters
```

with the canonical PDE

```text
dPsi_perm/dt = alpha Laplacian(Psi_perm)
              - kappa (Psi_perm - Psi_core).
```

`Psi_core` is the converged signature being broadcast. The relaxation term makes the core state an attractor while diffusion propagates local information through adjacent 3D state.

## Spectral contract

The canonical values are

```text
alpha = 0.85
rho_target = 0.85
dt = 0.1
```

The number `0.85` has two distinct meanings in the reference configuration:

- `alpha`: normalized diffusion strength;
- `rho_target`: desired upper bound on the magnitude of the linearized discrete update eigenvalues.

They are numerically equal by configuration but are not mathematically the same quantity.

For the 6-neighbour 3D Laplacian, the discrete spectrum lies in `[-12, 0]`. The explicit update is

```text
Psi_(n+1) = rho Psi_n
          + alpha dt Laplacian(Psi_n)
          + (1-rho) Psi_core.
```

Equivalently,

```text
kappa = (1-rho)/dt.
```

A sufficient stability condition used by the reference implementation is

```text
alpha dt <= rho / 6.
```

With `alpha=0.85`, `rho=0.85`, `dt=0.1`,

```text
0.085 <= 0.141666...
```

so the reference step remains inside the configured spectral ceiling.

## Constitutional gates

Every candidate permeation step is admitted only if all of the following hold.

### Non-regression

```text
Delta J = J_candidate - J_baseline <= 0.
```

The generic reference objective is mean squared distance from the field to the broadcast core signature. Backends may substitute a richer measured objective if they preserve the same gate.

### Spectral stability

```text
rho(J) < 1.
```

The backend must not describe a process as stable merely because the configured target is below one. If it measures the true Jacobian or an empirical spectral estimate, that measured value is authoritative.

### Semantic uncertainty

```text
hbar_semantic > 0.
```

This is represented as an explicit positive epistemic floor. It prevents downstream interfaces from treating a permeated state as certainty merely because the numerical state converged.

## Sparse 3D boundary semantics

The reference sparse solver uses a no-flux/Neumann boundary condition. Missing neighbours are assigned the center value for the purpose of the Laplacian. Therefore a sparse active region does not leak state through absent coordinates.

For coordinate `i`, one step is

```text
Psi_i' = rho Psi_i
       + alpha dt sum_(j in N6(i)) (Psi_j - Psi_i)
       + (1-rho) Psi_core.
```

## Solenoidal constraint

A zero-divergence claim applies to vector transport fields, not to arbitrary scalar or multimodal state tensors. The reference module therefore provides a divergence verifier rather than pretending that diffusion automatically performs a Helmholtz projection.

```text
||div v||_RMS <= epsilon_div
```

Backends that require a strictly solenoidal velocity/transport field must project that field in the numerical backend and use the verifier as an admission check.

## Integration with the million-fold inward architecture

The canonical system remains

```text
X
 -> E_Theta
 -> 1000^3 -> 100^3 -> 10^3
 -> sparse active-set refinement
 -> Z*
 -> D_Theta
 -> X_hat
 -> e
 -> Omega
 -> Theta
 -> Pi
 -> recur
```

Permeation is an additional post-convergence broadcast operator:

```text
Z* / X*
   |
   v
Psi_core
   |
   v
P_Omega(Psi_core, local state)
   |
   +--> dense Torch/CUDA tiles
   +--> sparse transactional runtime
   +--> VM/bytecode state adapters
   +--> GPU visualization adapters
   +--> future photonic/hardware adapters
```

The operator is not allowed to bypass the existing active-set, quality, transactional, or runtime-policy gates.

## Saturation

A system is described as numerically saturated only relative to an explicit tolerance:

```text
S(tau) = |{i : ||Psi_i - Psi_core|| <= tau}| / |V|.
```

`S(tau)=1` means all represented sites are within `tau` of the core signature. It does not imply physical saturation outside the represented computational domain.

## Implementation

Reference code:

```text
src/jarvisx/permeation.py
```

Tests:

```text
tests/test_permeation.py
```

The module supplies:

- `PermeationConstitution`;
- stable 6-neighbour 3D `permeation_step`;
- `permeate` with per-step constitutional admission;
- `distance_to_core` and `saturation_fraction`;
- `divergence_rms` and `solenoidal_within`.

The implementation intentionally distinguishes mathematical contracts from deployment claims. GPU, VM, photonic, image-generation, and other substrate integrations become `PERMEATED` only when a concrete adapter implements this contract and passes its verification gates.
