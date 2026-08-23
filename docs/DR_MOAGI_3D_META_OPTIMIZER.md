# Dr Moagi Inward 3D Meta-Optimizer

## Purpose

The inward meta-optimizer turns the Dr Moagi runtime onto its own **bounded operational configuration**. It does not rewrite source code. It evaluates alternate runtime configurations in isolated replay kernels and promotes a configuration only when it improves a multi-metric objective without violating reconstruction or state-fidelity regression gates.

The meta-state is:

```text
C_n = (compression_n, adaptation_n, dynamics_n)
```

with a local 3D search coordinate:

```text
delta = (dx, dy, dz),  dx,dy,dz in {-1,0,+1}
```

The three axes are:

```text
X = compression geometry
    block_size, quantization, prune_epsilon

Y = adaptive dynamics
    DM-DD learning_rate, rho, omega_gain, latent budget, pass depth

Z = spatial/fixed-point dynamics
    contraction, attenuation, fixed-point pass depth
```

## Inward optimization law

For current runtime configuration `C_n`, deterministic replay suite `B_n`, objective `J`, and bounded neighbourhood `N_3(C_n)`:

```text
C* = argmin_{C in N_3(C_n)} J(C; B_n)
```

Promotion is transactional:

```text
C_(n+1) = C*    if Pi_meta(C*, C_n) = ACCEPT
C_(n+1) = C_n   otherwise
```

The system-wide invariant remains:

```text
PROVISIONAL != AUTHORITATIVE
```

A candidate runtime is always executed in a separate `DrMoagiOSKernel`; it cannot mutate the production state during evaluation.

## Multi-metric objective

The current reference objective combines:

- AutoExec reconstruction MSE;
- DM-DD residual RMS;
- fixed-point residual;
- anchor drift from the replay input;
- exact DMOS2 transport bytes per source cell;
- active/latent compute proxy;
- phase-velocity activity;
- a large rejection penalty.

No single sparsity or reconstruction number is treated as sufficient evidence of improvement.

## Robustness replay

Each candidate is evaluated on three deterministic workloads:

1. the bounded authoritative sparse state sample;
2. a deterministic ±3% amplitude perturbation;
3. a one-voxel spatial translation.

The final score combines mean and worst-case replay performance:

```text
J_robust = mean(J_i) + 0.25 * max(J_i)
```

This prevents the search from overfitting only the exact current state.

## Successive halving

The search does not fully evaluate all 26 neighbours at maximum depth. It performs:

```text
26-neighbour lattice
 -> bounded candidate subset
 -> short probe replay
 -> rank
 -> survivor set
 -> confirmation replay
 -> meta gate
```

This keeps inward self-optimization bounded and reduces search cost relative to exhaustive full-depth evaluation.

## Meta promotion gate

A candidate can be promoted only if:

```text
candidate rejected == false
relative score improvement >= configured threshold
candidate reconstruction MSE <= baseline MSE * allowed_regression
candidate anchor drift <= baseline drift * allowed_regression
```

On promotion, `SelfOptimizing3DSystem` constructs a fresh kernel with the promoted configuration, restores the authoritative sparse state, carries forward DM-DD `Omega`, `Theta`, and iteration, preserves the OS cycle, verifies exact transport, persists a checkpoint, and atomically swaps the wrapper's active kernel reference.

The old kernel remains untouched during search.

## Self-reference loop

The higher-order runtime is therefore:

```text
X_t
 -> DrMoagiOSKernel(C_n)
 -> operational reports
 -> inward meta-observation
 -> 3D candidate lattice
 -> isolated replay kernels
 -> robust ranking
 -> Pi_meta
 -> C_(n+1)
 -> next OS cycle
```

or:

```text
M_(n+1) = Pi_meta[ MetaSearch(M_n, Replay(M_n)) ]
```

where `M_n` contains both the ordinary Dr Moagi state and its operational configuration.

## CLI

Run a bounded demo search:

```bash
python -m jarvisx.dr_moagi_meta_cli demo --side 16 --max-candidates 13 --pretty
```

Or optimize against a sparse JSON field:

```bash
python -m jarvisx.dr_moagi_meta_cli file field.json --side 64 --pretty
```

## SOTA boundary

The optimizer is **designed to search for configurations that outperform its incumbent runtime on matched replay workloads**. This is not the same as proving state-of-the-art performance.

The report deliberately emits:

```text
claim_status = unverified_against_external_sota
external_sota_verified = false
```

until a matched external baseline, benchmark protocol, hardware environment and reproducible results are supplied.

That boundary is intentional: self-improvement is an empirical property, and "beyond SOTA" requires matched external evidence rather than internal self-comparison alone.
