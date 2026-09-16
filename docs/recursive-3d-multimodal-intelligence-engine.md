# Recursive 3D Multimodal Intelligence Engine

## Locked operational baseline

This layer composes the bounded multimodal autoencoder with the recursive inward-folded swarm.
The governing execution rule is:

```text
encode globally
-> abstract inward
-> allocate frequency to residual information
-> refine selectively
-> decode
-> verify
-> optimize
-> recur
```

The implementation is `src/jarvisx/dr_moagi_recursive_intelligence.py`.

## Local 3D codec state

A logical codec voxel carries the abstract state

```text
S_i = [x_i, z_i, p_i, Omega_i, Theta_i, r_i, f_i, A_i]
```

where `x_i` is local multimodal input, `z_i` latent state, `p_i` geometry, `Omega_i` temporal
memory, `Theta_i` control/attention state, `r_i` reconstruction or fixed-point residual, `f_i`
allocated update frequency, and `A_i` abstraction leverage.

The local codec cycle is

```text
x_i -> E_i -> z_i -> R_i -> D_i -> x_hat_i -> r_i -> recur
```

The existing `DrMoagiMultimodal3DLoop` remains responsible for bounded multimodal encoding,
latent evolution, decoding, reconstruction/cycle metrics, and fixed-point convergence checks.

## Inward recursive abstraction

The scheduler uses the existing recursive fold:

```text
C_l : Z_l -> Z_(l+1)
R_l = Z_l - U_l(C_l(Z_l))
H_i = 1[|r_i| > tau]
Z' = U(C(Z)) + H * F(R)
```

Stable information remains represented at coarse scale. Fine computation is reactivated only where
residuals exceed the configured threshold.

For the default two-level `10 x 10 x 10` fold:

```text
per-level volume = 10^3
recursive fold volume = (10^3)^2 = 10^6
```

This is an analytical sparse-work ceiling, not measured acceleration.

## Frequency modulation

Residual magnitude controls local execution frequency:

```text
f(r) = f_min + (f_max - f_min) * sigmoid((r - tau) / kappa)
```

Therefore larger residuals receive at least as much update frequency as smaller residuals while all
frequencies remain bounded by `[f_min, f_max]`.

At the fixed point, residual-driven refinement approaches its minimum cadence rather than consuming
maximum update frequency indefinitely.

## Information-throughput model

For event rate `F`, physical information mass `M` bytes/event, abstraction leverage `A`, and
execution-efficiency factor `eta`:

```text
Q_physical = F * M * eta
Q_represented = Q_physical * A
```

For a spatially distributed field the corresponding continuum abstraction is

```text
Phi_info(t) = integral_V rho_codec(x,t) M(x,t) f(x,t) A(x,t) eta(x,t) dV
```

This is the formal version of:

```text
space x frequency x abstraction -> effective represented information throughput
```

The runtime reports three distinct quantities so that they are not conflated:

- `modeled_physical_bytes_per_second`: information mass attached to scheduled modeled events;
- `modeled_represented_bytes_per_second`: the same scheduled work multiplied by abstraction gain;
- `modeled_reallocated_ceiling_bytes_per_second`: an analytical ceiling if saved sparse capacity can
  be perfectly reassigned to independent useful work.

None is a wall-clock benchmark.

## Verification invariants

A cycle is marked `operational_mechanics_verified` only when all inherited geometry/work checks and
all new frequency/throughput checks pass.

The new invariants are:

```text
f_min <= f(r) <= f_max
f(r_1) <= f(r_2) for r_1 <= r_2
Q_physical = F * M * eta
Q_represented = Q_physical * A
```

The inherited closure checks remain conceptually:

```text
Z* = R(Z*, P*, Omega*, Theta*)
X* = D(Z*)
G(X*) = X*
```

with actual convergence represented by bounded numerical residuals rather than banners.

## Run

From the repository root:

```bash
python -m jarvisx.dr_moagi_recursive_intelligence --cycles 8
```

For JSON metrics:

```bash
python -m jarvisx.dr_moagi_recursive_intelligence --cycles 8 --json
```

Example modeled capacity and abstraction configuration:

```bash
python -m jarvisx.dr_moagi_recursive_intelligence \
  --capacity-hz 1e12 \
  --max-frequency 144 \
  --information-mass-bytes 512 \
  --abstraction-gain 1000 \
  --fold-levels 2 \
  --json
```

## Performance boundary

A modeled event capacity such as `1e12/s`, an abstraction factor such as `1e3`, or an inward
work-reduction ceiling such as `1e6` is not evidence of equivalent physical hardware throughput.
Promoting any modeled quantity to an empirical performance claim requires same-hardware wall-clock
benchmarks at matched output quality, including memory bandwidth, interconnect, synchronization,
kernel time, power, and profiler evidence.
