# DM-vOmegaXi+ Contour Operator

## Status

Executable mathematical kernel for the archived operator

```text
Psi = integral_Theta [ (Phi tensor grad(Lambda)) / Omega^Xi+ ]
                     * exp(DM - v*Omega*Xi+) dv
```

This document makes the implicit types, numerical assumptions, and integration semantics explicit. It is designed to feed the existing bounded `Psi -> Phi -> Lambda^-1 -> Omega -> Theta` fixed-point runtime; it does not replace that runtime or relax its transactional invariants.

## Typed state

Let

```text
Phi(v)         in R^m
grad Lambda(v) in R^d
Omega(v)       > 0
Xi+(v)         > 0
DM(v)          in R
```

Then

```text
Phi tensor grad(Lambda) in R^(m x d)
Psi                      in R^(m x d)
```

Componentwise,

```text
Psi_ij = integral Phi_i(v) * d_j Lambda(v)
                  * Omega(v)^(-Xi+(v))
                  * exp(DM(v) - v*Omega(v)*Xi+(v)) dv
```

All quantities appearing inside exponentials are normalized dimensionless runtime values.

## Local recurrence

Define the scalar weight

```text
W(v) = Omega(v)^(-Xi+(v)) * exp(DM(v) - v*Omega(v)*Xi+(v))
```

and the local tensor coupling

```text
C(v) = Phi(v) tensor grad(Lambda(v)).
```

Then

```text
dPsi/dv = W(v) * C(v)
```

and the discrete runtime recurrence is

```text
Psi[k+1] = Psi[k] + dv * W(v_k) * C(v_k).
```

The implementation is `DMvOmegaXiContourOperator.step`.

## Closed form for constant fields

For constant `Phi`, `grad(Lambda)`, `Omega`, `Xi+`, and `DM` over `v in [0,L]`,

```text
Psi = (Phi tensor grad(Lambda))
      * exp(DM)
      * (1 - exp(-L*Omega*Xi+))
      / (Xi+ * Omega^(Xi+ + 1)).
```

This is implemented exactly by `constant_causal_closed_form` using `expm1` for improved small-argument precision.

For the concrete arithmetic example

```text
Phi         = [2, -1]
grad Lambda = [3, 4]
Omega       = 2
Xi+         = 1
DM          = 0.5
L           = 1
```

we obtain

```text
Phi tensor grad(Lambda)
= [[ 6,  8],
   [-3, -4]]
```

and

```text
Psi ~= [[ 2.1384,  2.8512],
        [-1.0692, -1.4256]].
```

The test suite locks this result.

## Sensitivity structure

For the causal gate,

```text
log W = DM - Xi+*log(Omega) - v*Omega*Xi+.
```

Therefore

```text
d(log W)/dDM       = 1
d(log W)/dv        = -Omega*Xi+
d(log W)/dOmega    = -Xi+/Omega - v*Xi+
d(log W)/dXi+      = -log(Omega) - v*Omega.
```

These derivatives are exposed by `causal_sensitivities` and are suitable for diagnostics, controller tuning, or gradient-based parameter updates.

## Causal versus genuinely closed contours

The archived formula combines a closed-contour symbol with the non-periodic factor

```text
exp(-v*Omega*Xi+).
```

If `v=0` and `v=L` denote the same point on a closed loop, this factor generally has different endpoint values. The implementation therefore makes the choice explicit.

### Causal mode

`gate_mode="causal"` preserves the archived attenuation law exactly:

```text
G(v) = exp(DM - v*Omega*Xi+).
```

Use this for a forward path, time-like coordinate, or dissipative runtime interval.

### Periodic mode

`gate_mode="periodic"` uses

```text
G(v) = exp(DM - Omega*Xi+*(1-cos(theta)))
theta = 2*pi*v/period.
```

Then

```text
G(v + period) = G(v),
```

so the weighting is consistent on a true closed contour.

## Numerical safeguards

The executable contract enforces:

1. `Omega > 0` to avoid the power-law singularity.
2. `Xi+ > 0` for the intended positive selectivity/attenuation regime.
3. finite vector and scalar inputs.
4. finite positive integration step `dv`.
5. bounded logarithmic exponent before `exp` to prevent floating-point overflow.
6. invariant tensor shape across a discrete integration pass.

No large dense 3D allocation is implied by this operator.

## Relationship to the fixed-point engine

The contour kernel computes a measurable state increment. A higher-level inward loop may consume that increment before passing the candidate state through the existing encoder/decoder, recurrent memory fold, and bounded `Theta` projection.

Schematically:

```text
(Phi_t, grad Lambda_t, Omega_t, Xi+_t, DM_t)
    -> contour operator
    -> Delta Psi_t
    -> Psi_candidate
    -> Phi encode
    -> Lambda^-1 latent projection
    -> decode
    -> Omega recurrent fold
    -> Theta bounded projection
    -> H_(t+1)
```

The higher-level fixed point remains

```text
H* = F_DM(H*)
```

with measured convergence and a strictly positive semantic uncertainty floor. The contour equation is therefore an executable inner kernel, not a substitute for validation, observation, or external-world evidence.

## Reference API

```python
from jarvisx.dm_vomegaxi_contour_operator import (
    ContourSample,
    DMvOmegaXiContourOperator,
)

op = DMvOmegaXiContourOperator()
psi = ()
psi = op.step(
    psi,
    ContourSample(
        v=0.0,
        phi=(2.0, -1.0),
        grad_lambda=(3.0, 4.0),
        omega=2.0,
        xi_plus=1.0,
        dm=0.5,
    ),
    dv=0.01,
)
```
