# EEIITL Permeation Layer

**Status:** Experimental research architecture  
**System:** Ephemeral Electromagnetic Implicit Induced Telecommunications Logic (EEIITL)  
**Repository family:** Dr Moagi / Jarvis-X  
**Governing ADR:** `docs/adr/0023-eeiitl-volumetric-electromagnetic-permeation.md`  
**Attribution:** Inherits ADR-015 and `docs/attribution/PERMEATION_MANIFEST.md`

## 1. Purpose

The EEIITL Permeation Layer extends a discrete guided pulse/junction model into a volumetric electromagnetic state process.

The architectural transition is:

```text
discrete guided excitation
  -> volumetric propagation / coupling / scattering
  -> secondary induced sources
  -> field superposition
  -> thresholded spatial bit field
  -> sparse sensing
  -> edge/cloud state estimation
  -> bounded control update
  -> re-excitation
```

This layer is distinct from `docs/PERMEATION3D_NATIVE_BACKEND.md`, which describes software/backend execution permeation. Here, "permeation" means a physical electromagnetic field model over a three-dimensional medium.

## 2. Typed physical state

For position `r in V subset R^3` and time `t`, define

```text
S_EM(r,t) = [ E(r,t), H(r,t), J(r,t), B_logic(r,t), m(r,t) ]
m = { epsilon, mu, sigma, chi, ... }
```

The continuous field state obeys Maxwell's equations:

```text
curl E = - dB/dt
curl H = J_source + sigma E + dD/dt
D = epsilon E
B = mu H
```

The logical bit state is not identified with the raw field itself. It is produced by an explicit observation and threshold operator.

## 3. Permeation operator

Define the physical permeation operator

```text
P_m : J_source(r,t) -> Psi(r,t)
Psi = [E, H]^T
```

For a linear medium, the abstract Green-operator form is

```text
Psi(r,t)
  = integral_V integral_{-infinity}^t
      G_m(r,r',t-tau) J_source(r',tau)
    d tau d^3 r'
```

Thus a discrete coupling matrix is generalized to a continuous interaction kernel:

```text
M_ij -> G_m(r,r',t)
```

The implementation may use analytical Green functions, FDTD/FEM, measured impulse responses, reduced-order models, or another validated solver. The representation must state which approximation is authoritative.

## 4. Secondary induction

Local material response is represented by an induced-source operator

```text
J_ind(Psi)
  = sigma E
    + dP/dt
    + curl M
```

as applicable to the chosen constitutive model.

The self-consistent field is

```text
Psi = Psi_incident + G_m[J_ind(Psi)]
```

or, equivalently,

```text
Psi = P_m[J_source + J_ind(Psi)]
```

This is the physical basis for spatially distributed secondary carriers. It does not imply that uncontrolled electromagnetic leakage performs useful computation.

## 5. Volumetric bit field

Let `R` be a readout/projection operator and define

```text
s(r,t) = R[Psi(r,t)]
```

The threshold quantizer is

```text
Q_Theta(s) =
  +1,  s > theta_plus
   0, -theta_minus <= s <= theta_plus
  -1,  s < -theta_minus
```

so

```text
B_logic(r,t) = Q_Theta(R[Psi(r,t)])
```

The explicit zero/dead-band state prevents arbitrarily small field perturbations from being misrepresented as reliable bits.

Decision surfaces are threshold isosurfaces:

```text
Sigma_Theta(t) = { r in V : |R[Psi(r,t)]| = theta }
```

## 6. Spatial superposition

For `K` source events in a linear medium,

```text
J_source = sum_k J_k
Psi = sum_k P_m[J_k]
```

while the threshold layer provides the computational nonlinearity:

```text
Q_Theta(a+b) != Q_Theta(a) + Q_Theta(b)
```

Useful logic therefore requires jointly engineered source encoding, propagation geometry, material response, sensing geometry, and threshold/readout policy.

## 7. Penetration and attenuation

The common skin-depth expression

```text
delta = sqrt(2 / (omega mu sigma))
```

is retained only as the good-conductor approximation when `sigma >> omega epsilon`.

For a homogeneous lossy medium, use the attenuation constant

```text
alpha =
omega * sqrt(
  (mu epsilon / 2)
  * ( sqrt(1 + (sigma/(omega epsilon))^2) - 1 )
)
```

with effective penetration length approximately

```text
delta_eff ~= 1 / alpha
```

when that approximation is appropriate.

## 8. Sparse sensing and cloud loop

For probe `p`,

```text
y_p(t) = integral_V W_p(r) Psi(r,t) d^3r + nu_p(t)
```

Collect all observations as

```text
Y_n = H(Psi_n) + nu_n
Z_n = E_phi(Y_n)
Omega_(n+1) = M(Omega_n, Z_n)
U_(n+1) = pi_theta(Z_n, Omega_(n+1))
```

The actuator maps the bounded control state into source and/or material parameters:

```text
(J_(n+1), m_(n+1)) = A(U_(n+1))
```

and the next physical state is

```text
Psi_(n+1)
  = P_{m_(n+1)}[
      J_(n+1) + J_ind(Psi_(n+1))
    ]
```

## 9. EEIITL master operator

Define

```text
P_EEIITL
  = P
    o A
    o pi_theta
    o M
    o E_phi
    o H
    o Q_Theta
```

with recurrence

```text
S_(n+1) = P_EEIITL(S_n)
```

and operational invariant

```text
Excite
  -> Permeate
  -> Interfere
  -> Induce
  -> Threshold
  -> Sense
  -> Encode
  -> Reckon
  -> Reconfigure
  -> Re-excite
  -> recur
```

A fixed-point claim requires explicit sufficient conditions or measured convergence evidence:

```text
S* = P_EEIITL(S*)
```

Naming a recursive loop is not evidence that it is contractive or that a unique fixed point exists.

## 10. Verification boundary

A conforming implementation must distinguish:

1. **field solver state** — continuous/numerical E/H fields;
2. **logical state** — thresholded bit/ternary geometry;
3. **sensor state** — sampled measurements and uncertainty;
4. **estimated state** — edge/cloud reconstruction;
5. **control candidate** — proposed source/material update;
6. **authoritative actuation state** — only the verified, bounded control committed to hardware.

No candidate control becomes authoritative until it passes amplitude, frequency, timing, power, geometry, resource, and device-policy limits.

## 11. Engineering evidence requirements

Any implementation claim must identify:

- medium model and constitutive assumptions;
- frequency/bandwidth and pulse shape;
- source geometry and boundary conditions;
- field solver or measured transfer model;
- threshold/noise/hysteresis policy;
- sensor placement and sampling rate;
- reconstruction uncertainty;
- control-loop latency;
- power and thermal limits;
- measured or simulated bit error / classification error;
- validation environment and scale.

Chip, room, vehicle, water, soil, and outdoor regimes are not assumed equivalent. The operator abstraction may remain common while the physical realization changes.

## 12. Relationship to Jarvis-X

The physical EEIITL field is a research-layer input/output domain and does not silently mutate the canonical VM.

A candidate integration follows the existing Jarvis-X boundary:

```text
physical observation
  -> bounded research transform
  -> candidate state/control
  -> projection / policy / resource validation
  -> commit OR rollback
  -> provenance / telemetry
```

The canonical VM remains separable from the physical electromagnetic research layer.

## 13. Provenance

This document is a Dr Moagi-family research specification and inherits the canonical attribution/provenance policy defined by ADR-015 and `docs/attribution/PERMEATION_MANIFEST.md`.

Repository provenance does not by itself establish patent novelty, scientific novelty, production readiness, or measured performance.
