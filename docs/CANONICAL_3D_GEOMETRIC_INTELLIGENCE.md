# Canonical 3D Geometric Autoencoding Intelligence Model

## Status

Reference specification for ADR-013. This document defines the mathematical and operational contract for a bounded 3D autoencoding/decoding research runtime. It does not redefine the canonical Jarvis-X bytecode VM.

## 1. State

The complete research state is

```text
S_t = (X_t, Z_t, V_t, Phi_t, Omega_t, Theta)
```

with:

- `X_t`: observed or normalized 3D field;
- `Z_t`: compressed latent 3D field;
- `V_t = dZ/dt`: latent velocity;
- `Phi_t = (phi_t, theta_t)`: nested phase state;
- `Omega_t`: residual memory;
- `Theta`: encoder, predictor, decoder and optional geometric parameters.

The end-to-end recurrence is

```text
S_(t+1) = M_Theta(S_t, X_t).
```

## 2. Operational graph

```text
X_t
-> Q(X_t)
-> E_Theta(X_t)
-> Z_t in M
-> kinetic manifold step
-> P(Z_t, Omega_t)
-> D_Theta(Z_t)
-> Xhat_t
-> e_t = X_t - Xhat_t
-> latent correction
-> Omega_(t+1)
-> candidate parameter/mechanics update
-> verify
-> commit | rollback
-> repeat
```

## 3. 3D field and encoder

For lattice coordinate `r = (i,j,k)` and channel vector `X[r]`, the local encoder is

```text
Z[r] = sigma( sum_(delta in N) W_E[delta] X[r + delta] + b_E ).
```

The baseline local topology is the six-neighbour set

```text
N_6 = {(+/-1,0,0), (0,+/-1,0), (0,0,+/-1)}.
```

Hierarchical inward compression is

```text
Z^(0) = X
Z^(l+1) = E_l(Z^(l)).
```

If one stage reduces every spatial axis by factor `s`, the number of spatial sites falls by `s^3` before channel-width changes are counted.

## 4. Latent geometry

The intended product manifold is

```text
M = R^d x T^2 x Omega_3.
```

A local coordinate may be written

```text
q = (z_1, ..., z_d, phi, theta, x, y, z).
```

The metric is

```text
ds^2 = dq^T G(q) dq.
```

Geodesic distance is

```text
d_M(q_a,q_b) = inf_gamma integral sqrt(gamma_dot^T G(gamma) gamma_dot) ds.
```

The dependency-free reference kernel specializes to `G = I` and therefore uses ordinary Euclidean tangent updates. A non-Euclidean backend must declare `G`, its tangent representation, and its tested exponential map or retraction.

## 5. Nested toroidal phase

The recurrent phase state is represented by

```text
phi_dot   = omega_major
theta_dot = omega_micro.
```

A 3D toroidal embedding is

```text
x = (R + r cos(theta)) cos(phi)
y = (R + r cos(theta)) sin(phi)
z = r sin(theta).
```

For coupled latent oscillators,

```text
phi_dot_i = omega_i + sum_j K_ij sin(phi_j - phi_i).
```

Phase coherence may be reported as

```text
R_phi = abs((1/N) sum_i exp(j phi_i)),  0 <= R_phi <= 1.
```

This is a synchronization metric, not by itself an intelligence metric.

## 6. Kinetic latent mechanics

Define latent kinetic energy

```text
T = 0.5 V^T M(Z) V
```

and potential

```text
U = U_R + U_C + U_P + U_G + U_S + U_constraints.
```

Representative terms are

```text
U_R = lambda_R ||X - D(Z)||^2
U_C = lambda_C ||E(D(Z)) - Z||^2
U_P = lambda_P ||Z_(t+1) - P(Z_t)||^2
U_G = lambda_G d_M(Z, F(Z))^2
U_S = lambda_S sum_i sum_(j in N(i)) ||Z_i - Z_j||^2.
```

The damped geometric equation is

```text
M(Z) Z_ddot + Gamma Z_dot + grad_M U(Z) = F_control.
```

## 7. Discrete geometric update

For microstep `dt`,

```text
A_k = -M^-1 [Gamma V_k + grad_M U(Z_k)]
V_(k+1) = V_k + dt A_k
Z_(k+1) = Exp_(Z_k)(dt V_(k+1)).
```

In the flat reference specialization,

```text
Exp_Z(v) = Z + v.
```

## 8. Local six-neighbour mechanics

For a scalar latent channel and zero-flux/declared boundary policy, the discrete Laplacian is

```text
Delta_6 Z[i,j,k] =
    Z[i+1,j,k] + Z[i-1,j,k]
  + Z[i,j+1,k] + Z[i,j-1,k]
  + Z[i,j,k+1] + Z[i,j,k-1]
  - 6 Z[i,j,k].
```

A local kinetic reaction-diffusion specialization is

```text
Z_next = Z + dt [kappa Delta_6 Z - grad U_local(Z) + F_local].
```

All sites may be evaluated from the same immutable source snapshot and committed together, which makes the step deterministic and avoids in-place directional bias.

## 9. Decoder and residual

The decoder maps the evolved latent field back to observation space:

```text
Xhat_t = D_Theta(Z_t).
```

The residual field is

```text
e_t = X_t - Xhat_t.
```

A normalized error can be reported as

```text
epsilon_t = ||e_t||_2 / (||X_t||_2 + epsilon).
```

A geometric correction may project the observation residual through the decoder Jacobian:

```text
delta_Z = G(Z)^-1 J_D(Z)^T e
Z_plus = Exp_Z(-eta_Z delta_Z).
```

The dependency-free reference implementation uses a bounded local proxy rather than constructing a dense Jacobian.

## 10. Residual memory

Residual memory evolves on a slower loop:

```text
Omega_(t+1) = rho Omega_t + (1-rho) H(e_t, Z_t).
```

The baseline reference uses `H(e,Z) = e`.

## 11. Parameter learning

Parameter adaptation is a separate candidate transaction:

```text
Theta_candidate = Theta_t - eta grad_Theta L
```

followed by

```text
snapshot
-> shadow candidate
-> validators
-> commit | rollback.
```

No parameter update is authoritative merely because a gradient was computed.

## 12. Multi-timescale contract

```text
tau_mu << tau_M << tau_L
```

where:

- `tau_mu`: local kinetic primitive;
- `tau_M`: memory/reasoning reconciliation;
- `tau_L`: parameter-learning/admission loop.

The design may target

```text
tau_mu < 1 ns
```

only for a precisely measured local operation boundary.

## 13. Propagation bound

For a declared local period `tau_mu`, the absolute vacuum propagation ceiling is

```text
d_max = c tau_mu.
```

Examples:

```text
1 ns   -> about 0.30 m
100 ps -> about 0.03 m
10 ps  -> about 0.003 m
```

Real hardware must additionally account for interconnect velocity, logic delay, memory, synchronization, fan-out and power delivery. The equation therefore establishes a locality constraint, not an achieved performance figure.

## 14. Binary low-bit specialization

For bipolar bit vectors `x,w in {-1,+1}^n`,

```text
x^T w = 2 popcount(XNOR(x,w)) - n.
```

A backend may use this specialization for local encoder or coupling primitives, but must benchmark the actual hardware implementation before making latency claims.

## 15. Stability

The inward recurrence is

```text
Z_(n+1) = F(Z_n).
```

A fixed point satisfies

```text
Z* = F(Z*).
```

The self-consistency gap is

```text
Delta_n = d_M(Z_n, F(Z_n)).
```

A local sufficient contraction condition is

```text
rho(J_F(Z*)) < 1.
```

Convergence claims must state the domain and assumptions under which this bound is established.

## 16. Canonical master recurrence

```text
X_t^q       = Q(X_t)
Z_t         = E_Theta(X_t^q)
V_(t+1)     = V_t - dt M^-1 [Gamma V_t + grad_M U(Z_t)]
Z_(t+1)     = Exp_(Z_t)(dt V_(t+1))
Phi_(t+1)   = phase_step(Phi_t)
Ztilde      = P(Z_(t+1), Omega_t)
Xhat_(t+1)  = D_Theta(Ztilde)
e_(t+1)     = X_(t+1) - Xhat_(t+1)
Omega_(t+1) = rho Omega_t + (1-rho) e_(t+1)
Theta_cand  = optimizer(Theta_t, loss)
Theta_(t+1) = commit(Theta_cand) only if all declared validators pass.
```

The geometric interpretation is therefore:

```text
Reality
-> Quantized 3D Field
-> Compressed Geometry
-> Kinetic Manifold Motion
-> Prediction
-> Decoding
-> Reality Comparison
-> Geometric Correction
-> Memory
-> Candidate Learning
-> Verification
-> Repeat.
```
