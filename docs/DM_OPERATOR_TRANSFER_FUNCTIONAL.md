# DM Operator Transfer Functional

## Status

Reference specification for ADR-014. This is a Layer-5 research contract and does not redefine the canonical Jarvis-X bytecode VM.

## 1. Governing expression

The supplied expression is normalized to:

```text
D_M(Psi)
  = nu
    Omega^(Xi+)
    (Lambda ⊗ Theta)^dagger
    K_Phi(Psi)
```

with `K_Phi(Psi) = Psi star Phi`.

A dimensionally clean operator-power form is:

```text
D_M(Psi)
  = nu
    (Omega / Omega0)^xi
    (Lambda ⊗ Theta)^dagger
    K_Phi(Psi).
```

## 2. State field

Let

```text
Psi : Omega_3 x R -> R^d
```

with `Omega_3` a bounded 3D lattice or continuous domain. In the discrete reference runtime:

```text
Psi[i,j,k]
```

is a scalar field sample.

## 3. Local geometric transform

For a finite 3D kernel `Phi[a,b,c]`:

```text
Y[i,j,k]
  = sum_{a,b,c} Phi[a,b,c]
                  Psi[i+a,j+b,k+c].
```

The reference runtime uses deterministic zero boundary conditions.

## 4. Constraint normalization

The rigorous full-space form is:

```text
Z = (Lambda ⊗ Theta)^dagger Y.
```

The dependency-free executable reference uses a scalar jointly-diagonal mode:

```text
C = lambda_gain * theta_gain
Z = Y / C.
```

The implementation rejects `|C| <= epsilon`.

## 5. Recursive operator power

Define the dimensionless memory ratio

```text
r_omega = omega / omega0.
```

Then

```text
G_omega = r_omega^xi.
```

The reference requires `r_omega > 0` for arbitrary real `xi`.

## 6. Output and recurrence

```text
D_M = nu * G_omega * Z
Psi_next = Psi + dt * D_M.
```

The complete local recurrence is therefore:

```text
Psi
 -> K_Phi
 -> constraint normalization
 -> recursive memory gain
 -> kinetic gain
 -> Euler state update
 -> Psi_next.
```

## 7. Transfer function

For translation-invariant diagonal modes:

```text
D_hat(k) = H(k) Psi_hat(k)
```

with

```text
H(k)
 = nu * G_omega(k) * Phi_hat(k)
   / (lambda(k) * theta(k)).
```

The magnitude

```text
|H(k)|
```

is the modal operator gain.

## 8. Discrete-loop stability

Because the reference recurrence is

```text
Psi_next = (I + dt A) Psi,
```

where

```text
A = nu * G_omega * C^dagger * K_Phi,
```

a mode with eigenvalue `a_k` advances with multiplier

```text
m_k = 1 + dt * a_k.
```

A sufficient per-mode discrete stability condition is

```text
|m_k| < 1.
```

For the scalar upper-bound estimate used by the reference runtime:

```text
|a_k| <= |nu| * |G_omega| * ||Phi||_1 / |C|.
```

This upper bound is conservative and does not prove stability when it exceeds one.

## 9. Fixed point

A fixed point satisfies

```text
Psi* = F(Psi*).
```

For the linear homogeneous reference recurrence this becomes

```text
A Psi* = 0.
```

Thus non-zero fixed points lie in the null space of the transfer operator. A contraction claim requires an explicit Jacobian/spectral bound; naming the operator “recursive” does not imply convergence.

## 10. Sensitivity

In one scalar mode:

```text
D = nu * (omega/omega0)^xi * Y / (lambda * theta).
```

Therefore:

```text
d ln|D| / d ln|nu|     = 1
d ln|D| / d ln|omega|  = xi
d ln|D| / d ln|lambda| = -1
d ln|D| / d ln|theta|  = -1
d ln|D| / d xi         = ln(omega/omega0).
```

These elasticities are directly testable.

## 11. Relationship to ADR-013

ADR-013 defines the broader state:

```text
S_t = (X_t, Z_t, V_t, Phi_t, Omega_t, Theta).
```

ADR-014 supplies one admissible Layer-5 transformation law for a field component within that architecture. It does not replace ADR-013 kinetic state, transactional validation, memory/reasoning timing separation or hardware-latency caveats.

## 12. Operational interpretation

```text
STATE FIELD
 -> LOCAL 3D CORRELATION
 -> CONSTRAINT NORMALIZATION
 -> RECURSIVE MEMORY POWER
 -> KINETIC GAIN
 -> STATE INCREMENT
 -> VERIFY
 -> COMMIT / ROLLBACK
 -> RE-ENTER.
```

The essential mathematical identity is:

```text
A
 = nu
   (Omega/Omega0)^xi
   (Lambda ⊗ Theta)^dagger
   K_Phi,

D_M = A Psi.
```
