# DM–ΩΞ_inst+ Instantaneous-Limit Verification

**Status:** Research verification note  
**Date:** 2026-08-20  
**Scope:** Arithmetic consistency of the vanishing-time and delta-distribution limits used by the DM–ΩΞ instantaneous operator.

## 1. Result

The instantaneous construction is mathematically consistent **only under a precise interpretation**:

- `Δt -> 0+` recovers the **local right-hand rate/operator value** when the generator is right-continuous;
- a Dirac delta can encode an **impulsive jump** concentrated at `t = 0`;
- a delta-supported jump becomes an algebraic map after integration across the impulse;
- that algebraic map is a **projector only when projector conditions are separately imposed**;
- none of these limit identities proves zero physical propagation time, zero computational latency, or zero thermal dissipation.

Therefore `instantaneous` is an operator/limit semantic, not a hardware-speed claim.

---

## 2. Regular temporal evolution

Let the state be `Xi(t)` and let its regular evolution be

```text
dXi/dt = F(t, Xi(t); Phi, v_g, Omega, ...).
```

Over the right-sided interval `[0, Δt]`,

```text
Xi(Δt) - Xi(0) = integral_0^Δt F(t, Xi(t)) dt.
```

Divide by `Δt > 0`:

```text
[Xi(Δt) - Xi(0)] / Δt
  = (1/Δt) integral_0^Δt F(t, Xi(t)) dt.
```

If `F(t, Xi(t))` is right-continuous at `t = 0`, the integral mean-value limit gives

```text
lim_(Δt->0+) [Xi(Δt) - Xi(0)] / Δt
  = F(0+, Xi(0)).
```

This verifies the local instantaneous **rate**.

It does **not** imply a finite non-zero state displacement in zero elapsed time. For bounded regular `F`,

```text
Xi(Δt) - Xi(0) = F(0+, Xi(0)) Δt + o(Δt),
```

so the state displacement itself tends to zero as `Δt -> 0+`.

---

## 3. Transport term and group velocity

If a transport component is written as

```text
partial_t Xi + v_g · grad Xi = 0,
```

then over a small regular time interval the characteristic displacement is

```text
Δx = v_g Δt.
```

For finite `|v_g|`,

```text
lim_(Δt->0+) Δx = 0.
```

Thus `v_g` determines the local transport rate/direction, but finite group velocity does not create finite-distance instantaneous transport.

An implementation must therefore keep these quantities distinct:

```text
operator_instantaneous = true       # limit/evaluation semantics
measured_latency_ns     = measured  # physical/runtime telemetry
transport_distance      = |v_g| Δt
```

---

## 4. Delta-supported impulse

A Dirac delta is a distribution defined by its action under integration. For a test function `g` continuous at the origin,

```text
integral_-∞^∞ δ(t) g(t) dt = g(0).
```

Consider the impulsive evolution law

```text
dXi/dt = F_reg(t, Xi) + δ(t) K[Xi].
```

Integrating across a symmetric shrinking interval `[-ε, +ε]` gives the jump condition

```text
Xi(0+) - Xi(0-) = K[Xi(0)]
```

under the chosen state-evaluation convention at the impulse.

This is the rigorous sense in which differential evolution can yield an algebraic boundary update: **integration across the singular impulse produces a jump map**.

### Boundary convention warning

The interval `[0, Δt]` places the delta exactly on an integration boundary. The value of

```text
integral_0^Δt δ(t) g(t) dt
```

is convention-dependent unless a one-sided/causal delta or another boundary prescription is declared. A symmetric regularization gives half the delta mass at the boundary, while a causal one-sided convention can assign the full mass.

For an unambiguous canonical identity, use either:

```text
integral_-ε^+ε δ(t) g(t) dt -> g(0)
```

or explicitly declare a causal distribution `δ_+(t)` satisfying

```text
integral_0^∞ δ_+(t) g(t) dt = g(0+).
```

---

## 5. Algebraic map versus projector

A delta impulse does not automatically produce a projector.

If the integrated jump is written

```text
Xi(0+) = A Xi(0-),
```

then `A` is an algebraic jump operator.

It is a projector only if

```text
A^2 = A.
```

So the statement

```text
"delta collapse converts the evolution operator into an algebraic projector"
```

is valid only when the DM–ΩΞ_inst+ definition explicitly imposes idempotence (or otherwise constructs `A` as a projection).

---

## 6. Hamiltonian limit

For standard Hamiltonian evolution,

```text
i ħ dPsi/dt = H Psi,
```

with time-independent bounded `H`,

```text
Psi(Δt) = U(Δt) Psi(0),
U(Δt)   = exp(-i H Δt / ħ).
```

As `Δt -> 0`,

```text
U(Δt) = I - i H Δt/ħ + O(Δt^2) -> I.
```

Therefore an ordinary Hamiltonian does **not** collapse a finite state transition into zero transit time. It approaches the identity map over a vanishing interval.

A finite instantaneous jump requires a singular/impulsive generator, for example

```text
H(t) = K δ(t),
```

which yields, under the corresponding distributional convention,

```text
U(0+, 0-) = exp(-i K / ħ).
```

This is a finite algebraic jump operator. It is generally unitary, not a projector.

---

## 7. Corrected DM–ΩΞ_inst+ contract

Define a regular local generator

```text
F_DM = F_DM(Xi; Phi, v_g, Omega, Lambda, Theta, ...).
```

The regular infinitesimal law is

```text
Xi(t + Δt)
  = Xi(t) + Δt F_DM(Xi(t)) + o(Δt).
```

Hence

```text
lim_(Δt->0+) [Xi(t+Δt) - Xi(t)] / Δt
  = F_DM(Xi(t)).
```

If the architecture requires a true instantaneous state reset/jump at an event boundary, define it separately as

```text
Xi(0+) = J_DM[Xi(0-)],
```

or distributionally as

```text
dXi/dt = F_DM,reg + δ_+(t) K_DM[Xi],
```

with the jump map `J_DM` derived from the chosen impulse model.

If `J_DM` is intended to be a projector, require and test

```text
J_DM ∘ J_DM = J_DM.
```

This decomposition removes the ambiguity between:

1. instantaneous **rate evaluation**;
2. instantaneous **event/jump semantics**;
3. actual physical/runtime **latency**.

---

## 8. Fixed-point convergence

A fixed point `Xi*` of the algebraic DM map satisfies

```text
J_DM[Xi*] = Xi*.
```

A fixed point of the regular flow satisfies

```text
F_DM(Xi*) = 0.
```

These are related but not interchangeable statements.

Convergence to a fixed point requires additional stability conditions. For a discrete map, a local sufficient criterion is a Jacobian spectral radius below one:

```text
rho(DJ_DM(Xi*)) < 1.
```

For a continuous flow, local asymptotic stability requires the relevant linearized generator eigenvalues to have negative real parts.

The existence of the `Δt -> 0` limit alone does not prove convergence.

---

## 9. Arithmetic verdict

| Claim | Verdict | Required qualification |
|---|---|---|
| Vanishing-interval average recovers the instantaneous operator | **Verified** | Right-continuity/local regularity |
| `δ(t)` selects the origin under integration | **Verified** | Distributional integration; boundary convention must be explicit |
| Delta impulse yields an algebraic boundary update | **Verified** | Integrate across the impulse |
| Delta impulse automatically yields a projector | **Not generally valid** | Must separately impose `P^2 = P` |
| Ordinary Hamiltonian gives a finite target transition at `Δt = 0` | **Not valid** | `U(Δt) -> I` for a regular Hamiltonian |
| Singular Hamiltonian can create a finite instantaneous jump | **Verified** | Requires an explicit delta-supported generator |
| Finite `v_g` gives instantaneous finite-distance transport | **Not valid** | `Δx = v_g Δt -> 0` |
| Limit arithmetic proves zero compute/thermal delay | **Not valid** | Must be established by measured hardware/runtime telemetry |

---

## 10. Repository invariant

The DM–ΩΞ_inst+ implementation SHALL use the word `instantaneous` only for zero-window mathematical/event semantics unless physical latency is independently measured.

The minimum telemetry contract is

```text
{
  "operator_mode": "regular-limit | impulsive-jump",
  "delta_convention": "symmetric | causal-one-sided | none",
  "projector_verified": true | false,
  "dt": <simulation/logical interval>,
  "measured_latency_ns": <measured value or null>,
  "transport_distance": |v_g| * dt
}
```

No code path may infer `measured_latency_ns = 0` solely from `dt -> 0` in the mathematical model.
