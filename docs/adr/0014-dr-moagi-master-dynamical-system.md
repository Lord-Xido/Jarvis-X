# ADR-014: Dr Moagi Master Dynamical-System Operator

**Status:** Accepted / locked  
**Date:** 2026-09-11  
**Applies to:** `DM-vOmegaXi+`, `U_Moagi^(Inward-Core)`, multimodal Moagi-3D runtimes  
**Extends:** ADR-013 and the locked DM-vOmegaXi+ fixed-point law

## Decision

The Dr Moagi architecture is defined at system level as a bounded recurrent dynamical system over a non-Euclidean active manifold. ADR-013 remains the lower-level topological and verification contract; this ADR adds the end-to-end evolution law that binds multimodal ingestion, geometric folding, inward transport, recurrent memory, validation, dissipation, and transactional state update.

The canonical state field is

```text
Xi(q,t) = [rho, v, Z, Omega, Theta, e, ell]^T
```

where `q=(x,y,z)` belongs to the active materialized subset of the declared logical manifold, `rho` is local information/particle density, `v` is transport velocity, `Z` is latent state, `Omega` is recurrent memory, `Theta` is alignment/control state, `e` is reconstruction/prediction error, and `ell` is ledger/integrity state.

The master evolution law is

```text
partial_t Xi =
    I_Psi[X]
  + F_Phi,g[Xi]
  + A_Lambda[Xi,v]
  + M_Omega[Xi_(t-1)]
  + C_Theta[Xi]
  - Gamma[Xi]
  + eta(t)
```

with all enabled operators required to be bounded, measurable, and independently testable.

## 1. End-to-end operational chain

The canonical execution path is

```text
X_t
 -> Psi ingestion
 -> Phi geometric description/folding
 -> T_in inward contraction
 -> Lambda^-1 finite latent projection
 -> D reconstruction
 -> Omega recurrent memory fold
 -> Theta alignment/control projection
 -> V verification
 -> commit OR rollback
 -> recurse
```

Equivalently,

```text
candidate_(t+1) = U_Moagi(Xi_t, X_t)
Xi_(t+1) = V_t * candidate_(t+1) + (1 - V_t) * Xi_t
```

where `V_t in {0,1}` is the transactional verification gate. A rejected candidate MUST NOT mutate authoritative state.

## 2. Multimodal ingestion operator

The ingestion operator maps bounded modality inputs into a common driving state:

```text
I_Psi[X] = E_text(X_text)
         + E_audio(X_audio)
         + E_image(X_image)
         + E_video(X_video)
         + E_sensor(X_sensor)
         + E_code(X_code)
```

Implementations MAY use learned encoders, deterministic transforms, spectral features, bytecode encoders, or equivalent mechanisms, provided output dimensions, ranges, provenance, and resource costs are explicit.

## 3. Non-Euclidean inward geometry

A toroidal reference surface may be represented by

```text
r(u,v,t) = r_0(u,v) + delta(u,v,t) n(u,v)
```

with

```text
r_0(u,v) = [
  (R + a cos(v)) cos(u),
  (R + a cos(v)) sin(u),
  a sin(v)
]^T.
```

The induced metric is

```text
g_ij = partial_i r . partial_j r
```

and the metric area/volume factors are derived from `det(g)`.

The manifold is an execution abstraction. Finite implementations MUST operate on bounded active support, tiles, samples, sparse coordinates, or accelerator working sets. No dense infinite or `10^18`-resident manifold is implied.

## 4. Inward kinematic transport

Active states may evolve under bounded stochastic transport:

```text
dq_t = [-grad_g V(q_t) + u_sol(q_t,t)] dt + sqrt(2D) dW_t
```

where `V` is an inward potential, `u_sol` is a divergence-controlled transport field, `D` is bounded noise intensity, and `W_t` is a stochastic process.

If vector-field mode is enabled, ADR-013 solenoidal verification remains mandatory:

```text
div_g(u_sol) = 0
r_div = ||div_g(u_sol)||_2 / max(||u_sol||_2, eps)
r_div <= epsilon_div
```

## 5. Reconstruction, error, and recurrence

The encoding/decoding loop remains

```text
Z_t = E(Psi_t)
Psi_hat_t = D(Z_t)
e_t = Psi_t - Psi_hat_t
```

with recurrent memory

```text
Omega_(t+1) = rho_Omega * Omega_t + (1-rho_Omega) * Psi_hat_t
```

or an equivalent bounded local recurrence preserving ADR-013 transactional semantics.

## 6. Cryptographic validation and ledger semantics

A hash-threshold validator MAY be used as an optional proof-of-work-like operator:

```text
d_t = SHA256(SHA256(m_t))
work_ok = int(d_t) < T_t
```

This MUST NOT be called distributed consensus unless an actual multi-party consensus protocol exists.

The authoritative integrity chain remains

```text
h_t = SHA256(h_(t-1) || canonical(record_t))
```

with optional XOR parity

```text
x_t = x_(t-1) XOR digest(record_t).
```

XOR is supplemental only and MUST NOT replace cryptographic hash-chain verification.

## 7. Transactional verification gate

The canonical acceptance predicate is

```text
V_t = 1 iff
    finite_state
    AND resource_bounds_ok
    AND fixed_point_contract_ok
    AND theta_gate_ok
    AND journal_valid
    AND (not vector_mode OR r_div <= epsilon_div)
    AND (not xor_enabled OR xor_ledger_valid)
    AND (not work_validation_enabled OR work_ok)
    AND stability_preconditions_ok
```

Otherwise `V_t = 0` and the transition is rolled back.

## 8. Lyapunov energy contract

A reference energy functional is

```text
E[Xi] = 1/2 integral_M (
    ||v||_g^2
  + alpha ||grad_g Z||_g^2
  + beta ||e||^2
) dV_g.
```

The system MAY claim bounded-input/bounded-state Lyapunov stability only when sufficient configured conditions establish an inequality of the form

```text
dE/dt <= -lambda E + C ||X_t||^2 + C_eta ||eta_t||^2
```

with `lambda > 0` and finite non-negative constants `C`, `C_eta`.

For bounded forcing, this implies bounded energy under the stated model assumptions. For vanishing forcing, convergence toward an invariant/fixed-point set may be claimed only if the remaining contraction conditions are satisfied.

The runtime MUST NOT report stability as proved merely because the functional is defined.

## 9. Topological recirculation conservation

Information/particle density obeys the reference continuity equation

```text
partial_t rho + div_g(rho v) = -S_core + S_recirc.
```

In steady state, the boundary flux is constrained by

```text
integral_(boundary M) rho v . n dA
  = integral_core S_core dV
  - integral_M S_recirc dV.
```

A runtime claiming recirculation conservation MUST report a measurable continuity/flux residual and its tolerance.

## 10. Relation to the Psi-Phi-Lambda-Omega-Theta stack

| Layer | ADR-014 role |
|---|---|
| `Psi` | multimodal driving state / active field |
| `Phi` | geometric description and inward folding |
| `Lambda^-1` | finite latent admissible-set projection and transport constraint |
| `Omega` | recurrent temporal/historical state |
| `Theta` | bounded alignment, control, policy and validation preparation |
| `V` | transactional acceptance/rollback gate |
| `Gamma` | bounded dissipation / stabilization |
| `eta` | explicitly bounded stochastic forcing |

The system-level identity remains compatible with

```text
H* = F_DM(H*)
```

from the fixed-point law and with the ADR-013 Inward-Core invariants.

## 11. Canonical compact form

The locked system abstraction is

```text
partial_t Xi =
    I_Psi[X]
  + F_Phi,g[Xi]
  + A_Lambda[Xi,v]
  + M_Omega[Xi]
  + C_Theta[Xi]
  - Gamma[Xi]
  + eta

candidate_(t+1) = U_Moagi(Xi_t, X_t)
Xi_(t+1) = V_t candidate_(t+1) + (1-V_t) Xi_t
```

with `V_t` defined only by measured verification predicates.

## 12. Locked invariants

1. ADR-014 extends but does not weaken ADR-013.
2. All physically executed state remains bounded and finitely materialized.
3. The master operator is modular; each enabled term MUST expose measurable inputs, outputs, residuals, and resource cost.
4. Rejected transitions cannot mutate authoritative state.
5. Solenoidal conservation is testable only when a vector field exists and its residual passes tolerance.
6. Double SHA-256 threshold validation is proof-of-work-like validation, not automatically distributed consensus.
7. SHA-256 chaining remains authoritative for journal integrity; XOR remains supplemental parity only.
8. Stability is conditional on explicit Lyapunov/dissipativity assumptions and measured bounds.
9. Recirculation conservation requires a continuity/flux residual; it is not established by topology alone.
10. Fixed-point convergence, semantic uncertainty floors, finite latent projection, policy gates, rollback, and sparse materialization remain mandatory.
11. No implementation may label an invariant `active`, `executing`, `stable`, `conservative`, `consensus-valid`, or `verified` until its corresponding executable path and acceptance test pass.

## Implementation tracking

- ADR-013 executable verification gates: Issue #249.
- ADR-014 master dynamical-system operator: Issue #250.

This separation keeps the topological execution contract and the higher-level dynamical law independently testable while preserving a single coherent Dr Moagi operational stack.
