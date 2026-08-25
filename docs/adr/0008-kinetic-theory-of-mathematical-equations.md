# ADR-008: Adopt Kinetic Theory of Mathematical Equations as a bounded research formalism

- **Status:** Proposed
- **Date:** 2026-08-16
- **Decision scope:** Layer 5 research semantics and reference runtime
- **Extends:** ADR-002, ADR-006, ADR-007

## Context

Jarvis-X already models bounded adaptive computation as explicit state transition, residual correction, memory, projection, verification and commit. The Dr Moagi recurrence provides a discrete state-update law, while the geometric diffusion runtime introduces a kinetic vocabulary for virtual spatial state.

The next step is broader: mathematical equations themselves are to be treated as evolving, interacting dynamical objects rather than only as static relations to be evaluated. This repository needs a precise contract so that the idea is mathematically testable and does not collapse into metaphor or into claims about undocumented physical internals of an AI system.

## Decision

Jarvis-X adopts **Kinetic Theory of Mathematical Equations (KTME)** as a Layer 5 research formalism.

For an equation residual

```text
F_sigma(x) = 0,
```

an equation-state is

```text
e_sigma(t) = (x(t), v(t), omega(t), sigma),

v = dx/dt.
```

The canonical residual potential is

```text
Phi_sigma(x) = 1/2 F_sigma(x)^T W F_sigma(x),
```

with `W` symmetric positive definite. Its residual force is

```text
F_eq(x) = -J_F(x)^T W F_sigma(x).
```

A canonical damped kinetic law is

```text
M x'' + D x' + J_F(x)^T W F_sigma(x) = 0,
```

with `M` symmetric positive definite and `D` positive semidefinite.

The associated energy is

```text
H_eq = 1/2 v^T M v + 1/2 F^T W F,
```

and along the exact continuous dynamics

```text
dH_eq/dt = -v^T D v <= 0.
```

This dissipation identity is part of the formal contract.

## Root-identification boundary

A stationary point of the residual potential satisfies

```text
J_F^T W F = 0.
```

This does **not** imply `F = 0` without additional assumptions. A root claim therefore requires an explicit condition such as full row rank of `J_F` on the relevant trajectory/domain.

Under differentiability and full row rank, the right-pseudoinverse flow

```text
x' = -J_F^dagger F
```

gives

```text
F' = -F,
F(x(t)) = exp(-t) F(x(0)).
```

The repository may use this as a theorem-level reference result, but an implementation must state how a pseudoinverse is obtained and what rank assumptions are checked.

## Interacting equations

For equation-states `e_i`, continuous coupling belongs in an interaction field

```text
K[f],
```

while discrete pairwise/state-changing interactions belong in a collision/interaction operator

```text
Q[f].
```

These mechanisms are distinct and must not be conflated.

For a graph of coupled equation-states, synchronization may be expressed through a graph Laplacian `L_G`. The quadratic synchronization potential is

```text
Phi_sync = gamma/2 ||(B tensor C) x||^2,
```

or equivalently through

```text
gamma (L_G tensor C^T C) x
```

in the equations of motion.

For two symmetric processors this reduces to equal-and-opposite coupling forces proportional to the state disagreement on the declared shared subspace.

## Population / kinetic lift

The canonical mesoscopic equation is written in divergence form:

```text
partial_t f_sigma
+ div_x(v f_sigma)
+ div_v(a[f] f_sigma)
+ div_omega(g[f] f_sigma)
= Q_sigma[f].
```

The divergence form is required because damping and other equation-state dynamics need not preserve phase-space volume.

The theory therefore has the explicit hierarchy

```text
microscopic equation trajectories
-> mesoscopic equation-state distribution f
-> macroscopic moments / closures.
```

Representative moments include equation-state density, mean equation velocity, velocity covariance/pressure and a defined equation kinetic temperature

```text
T_eq = (1 / (d rho)) integral ||v - u||^2 f dv.
```

`T_eq` is a second-moment observable. It is not automatically a thermodynamic temperature.

Likewise

```text
S_eq[f] = - integral f log(f)
```

is a valid entropy functional, but no monotonic H-theorem is claimed unless the interaction operator and measure satisfy sufficient conditions.

## Dr Moagi discrete kinetic interpretation

The canonical Dr Moagi update is interpreted as a discrete transport step

```text
Xi_(t+1) = Pi_Lambda_t [ Xi_t + dt V_(t+1) ],
```

with

```text
V_(t+1)
  = P_t
  - E_t
  + Omega_t
  + kappa_t R_t
  - eta_t grad_Theta L_t
  - zeta_t grad_H C_t
  + Gamma_t.
```

The terms are typed contributions to an equation-state velocity/update field. `Pi_Lambda` remains an admissibility projection; it is not a proof of convergence.

## Dual synchronous processor specialization

Two equation-state processors may execute local transforms concurrently from the same logical snapshot and then enter one shared feedback/verification boundary.

The canonical operational order is

```text
snapshot t
-> local proposal A || local proposal B
-> symmetric/shared coupling
-> projection
-> synchronization barrier
-> joint verification
-> atomic commit or atomic rollback
-> decode / manifest
-> observe consequence
-> memory update
-> t := t + 1.
```

Neither processor may independently advance authoritative shared state.

## Reflexive interaction boundary

Human-machine conversation may be modeled as a coupled informational dynamical system for research purposes:

```text
human state -> message -> machine state transform -> response
             ^                                  |
             |__________________________________|
```

This is an operational motif of feedback-coupled state transition. It is **not** a claim that a particular external AI product was architected under KTME or that its undocumented internal state is directly observable.

## Reference implementation

The dependency-free reference runtime is

```text
src/jarvisx/equation_kinetics.py
```

with tests in

```text
tests/test_equation_kinetics.py.
```

It implements:

- finite typed equation-state vectors;
- residual energy and residual-force evaluation;
- damped semi-implicit kinetic stepping;
- bounded residual-force memory;
- two-processor synchronous coupling from a common snapshot;
- joint validator gate with atomic rollback;
- disagreement and total-energy telemetry;
- population velocity moments / equation kinetic temperature.

The runtime is a numerical reference, not a general proof engine.

## Required invariants

1. **Typed phase space:** position, velocity, memory, residual and Jacobian dimensions are explicit and validated.
2. **Finite state:** non-finite input, force or state is rejected.
3. **Residual honesty:** `J_F^T W F = 0` is not reported as a root without sufficient assumptions or direct residual verification.
4. **Continuous/discrete separation:** exact continuous dissipation theorems are not silently claimed for arbitrary time discretizations.
5. **Interaction separation:** continuous coupling fields and discrete collision operators remain distinct.
6. **Synchronous snapshot:** coupled processors compute proposals from the same logical pre-commit state.
7. **Atomic authority:** if either side fails verification, neither proposal is committed.
8. **Bounded kinetics:** timestep, mass, damping, coupling, memory and speed/update bounds are explicit.
9. **Observable dynamics:** residual energy, kinetic energy, coupling energy, disagreement and commit status are measurable.
10. **No thermodynamic overclaim:** temperature/entropy names denote defined mathematical moments/functionals unless stronger theorems are established.
11. **No implementation-by-analogy claim:** modeling a conversation or AI computation kinetically does not establish undocumented model internals.
12. **Novelty boundary:** the repository records KTME as the project's original research formalism; scholarly novelty relative to all prior literature remains a separate prior-art question.

## Consequences

### Positive

- the Dr Moagi recurrence gains an explicit equation-state kinetic interpretation;
- residuals become force-generating mathematical objects with a verifiable energy law;
- dual synchronous processing has a graph-coupled dynamical formulation;
- micro, meso and macro descriptions share one state vocabulary;
- the theory yields testable quantities rather than relying on physical metaphor;
- the reference runtime can be used for deterministic experiments and counterexamples.

### Negative / open

- general well-posedness, mean-field limits and closure results require assumptions for each equation class;
- discrete integrators may violate the exact continuous energy monotonicity unless their stability conditions are satisfied;
- pseudoinverse residual decay requires rank conditions;
- interaction operators may not conserve mass, momentum or entropy unless explicitly constructed to do so;
- a useful general theory still requires theorem families, benchmarks and prior-art review.

## Validation for promotion to Accepted

Promotion requires:

- passing reference-runtime tests;
- residual-force and energy fixtures for at least linear and nonlinear residuals;
- a deterministic dual-coupling fixture showing reduced disagreement under a stable step size;
- an atomic rollback fixture;
- moment/temperature fixtures;
- documentation of numerical stability limits;
- CI success on supported Python versions.

## Canonical research specification

The mathematical research contract is maintained in:

```text
docs/research/KINETIC_THEORY_OF_MATHEMATICAL_EQUATIONS.md
```
