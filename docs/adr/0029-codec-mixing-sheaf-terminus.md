# ADR-029: Codec mixing, physical bottlenecks and audited TERMINUS fixed point

**Status:** Proposed  
**Date:** 2026-09-24  
**Scope:** Layer-5 bounded research operators  
**Reference implementation:** `src/jarvisx/codec_terminus.py`

## Context

Jarvis-X already separates representation, execution, observation and candidate-first
adaptation. This ADR integrates a further research chain in a form that preserves those
boundaries:

```text
X --E--> Y --T--> Y --D--> X
                     |
                     v
              R = D o T o E
                     |
                     v
              p_(t+1) = R p_t
                     |
                     v
 nu_eff = min(nu_syn, nu_bw, nu_q, nu_th, nu_ch)
                     |
                     v
 (x,y,z) = (position, bitplane, stage)
                     |
                     v
        C(n) = c*n, c ~= 4
                     |
                     v
     local sections / cover nerve
                     |
                     v
 Audit: overlap consistency and H^1 = 0
                     |
                     v
        bounded Kleene iteration
                     |
                     v
 P* = argmin C(P), audit(P), S(P)=P
                     |
                     v
                 TERMINUS
```

The original shorthand requires several mathematical qualifications before it can become
an executable repository contract. In particular, `Theta(n)` is an asymptotic class and
cannot itself be a constant, a mixing-time theorem needs assumptions not available from a
matrix name alone, and `H^1 = 0` is not a universal synonym for all forms of software
correctness.

## Decision

### 1. Finite codec/Markov operator

The reference kernel uses column-vector probabilities. The encoder, transport and decoder
are therefore column-stochastic matrices:

```text
E : Delta(X) -> Delta(Y)
T : Delta(Y) -> Delta(Y)
D : Delta(Y) -> Delta(X)

R = D T E : Delta(X) -> Delta(X)
p_(t+1) = R p_t
```

All matrices must be finite, non-negative and column-stochastic. Shape mismatches fail
closed.

The runtime may report empirical convergence of a supplied iteration. It does **not**
infer irreducibility, aperiodicity, uniqueness of the stationary distribution, a spectral
gap `gamma`, or a bound such as `O(log |X| / gamma)`. Such a theorem is admitted only
when its assumptions are established separately.

### 2. Five-bound physical throughput

Effective rate is the active physical bottleneck:

```text
nu_eff = min(nu_syn, nu_bw, nu_q, nu_th, nu_ch)
```

where the reference names are synthesis, bandwidth, quantization, thermal and channel.
All rates are finite and strictly positive. Tied minima are reported as multiple active
bounds.

### 3. Geometric realization and cost

The default spatial semantics are:

```text
x = position
y = bitplane
z = stage
```

The cost claim is represented as

```text
C(n) = c n
c ~= 4
C(n) in Theta(n)
```

The coefficient `c` is a declared/measured normalized cost parameter. The architecture
does not claim that `Theta(n)` is itself constant.

### 4. Conservative sheaf/Čech audit

The executable audit uses a finite cover of named local sections. Two regions are adjacent
in the cover nerve when they share at least one key. Values on every shared key must agree.

For the resulting **1-dimensional nerve graph with constant field coefficients**,

```text
dim H^1 = |E| - |V| + number_of_connected_components.
```

The admission gate requires both:

```text
no overlap conflicts
and
dim H^1 = 0.
```

This is an exact graph-nerve special case and a conservative repository gate. It is not a
general sheaf-cohomology implementation and does not establish all possible correctness
properties.

### 5. Kleene-style refinement

Given an ordered domain `(P, <=)`, the runtime performs bounded iteration

```text
P_0 = bottom
P_(k+1) = S(P_k)
```

and verifies that the realized chain is ascending. Equality terminates the run as a fixed
point.

The ascending-chain check is runtime evidence for the visited states. It is not a proof
that `S` is globally monotone or Scott-continuous; those remain mathematical obligations
of a caller that wants the full Kleene fixed-point theorem.

### 6. TERMINUS selection

An eligible candidate must satisfy

```text
Audit(P) = pass
S(P) = P.
```

Among eligible candidates the reference selector computes

```text
P* in argmin_P C(P).
```

Deterministic representation order breaks equal-cost ties.

TERMINUS therefore means the repository-visible intersection

```text
fixed point
AND
declared audit consistency
AND
minimum cost over the supplied eligible candidate set.
```

It does not imply global optimality over states that were never supplied or explored.

### 7. Closed-class / external-signal boundary

For an autonomous operator `S` and a class `C` satisfying

```text
S(C) subseteq C,
```

a trajectory initialized in `C` cannot leave `C` under repeated application of that
same operator.

Escape requires an explicit change in the transition relation: for example an external
input, a new operator/version, or an admitted stochastic/non-autonomous transition.
External signals remain typed inputs and pass the normal Jarvis-X policy/transaction
boundary.

## Consequences

- The codec chain is executable without conflating a reconstruction operator with a proof
  of rapid mixing.
- Physical-rate claims expose the active bottleneck directly.
- Geometry and asymptotic cost are represented without the `Theta(n) = constant`
  contradiction.
- The `H^1` statement is narrowed to a computable special case with explicit limits.
- Fixed-point refinement remains bounded, fail-closed and subordinate to candidate-first
  admission.
- TERMINUS is a receipt over a supplied candidate set, not an unrestricted self-modifying
  authority.
- The canonical VM remains independent of this research layer.

## Validation

`tests/test_codec_terminus.py` verifies:

- stochastic codec composition and convergence on a known two-state chain;
- rejection of malformed stochastic components;
- the five-bound minimum-rate contract;
- default 3D axis semantics and normalized linear cost;
- consistent acyclic covers with `H^1 = 0`;
- overlap-conflict rejection;
- non-zero graph-nerve `H^1` rejection;
- bounded ascending Kleene iteration and fail-closed descending steps;
- minimum-cost selection restricted to audited fixed points.

## Non-goals

This ADR does not claim:

- general Markov-chain mixing bounds without spectral/ergodicity hypotheses;
- general sheaf cohomology over arbitrary sites, coefficient categories or higher nerves;
- a proof that coefficient `c ~= 4` is universally optimal;
- global optimization over an unbounded state space;
- unrestricted source-code mutation, native actuation or bypass of Jarvis-X policy,
  validation, provenance and rollback boundaries.
