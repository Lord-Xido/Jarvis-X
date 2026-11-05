# ADR-033: Bounded 3D Cognitive Constraint Process

- **Status:** Accepted for reference implementation
- **Date:** 2026-10-02
- **Applies to:** Dr Moagi Cognitive Engine, relational reasoning, fixed-point verification, CTR evidence
- **Extends:** ADR-028 and `docs/DR_MOAGI_COGNITIVE_ENGINE.md`

## Context

Jarvis-X already distinguishes latent representation, relational state, decoded
prediction, evidence, and authoritative state. A recurring requirement is to
make small symbolic reasoning chains spatially inspectable without confusing a
visual embedding with physical cognition.

The motivating geometry example starts with two observations:

[
e_{top}=130^circ,qquad r=50^circ.
]

The desired conclusion is the lower exterior angle (d). The reasoning chain is

[
130^circ ightarrow 50^circ ightarrow 80^circ ightarrow 100^circ.
]

## Decision

Add a bounded computational process that places scalar reasoning variables on
an explicit three-axis software chart:

[
(x,y,z)=
(	ext{perceptual structure},
 	ext{relational structure},
 	ext{abstraction depth}).
]

The chart coordinate is metadata used for inspection, visualization, routing,
and lineage. It does **not** claim that cognition is physically three
dimensional.

A constraint is represented as a declared linear equality

[
sum_i a_i v_i = b.
]

At iteration (k), if exactly one variable in a constraint is unknown, the
runtime infers it by

[
v_j =
rac{b-sum_{i
e j}a_i v_i}{a_j}.
]

Propagation continues until no new values are produced or the configured
iteration bound is reached.

The software state is

[
S_k=(V_k,mathcal C,mathcal P,T_k),
]

where (V_k) is the current value map, (mathcal C) is the constraint set,
(mathcal P) is the 3D coordinate map, and (T_k) is the inference lineage.

The fixed-point operator is

[
S_{k+1}=Phi_{mathrm{constraint}}(S_k).
]

A state is accepted only when:

[
V_{mathrm{accept}}=
V_{mathrm{fixed point}}
land
V_{mathrm{resolved}}
land
V_{mathrm{all constraints evaluable}}
land
V_{mathrm{residual}leepsilon}.
]

Thus stabilization alone is insufficient; an underdetermined or inconsistent
state fails closed.

## Angle demonstration

The reference implementation declares:

[
e_{top}+a=180^circ,
]

[
a+r+b=180^circ,
]

[
b+d=180^circ.
]

With (e_{top}=130^circ) and (r=50^circ), propagation yields:

[
a=50^circ,
qquad
b=80^circ,
qquad
oxed{d=100^circ}.
]

The inference trajectory is retained as an auditable sequence of
`CognitiveInferenceStep` receipts.

## Implementation

Reference module:

`src/jarvisx/cognitive_constraint_field3d.py`

Primary API:

```python
from jarvisx.cognitive_constraint_field3d import angle_demo_130_50

receipt = angle_demo_130_50()
assert receipt.accepted
assert receipt.value("d") == 100.0
```

The module is dependency-free and bounded by `max_iterations`. It does not
mutate authoritative runtime state.

## Integration with DMCE

The process fits into the cognitive engine as:

```text
observation
  -> symbolic variables
  -> 3D cognitive chart
  -> declared constraints
  -> bounded propagation
  -> fixed-point receipt
  -> CTR evidence
  -> Pi_Lambda
  -> COMMIT / ROLLBACK
```

For general domains, the linear solver is a reference kernel rather than a
complete reasoning system. Nonlinear, probabilistic, learned, or
domain-specific operators should preserve the same candidate-first,
lineage-preserving, fail-closed contract.

## Consequences

### Positive

- Converts a visual reasoning chain into executable, testable software.
- Preserves an explicit 3D process representation without metaphysical claims.
- Produces deterministic inference lineage and numerical residuals.
- Rejects inconsistent and underdetermined terminal states.
- Provides a reusable kernel for geometry, algebra, resource constraints, and
  other scalar relational problems.

### Limits

- Linear equalities do not cover arbitrary symbolic reasoning.
- 3D coordinates do not improve correctness by themselves.
- A numerically valid fixed point is not external-world truth.
- Domain validity still requires evidence and CTR verification.
