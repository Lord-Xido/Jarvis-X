# Suspensory Consideration Algebra (SCA)

**Status:** reference implementation / candidate-first research boundary. Not an autonomous decision authority.

An *idea in suspension* is neither an executing function nor a forgotten function. Its consequences can be calculated in a separate arithmetic model while the authoritative state is held constant. This formalizes the distinction between **consideration** and **commitment**.

## Mathematical contract

Let the authoritative sparse 3D field be `X_t: Λ → ℝ`, with `Λ = {0,...,N-1}³`. Only a bounded subset `A ⊂ Λ` is materialized. A proposed arithmetic hypothesis `h` is a pure transform `h(v) = a v + b`, applied to `A` only. The suspension operator produces an immutable candidate:

```text
S_h(X_t) = (X_t, C_h(X_t), D=0)
C_h(X_t)[p] = a X_t[p] + b, for p in A
```

No update to `X_t` occurs during `consider`. Nested consideration is function composition in hypothetical space:

```text
C_{h2 ∘ h1}(X) = h2(h1(X))
```

The commit transition is **separate**:

```text
X_(t+1) = (1 - D) X_t + D C_h(X_t)
```

where `D=1` is admissible only when there is explicit `authorization=True`, a matching candidate digest, a matching originating revision and base-state digest, and an in-bounds finite candidate. Otherwise `D=0`. A SHA-256 hash checks state identity, not truth, model suitability, source provenance, or authorization outside this API.

## API example

```python
from jarvisx.suspensory_algebra import AffineHypothesis, SuspensoryAlgebra

X = (1, 2, 3)
engine = SuspensoryAlgebra({X: 4.0}, side=100)

idea_a = engine.consider("A", AffineHypothesis("2x+3", scale=2, bias=3))
idea_b = engine.consider("B", AffineHypothesis("4x-3", scale=4, bias=-3))
assert idea_a.snapshot()[X] == 11
assert idea_b.snapshot()[X] == 13
assert engine.snapshot()[X] == 4  # neither idea was executed

# Refine A hypothetically without modifying reality:
idea_a2 = engine.consider(
    "A2", AffineHypothesis("3x-1", scale=3, bias=-1), parent_id="A"
)
assert idea_a2.snapshot()[X] == 32
assert engine.snapshot()[X] == 4

# The default closed gate denies commitment:
assert not engine.commit("A").committed

# Caller explicitly chooses the reviewed candidate:
decision = engine.commit(
    "A", authorization=True, expected_hash=idea_a.candidate_hash
)
assert decision.committed and engine.snapshot()[X] == 11
```

The second example hypothesis is `4x-3`, which matches the numeric value of `x²-3` **only at the chosen input `x=4`**. Affine candidates do not implement a general quadratic function.

## Invariants and constraints

1. **Non-mutation**: hypothetical consideration never changes the authoritative field, its revision, or any external system.
2. **Branch isolation**: recursive evaluation operates on a prior candidate's snapshot, not on the real field.
3. **Authorization**: a closed gate, absent hash, stale revision, or mismatched candidate digest cannot commit.
4. **Bounded evaluation**: validate 3D coordinates, scalar finiteness, value magnitudes, active-cell count and candidate count. No dense `N³` allocation.
5. **Deterministic integrity**: candidate digest is over canonical sorted numeric state. It is not a signed or tamper-proof external ledger.
6. **Scope**: commitment only mutates the `SuspensoryAlgebra` instance's in-memory field; integration into VM state, persistent storage, production actuators, or policies requires additional explicitly designed authority and verification gates.

## Relation to DM-vOmegaXi+ and CTR

`DMvOmegaXiConsiderationLoop` currently updates its own internal state when `step()` is called. To preserve non-commitment, any simulation of that loop must run on a **separate instance loaded from a copy** of the authorized snapshot, or under an equivalent isolation contract. This new SCA module supplies a separate, pure arithmetic decision layer and **does not automatically intercept or override** existing execution paths.

Future extensions can replace the affine map with a pure, bounded autoencoder/decoder or toroidal transform provided the same non-mutation, deterministic verification, sparse resource, and explicit authority properties are maintained. Evidence evaluation belongs in CTR and/or the caller's verifier; numerical convergence alone is insufficient to justify a real-world action.

## Test command

```bash
python -m pytest tests/test_suspensory_algebra.py -q -o addopts=''
```
