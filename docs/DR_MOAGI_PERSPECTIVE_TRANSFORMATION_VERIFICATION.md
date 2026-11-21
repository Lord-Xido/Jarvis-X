# DM-PTVP — Dr Moagi Perspective Transformation & Verification Principle

**Status:** Canonical proposed epistemic contract; bounded C++17 reference conformance v0.1.
**Owner / architect:** Dr Matladi Maxwell Moagi.
**Repository:** Jarvis-X. **Scope:** geometric reference, evidence admission, 3D observer models.

## Foundational principle

A change in representation can alter a model's interpretation even when the underlying information is recoverable exactly. Such interpretive variation is **not evidence** that the underlying physical object has changed. Conversely, newly acquired physical viewpoints can provide information absent from the original observation.

Distinguish:

1. **Physical state** `S`: the object/process, which need not be fully observable.
2. **Observation** `Y_v = P_v(S) + noise`: a measured signal under a stated sensor and viewpoint `v`.
3. **Reversible representation transformation** `T(Y_v)`: rearrangement of the *same acquired evidence*; ideal 90° image rotation is a pixel permutation.
4. **Physical viewpoint change** `Y_w = P_w(S) + noise`: a *new measurement* that may reveal previously occluded information.
5. **Model inference** `H = f_theta(Y)`: an interpretation, not an observation or proof.
6. **Verification** `V(H, evidence, assumptions)`: a reproducible test within an explicit claim boundary.

**Critical separation:** `T(Y_v) != Y_w` in general, even if both use words such as "rotation". In particular, no number of ideal 2D pixel rotations can recover depth absent from an image.

## Mathematical contract

Let `R` be a proper 3D rotation (`R^T R = I`, `det R = +1`). For a scalar invariant `c` and a geometric vector prediction `g`, the correct laws are different:

```
c(R S) = c(S)                 # invariance
g(R S) = R g(S)               # equivariance
T^{-1}(T(Y)) = Y             # reversible observation representation
```

In the discrete reference, `R` ranges over the **24** orientation-preserving signed-permutation matrices of the cube; this finite subgroup of `SO(3)` is **not all of `SO(3)`**. Every 3D rotation is an exact byte-index permutation of an `n x n x n` lattice, for `1 <= n <= 128`.

For a fixed, calibrated orthographic occupancy projection `P_xy`, non-injectivity is explicit:

```
S_a != S_b  while  P_xy(S_a) == P_xy(S_b).
```

Therefore equal projections alone imply an **UNDERDETERMINED**, not a VERIFIED, full-volume identity claim. Different projections from the same calibrated operator refute equality of the source volume. Exact full-volume byte comparison can settle byte identity **only** for the bounded supplied cubes; it cannot establish physical or semantic identity.

## Evidence verdict protocol

| Outcome | Necessary evidence | What it does **not** establish |
| --- | --- | --- |
| `VERIFIED` | Complete bounded byte states agree under exact comparison, or a declared finite algebraic invariant passes its protocol | Truth of a semantic interpretation, 3D reconstruction of unseen reality, or complete knowledge |
| `REFUTED` | Exact states disagree, or two calibrated projections disagree under an asserted equality hypothesis | Causal explanation for the difference |
| `UNDERDETERMINED` | Projections agree but their measurement operator is not injective | Object identity or absence of hidden structure |

A hypothesis admission gate should record the claim, scope, measurement protocol, transformations, tolerances, observed results, falsifying cases, and untested regions. Never promote a provisional hypothesis merely because an ANN or an observer returns high confidence. A new perspective supplies additional evidence only when a new physical measurement is performed; rotating existing pixels provides *no* new physical sample.

## Bounded executable reference

Files:

- `cpp_runtime/perspective3d/perspective3d.hpp` — dependency-free C++17 exact 2D/3D rotations, 3D occupancy projection, centroid and histogram checks, three-valued verdicts.
- `cpp_runtime/perspective3d/perspective3d_tests.cpp` — deterministic algebra, round-trip, equivariance, ambiguity, and corruption conformance.
- `cpp_runtime/perspective3d/CMakeLists.txt` — standalone build/test.
- `.github/workflows/dm-perspective-verification.yml` — GCC and Clang matrix and machine-readable report artifact.

Local independent build:

```sh
cmake -S cpp_runtime/perspective3d -B build/perspective3d -DCMAKE_BUILD_TYPE=Release
cmake --build build/perspective3d --parallel
ctest --test-dir build/perspective3d --output-on-failure
./build/perspective3d/jarvisx-perspective3d-tests
```

Optional inclusion in root `cpp_runtime` build:

```sh
cmake -S cpp_runtime -B build/cpp -DJARVISX_BUILD_PERSPECTIVE3D=ON
cmake --build build/cpp --target jarvisx-perspective3d-tests
ctest --test-dir build/cpp -R perspective3d-conformance --output-on-failure
```

Conformance output is one JSON object with schema `dm-ptvp-conformance.v0.1`, pass/fail, proper-rotation count, check count, and explicit inference boundary. Failure exits nonzero. Deterministic checks include: all 24 proper cube rotations, composition closure, inverse round trip, byte-histogram invariance, weighted-centroid equivariance, exact 90-degree 2D rotation (rectangular image), finite input rejection, a constructed two-volume/one-projection counterexample, distinguishing alternative views, and byte corruption detection.

## Non-claims and outstanding tests

This component **does not** reconstruct real 3D geometry from an arbitrary image; infer bird/ship/object identity; train an ANN; estimate uncertainty calibrated to a real dataset; model occlusion, light transport, lens distortion or noise; prove causal mechanisms; certify full physical understanding; or establish formal system-wide conformance. Those require separate sensor models, calibration, curated ground truth, empirical accuracy studies, specified safety gates, and independent replication.

A failed or untested condition must remain explicit. Conformance to exact finite transformations is not a proof of complete reality knowledge.

## Integration with existing Jarvis-X specifications

- `docs/REALITY_GROUNDED_OBSERVER_DYNAMICS.md`: physical reality, observation and inference occupy distinct epistemic levels; DM-PTVP adds perspective-specific test requirements.
- `docs/CANONICAL_3D_GEOMETRIC_INTELLIGENCE.md`: distinguishes invariant scalars from equivariant geometric quantities and adds an explicit ambiguity verdict.
- `docs/EMPIRICAL_VALIDATION.md`: DM-PTVP uses a **separate** native CI conformance gate, not the Python `jarvisx.empirical_validation` gate.
- `cpp_runtime/permeate3d`: existing 3D tile codec and transactional semantics are unchanged. Perspective verification is an opt-in companion, not an assertion that the codec has been fully audited.

**Canonical governing constraint:** Assert only what the declared evidence, domain, and test protocol demonstrably establish. Identify uncertainty and missing information rather than substituting plausible interpretations.
