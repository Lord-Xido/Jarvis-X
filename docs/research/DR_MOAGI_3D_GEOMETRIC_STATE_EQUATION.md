# Dr Moagi 3D Geometric State Equation

**Status:** Research specification aligned to ADR-0010  
**Repository:** `Lord-Xido/Jarvis-X`  
**Date:** 2026-08-17

## 1. Canonical recurrence

The geometric state law is

\[
\boxed{
\Xi_{t+1}^{3D}
=
\Pi_{\Lambda_t}\left[
\Xi_t^{3D}
+P_{1:M}^{\circlearrowleft}(\Xi_t^{3D})
-E_t^{3D}
+\Omega_t^{3D}
+\kappa_tR_t^{\circlearrowleft}
-\eta_t\nabla_\Theta L_t
-\zeta_t\nabla_HC_t
\right]
}
\]

It is interpreted as a discrete-time nonlinear state-space recurrence with predictive branching, correction, persistent memory, inward refinement, optimization gradients and admissibility projection.

It is a Jarvis-X research architecture, not an established physical law and not a claim of autonomous intelligence.

## 2. Geometric state space

For a sparse scalar field, let

\[
\mathcal V_t\subset\mathbb Z^3
\]

be the active support and

\[
\Xi_t:\mathcal V_t\rightarrow\mathbb R.
\]

The three geometric axes may be interpreted operationally as:

```text
X -> operations / workflow
Y -> state / time
Z -> control / abstraction depth
```

This interpretation is descriptive. An implementation remains authoritative only through its declared data structures and transition contract.

For vector- or tensor-valued voxels, replace the scalar codomain with an explicitly typed compatible vector space. Terms may be added only when their shapes, units and support are compatible.

## 3. Term semantics

### 3.1 Current state

\[
\Xi_t^{3D}
\]

is the frozen authoritative research-layer snapshot used to evaluate one candidate transition.

### 3.2 Anticipatory branching

For `M` predictor branches:

\[
P_{1:M}(\Xi_t)=\{P_1(\Xi_t),\ldots,P_M(\Xi_t)\}.
\]

A bounded deterministic merge produces one additive predictive field:

\[
\bar P_t=\sum_{m=1}^{M}a_mP_m(\Xi_t),
\qquad a_m\ge0,\quad\sum_ma_m=1.
\]

The inward marker `circlearrowleft` denotes that predictors may consume recurrent latent or field context. It does not authorize unbounded recursion.

### 3.3 Error field

\[
E_t^{3D}
\]

is a same-space discrepancy field. Examples include reconstruction residual, task residual, model residual or a deliberately transformed correction field. If the residual is defined in another space, it must be mapped into the authoritative state space before addition.

### 3.4 Persistent memory

\[
\Omega_t^{3D}
\]

is a bounded persistent field carrying validated historical influence. An implementation should distinguish authoritative memory from append-only logs or provenance.

### 3.5 Inward refinement

\[
R_t^{\circlearrowleft}
\]

is a bounded recurrent refinement field. It may be derived from latent recursion, a contractive spatial transform, geometric diffusion, residual closure or another declared operator. `kappa_t` controls its contribution.

### 3.6 Learning gradient

\[
-\eta_t\nabla_\Theta L_t
\]

represents a same-space contribution induced by optimization of objective `L`. A parameter-space gradient cannot be added directly to a voxel field unless a declared Jacobian, decoder or update map transforms it into the authoritative state space.

### 3.7 Constraint gradient

\[
-\zeta_t\nabla_HC_t
\]

represents a same-space correction induced by constraint functional `C`. It supplements but does not replace `Pi_Lambda`.

### 3.8 Projection

\[
\Pi_{\Lambda_t}
\]

maps or rejects a proposed state against the admissible set. Depending on the runtime, `Lambda_t` may encode:

- numeric bounds;
- shape/support constraints;
- resource ceilings;
- topology or geometry rules;
- codec/version coherence;
- drift and distortion limits;
- policy constraints;
- iteration and latency budgets.

Projection is the authority boundary between a proposed candidate and a committable Layer-5 state.

## 4. Candidate formation

Define the unprojected displacement

\[
\Delta\Xi_t
=
\bar P_t-E_t+\Omega_t+\kappa_tR_t
-\eta_t\nabla_\Theta L_t
-\zeta_t\nabla_HC_t.
\]

Then

\[
\Xi_t^*=\Xi_t+\Delta\Xi_t
\]

and

\[
\Xi_{t+1}=\Pi_{\Lambda_t}(\Xi_t^*).
\]

Operationally:

```text
snapshot
  -> branch
  -> merge
  -> correct
  -> remember
  -> refine inward
  -> apply learning field
  -> apply constraint field
  -> project
  -> validate
  -> commit / rollback
```

## 5. Geometric interpretation

The equation can be visualized as the superposition of vector fields over a 3D manifold:

```text
                     predictive fan P_1:M
                           \ | /
                            \|/
                memory ---> Xi_t <--- error correction
                             |
                             v
                       inward torus R
                             |
                loss gradient + constraint gradient
                             |
                             v
                         Pi_Lambda
                             |
                             v
                         Xi_(t+1)
                             |
                             +---- recurrent re-entry ----+
```

The geometry is therefore a constrained recurrent flow, but the executable implementation remains discrete and candidate-first.

## 6. Relationship to the codec-runtime

The codec runtime may supply the terms as follows:

```text
Xi_t               <- working field or latent field
E_t                <- local / anchor / task residual mapped to Xi-space
Omega_t            <- persistent validated adaptive memory
R_t                <- inward latent refinement or spatial re-entry field
nabla_Theta L_t    <- optimizer-induced state correction
nabla_H C_t        <- constraint-induced state correction
Pi_Lambda          <- codec/runtime admissibility gate
```

The entropy codec invariant remains separate:

```text
decode(encode(Z_t)) == Z_t
```

The geometric state equation does not redefine entropy coding or the deterministic bitstream contract.

## 7. Relationship to the sparse field runtime

`src/jarvisx/dr_moagi_field_runtime.py` already implements a bounded sparse candidate-first field equation. The geometric state equation is a higher-level algebraic interface that can consume terms generated by that runtime or run beside it.

No existing field term is automatically equivalent to a geometric-state term merely because names are similar. A binding must state the mapping explicitly.

## 8. Reference implementation

The executable reference is:

```text
src/jarvisx/dr_moagi_state_equation.py
```

It provides:

```text
DrMoagiEquationConfig
DrMoagiEquationTerms
DrMoagiEquationStep
DrMoagiStateEquation
merge_predictive_branches
box_projector
```

The implementation deliberately uses standard Python mappings rather than dense arrays so that the contract remains clear and bounded.

## 9. Reference lowering

A future bytecode or scheduler lowering may use a sequence such as:

```text
SNAPSHOT_XI
PREDICT_BRANCH x M
MERGE_PREDICTIONS
LOAD_ERROR_FIELD
LOAD_MEMORY_FIELD
REFINE_INWARD
LOAD_LOSS_GRADIENT
LOAD_CONSTRAINT_GRADIENT
FUSED_STATE_UPDATE
PROJECT_LAMBDA
VALIDATE_CANDIDATE
COMMIT_OR_ROLLBACK
EMIT_TELEMETRY
```

This is a research lowering contract. It does not add opcodes to the canonical VM unless a later ADR defines exact encoding and instruction semantics.

## 10. Conformance invariants

A conforming implementation must satisfy:

1. all additive terms share a compatible state space;
2. one logical step uses a frozen `Xi_t` snapshot;
3. predictive aggregation has an explicit scaling law;
4. non-finite arithmetic fails closed;
5. resource/materialization ceilings are explicit;
6. `Pi_Lambda` runs before authority is granted;
7. rejection preserves the previous authoritative state;
8. visualization is observational unless bound to the same transition;
9. virtual depth and branch count are reported separately from measured throughput;
10. parameter-space gradients are transformed into state space before addition.

## 11. Stability

Let the complete projected operator be

\[
G(\Xi)=\Pi_\Lambda[\Xi+F(\Xi)].
\]

For a differentiable local model around equilibrium `Xi*`, stability analysis may use

\[
\delta\Xi_{t+1}=J_G(\Xi^*)\delta\Xi_t.
\]

Local asymptotic stability requires the spectral radius to remain below one in the analyzed regime:

\[
\rho(J_G)<1.
\]

Projection, clipping or non-smooth branch selection can invalidate a naive linear analysis, so empirical boundedness and rollback tests remain mandatory.

## 12. Telemetry

Recommended telemetry per step:

```text
cycle
active_cells
branch_count
prediction_weight_entropy
raw_candidate_norm
projected_candidate_norm
projection_delta_norm
error_norm
memory_norm
refinement_norm
loss_gradient_norm
constraint_gradient_norm
committed
rejection_reason
wall_clock_time
```

Any claim about acceleration must additionally expose measured work and wall-clock time rather than infer throughput from virtual recurrence depth.

## 13. Current validation surface

The reference tests cover:

- exact recurrence arithmetic;
- convex predictive merging;
- branch-count scaling behavior;
- same-space support enforcement;
- non-finite rejection;
- projection support integrity;
- candidate validation and rollback.

Future extensions should add vector/tensor-valued field tests, property-based tests, bounded stochastic predictors with seeded replay, and integration tests against the codec and field runtimes.
