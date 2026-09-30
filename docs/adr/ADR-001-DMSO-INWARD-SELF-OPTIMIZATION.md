# ADR-001: DMSO inward self-optimization

**Status:** Proposed  
**Date:** 2026-08-04  
**Owner:** Dr. Matladi Moagi  
**Repository:** Jarvis-X  
**Architectural layer:** Layer 5 — Adaptive research systems

## Context

The Dr. Moagi System of Operations (DMSO) proposes an inward-turned adaptive hierarchy in which operational parameters are represented as part of the system state and are therefore eligible for the same encode, evaluate, correct and constrain cycle as ordinary runtime state.

The core idea is self-similarity across a finite hierarchy:

- level `0` evolves the primary operational state, such as token positions, geometric state or VM-adjacent research state;
- level `1` evolves selected parameters of level `0`;
- level `2` evolves selected parameters of level `1`;
- the hierarchy terminates at an explicit maximum depth `L`;
- every proposed update is projected into an admissible set, evaluated in shadow state and either committed or rolled back.

This proposal must remain consistent with the canonical Jarvis-X architecture:

1. the authoritative VM remains deterministic and understandable without DMSO;
2. adaptation is opt-in and bounded;
3. no adaptive layer may silently mutate canonical VM state;
4. every accepted transition must be reproducible, constrained and journaled;
5. mathematical self-reference is represented as finite state recursion, not unrestricted native-code rewriting.

## Decision

DMSO is accepted as a **proposed research architecture**, not as an implemented or production-capable subsystem.

The mathematically valid form uses additive forces, typed cross-level projections, finite recursion, annealed stochasticity and transactional state transitions.

## 1. Finite recursive hierarchy

Let the hierarchy contain levels

\[
\ell \in \{0,1,\ldots,L\}.
\]

Define the level state blocks

\[
\mathbf{s}^{(0)} = \mathbf{P},
\qquad
\mathbf{s}^{(\ell)} = \Theta^{(\ell-1)} \quad \text{for } \ell \ge 1.
\]

The augmented state is

\[
\mathbf{S}
=
\operatorname{concat}
\left(
\mathbf{s}^{(0)},
\mathbf{s}^{(1)},
\ldots,
\mathbf{s}^{(L)}
\right).
\]

Each block has an explicit schema, dimensionality, unit convention, admissible range and mutability mask. Infinite materialization is prohibited. A higher level may be instantiated lazily, but the active runtime depth is always finite and bounded.

## 2. Typed cross-level representation

Raw token coordinates, optimizer coefficients, matrices and scalar control parameters do not inhabit the same physical or numerical space. They therefore cannot be directly compared with one Euclidean distance.

Each state block is mapped into a common interaction space through a type-specific projection:

\[
\mathbf{h}_i = \phi_{\tau(i)}(\mathbf{s}_i) \in \mathbb{R}^{d_h},
\]

where `τ(i)` identifies the block type and `φ` is a validated encoder for that type.

Cross-level attention is then defined by

\[
q_i = W_Q^{\tau(i)} h_i,
\qquad
k_j = W_K^{\tau(j)} h_j,
\qquad
v_j = W_V^{\tau(j)} h_j,
\]

\[
A_{ij}
=
\operatorname{softmax}_j
\left(
\frac{q_i^\top k_j}{\sqrt{d_k}}
+ b_{\tau(i),\tau(j)}
+ m_{ij}
\right),
\]

where `b` is a learned or fixed type-pair bias and `m` is an admissibility mask. Forbidden couplings receive `-∞` before the softmax.

This preserves the intended interactions:

- state-to-state coordination;
- state-to-parameter sensitivity;
- parameter-to-state adaptation;
- parameter-to-parameter coordination;

without pretending heterogeneous quantities are directly commensurate.

## 3. Corrected potential

The original inward formulation multiplied the attention, reconstruction and regularization objectives. Multiplication is rejected because any near-zero factor can suppress every other objective and because the resulting units are generally inconsistent.

DMSO uses an additive potential:

\[
U(\mathbf{S})
=
\alpha U_{\mathrm{att}}(\mathbf{S})
+
\beta U_{\mathrm{rec}}(\mathbf{S})
+
\lambda R_{\mathrm{meta}}(\mathbf{S})
+
\mu L_{\mathrm{task}}(\mathbf{S}).
\]

### 3.1 Attention/coherence energy

A stable graph-coherence energy is preferred over an inverse-distance singularity:

\[
U_{\mathrm{att}}
=
\frac{1}{2}
\sum_{i,j}
A_{ij}
\left\|
\psi_{\tau(i)}(s_i)
-
\psi_{\tau(j)}(s_j)
\right\|_2^2.
\]

This makes the attraction term differentiable and prevents the unbounded force produced by `1 / ||s_i-s_j||` near coincidence.

### 3.2 Reconstruction energy

Let

\[
r(\mathbf{S})
=
\mathcal{D}(\mathcal{E}(\mathbf{S}))-\mathbf{S}.
\]

Then

\[
U_{\mathrm{rec}}
=
\frac{1}{2}\|r(\mathbf{S})\|_F^2.
\]

The corresponding force is the true gradient

\[
-\nabla_{\mathbf{S}}U_{\mathrm{rec}}
=
-J_r(\mathbf{S})^\top r(\mathbf{S}),
\]

not merely the raw reconstruction residual.

### 3.3 Meta-regularization

Meta-regularization constrains drift relative to a reference or trust region:

\[
R_{\mathrm{meta}}
=
\frac{1}{2}
\sum_{\ell=0}^{L}
\left\|
C_{\ell}^{1/2}
\left(
\Theta^{(\ell)}-\Theta_{\mathrm{ref}}^{(\ell)}
\right)
\right\|_2^2.
\]

The matrices `C_ℓ` encode parameter sensitivity, scaling and any frozen coordinates.

## 4. Continuous inward dynamics

The corrected continuous-time model is

\[
M\ddot{\mathbf{S}}
=
-P_A\nabla_{\mathbf{S}}U(\mathbf{S})
-\Gamma\dot{\mathbf{S}}
+\Sigma(t)\boldsymbol{\xi}(t),
\]

where:

- `M` is a positive-definite block mass or preconditioning matrix;
- `P_A` projects gradients onto active, mutable coordinates;
- `Γ` is positive semidefinite damping;
- `Σ(t)` is the stochastic amplitude schedule;
- `ξ(t)` is a seeded stochastic process when stochastic exploration is enabled.

The force terms are additive. Attention, reconstruction, task error, damping and stochastic exploration must never be multiplied together.

## 5. Discrete transactional runtime

The implementable form is a bounded discrete transition.

### 5.1 Proposal

\[
\mathbf{v}_{k+1}
=
\mathbf{m}_k \odot
\left[
\rho\mathbf{v}_k
-
\eta_k P_A\nabla U(\mathbf{S}_k)
+
\sigma_k\boldsymbol{\xi}_k
\right],
\]

\[
\widetilde{\mathbf{S}}_{k+1}
=
\Pi_{\Lambda_k}
\left(
\mathbf{S}_k+\mathbf{v}_{k+1}
\right).
\]

Here `m_k` is the blockwise mutability mask and `Π_Λ` projects the candidate into dimensional, numerical, policy and coherence constraints.

### 5.2 Verification and commit

The candidate is evaluated in shadow state:

\[
V_k
=
\operatorname{Verify}
\left(
\mathbf{S}_k,
\widetilde{\mathbf{S}}_{k+1}
\right).
\]

The authoritative transition is

\[
\mathbf{S}_{k+1}
=
\begin{cases}
\widetilde{\mathbf{S}}_{k+1}, & V_k = \text{accept},\\
\mathbf{S}_k, & V_k = \text{reject}.
\end{cases}
\]

Every decision records:

- prior state digest;
- candidate state digest;
- active recursion depth;
- random seed and stochastic schedule;
- objective components;
- constraint results;
- acceptance or rollback reason;
- resulting state digest.

## 6. Locking semantics

A lock is blockwise, explicit and journaled.

For level `ℓ`, define

\[
m^{(\ell)} = 0
\]

when that level is frozen. Locking all levels `ℓ ≥ L_lock` does not imply that the complete augmented state is stationary if lower levels remain active.

An exact deterministic fixed point requires:

\[
P_A\nabla U(\mathbf{S}^*)=0,
\qquad
\mathbf{v}^*=0,
\qquad
\sigma_k \rightarrow 0.
\]

If stochastic amplitude remains nonzero, the appropriate object is a stationary distribution, not an exact fixed point.

Convergence is declared only after all of the following hold for `K` consecutive accepted iterations:

\[
\|P_A\nabla U(\mathbf{S}_k)\|_2 \le \delta_g,
\]

\[
\|\mathbf{v}_k\|_2 \le \delta_v,
\]

\[
|U(\mathbf{S}_k)-U(\mathbf{S}_{k-1})| \le \delta_U,
\]

with no constraint, replay or journal-integrity violation.

## 7. Valid self-reference

The statement

\[
\mathcal{M}_{\mathrm{DMSO}}=\mathcal{M}_{\mathrm{DMSO}}
\]

is a tautology and does not formalize an operator acting on itself.

DMSO represents self-reference in two precise ways.

### 7.1 Parameters inside state

\[
\mathcal{M}(\mathbf{S};\Theta),
\qquad
\Theta \subset \mathbf{S}.
\]

The operator changes because selected parameters contained in `S` change under the same constrained state-transition law.

### 7.2 Operator fixed point

Let `T` map one bounded operator configuration to another:

\[
\mathcal{M}_{k+1}=\mathcal{T}(\mathcal{M}_k).
\]

A self-consistent operator satisfies

\[
\mathcal{M}^*=\mathcal{T}(\mathcal{M}^*).
\]

`T` may alter only declared configuration fields, model parameters, schedules or bytecode tables. It may not rewrite arbitrary native code, bypass policy, remove rollback or expand its own authority.

## 8. Operational cycle

The DMSO research loop is:

```text
Observe augmented state
  → Validate schemas and active depth
  → Encode typed state blocks
  → Compute masked cross-level attention
  → Evaluate objective components
  → Propose bounded update
  → Project through Λ constraints
  → Execute candidate in shadow state
  → Verify invariants and regression metrics
  → Commit or rollback
  → Journal provenance
  → Evaluate convergence and lock conditions
  → Continue or halt
```

This cycle refines selected internal parameters by executing one explicit transition contract. It does not imply unlimited autonomy or unrestricted self-modification.

## 9. Required state contracts

A future implementation must define at minimum:

```text
AugmentedState
  schema_version
  recursion_depth
  state_blocks[]
  parameter_blocks[]
  velocity_blocks[]
  mutability_masks[]
  constraint_profile
  encoder_version
  decoder_version
  random_seed
  iteration

ObjectiveReport
  attention_energy
  reconstruction_energy
  meta_regularization
  task_loss
  total_energy
  gradient_norm

TransitionRecord
  prior_digest
  candidate_digest
  projected_digest
  accepted
  rejection_reason
  invariant_results
  resulting_digest
```

All persistent encodings require versioning, deterministic serialization and malformed-input rejection.

## 10. Safety and capability boundary

DMSO must enforce:

1. **Finite active depth.** `L` and the number of active blocks are hard bounded.
2. **Typed coupling.** Cross-level interactions pass through validated projections and masks.
3. **Bounded parameter deltas.** Per-block trust regions and norm caps are mandatory.
4. **Shadow evaluation.** Candidate changes cannot directly alter authoritative state.
5. **Atomic commit or rollback.** Partial state mutation is prohibited.
6. **Deterministic replay.** Seeds, schedules and serialization are recorded.
7. **External authority boundary.** DMSO cannot modify policy, permissions or security controls from within its ordinary adaptive state.
8. **No arbitrary native-code rewriting.** Adaptation is limited to declared data, configuration and validated bytecode representations.
9. **Resource accounting.** Every recursion level reports resident memory, compute budget and iteration limit.
10. **Honest status.** A mathematical specification is not evidence that the system is implemented, convergent, safe or superior to existing methods.

## 11. Validation plan

A reference implementation is not eligible for canonical integration until it includes tests for:

- dimensional and type compatibility of every cross-level projection;
- finite-difference agreement with analytic or automatic gradients;
- deterministic replay under a fixed seed;
- immutability of locked blocks;
- projection into every declared constraint set;
- rollback after rejected candidates;
- bounded parameter deltas and bounded recursion depth;
- convergence on convex fixtures;
- non-convergence reporting on adversarial or non-convex fixtures;
- annealing of stochastic amplitude before deterministic lock;
- journal-chain integrity;
- preservation of canonical VM semantics when DMSO is disabled;
- absence of authoritative VM mutation before explicit commit.

## 12. Staged implementation

### Phase 0 — Specification

- finalize schemas, equations, constraints and invariants;
- define deterministic fixtures;
- keep status `proposed`.

### Phase 1 — Pure Python reference

- implement a small finite hierarchy;
- use explicit typed blocks;
- provide deterministic optimization fixtures;
- expose no canonical VM mutation path.

### Phase 2 — Transaction and replay

- add shadow-state evaluation;
- add atomic commit/rollback;
- add hash-chained transition journals;
- test checkpoint and replay.

### Phase 3 — C++ processor laboratory integration

- integrate only bounded parameter and schedule search;
- measure memory, runtime and convergence behavior;
- preserve the existing processor-laboratory capability boundary.

### Phase 4 — Optional VM-adjacent adapter

- keep DMSO disabled by default;
- require a narrow API and explicit policy authorization;
- prevent silent modification of ordinary assembly semantics.

## Consequences

### Positive

- The inward-turn concept becomes mathematically coherent and implementable.
- Recursive adaptation is finite, typed and auditable.
- The model aligns with Jarvis-X determinism, transaction and provenance requirements.
- Exact fixed points are distinguished from stochastic stationary behavior.
- The specification creates a testable route from mathematical proposal to bounded software.

### Costs and risks

- Full-state autoencoding may be expensive and may obscure local failure modes.
- Cross-level coupling can destabilize optimization unless projections, masks and trust regions are carefully designed.
- Jointly evolving state and optimizer parameters creates a non-stationary objective.
- Fixed-point convergence is not guaranteed for non-convex objectives.
- A compressed latent representation can discard operationally significant state unless reconstruction and invariant tests are strong.

## Non-goals

This ADR does not claim:

- consciousness or sentience;
- infinite materialized recursion;
- guaranteed global convergence;
- unrestricted autonomous self-modification;
- arbitrary code rewriting;
- lossless compression of arbitrary state into a smaller latent code;
- production safety or superiority without reproducible evidence.

## Canonical DMSO research form

The proposed continuous form is

\[
\boxed{
M\ddot{\mathbf{S}}
=
-P_A\nabla_{\mathbf{S}}
\left[
\alpha U_{\mathrm{att}}
+
\beta U_{\mathrm{rec}}
+
\lambda R_{\mathrm{meta}}
+
\mu L_{\mathrm{task}}
\right]
-
\Gamma\dot{\mathbf{S}}
+
\Sigma(t)\boldsymbol{\xi}(t)
}
\]

subject to finite recursion, typed projections, blockwise locking, `Λ`-projection, shadow verification and atomic commit or rollback.

The proposed operational form is

\[
\boxed{
\begin{aligned}
\mathbf{v}_{k+1}
&=
\mathbf{m}_k\odot
\left[
\rho\mathbf{v}_k
-
\eta_kP_A\nabla U(\mathbf{S}_k)
+
\sigma_k\boldsymbol{\xi}_k
\right],\\
\widetilde{\mathbf{S}}_{k+1}
&=
\Pi_{\Lambda_k}(\mathbf{S}_k+\mathbf{v}_{k+1}),\\
\mathbf{S}_{k+1}
&=
\operatorname{CommitOrRollback}
(\mathbf{S}_k,\widetilde{\mathbf{S}}_{k+1}).
\end{aligned}
}
\]

This is the bounded inward turn: the system state contains selected parameters of its own transition law, while external constraints, verification and authority boundaries remain explicit.
