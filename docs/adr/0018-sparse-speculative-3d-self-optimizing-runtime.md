# ADR-018: Sparse speculative 3D self-optimizing runtime

**Status:** Accepted  
**Date:** 2026-09-13  
**Applies to:** Dr Moagi operational autoencoding equation, multimodal 3D runtimes, volumetric ROM ANN, browser visualization/control planes, runtime/meta-optimizers and future CPU/GPU/distributed/hardware backends  
**Extends:** ADR-016 and ADR-017

## Context

ADR-016 established structural closure: one typed state, one candidate-first transaction model, one verification contract and multiple interchangeable geometry/backend profiles. ADR-017 then established the canonical encode -> compact -> fixed-point -> decode -> evidence -> staged-adaptation law.

The next systems problem is computational selectivity. Existing demonstrations and bounded reference runtimes still spend work on regions, experts or iterations that may not materially affect the accepted output. The intended research direction is therefore to reduce unnecessary computation while preserving or improving output quality, stability and transaction guarantees.

This ADR formalizes that direction as a sparse speculative 3D runtime. It also defines the meaning of the phrase "beyond SOTA" within Jarvis-X: it is a benchmark target, not an implementation or performance claim.

## Decision

Jarvis-X adopts the following research execution principle:

```text
encode
 -> activate only informative 3D support
 -> route to a bounded expert subset
 -> refine inward only while unresolved
 -> generate bounded speculative branches
 -> verify/select one candidate
 -> decode / contrast
 -> stage model and runtime adaptation
 -> verify / Pi_Lambda
 -> atomic commit OR rollback
```

The canonical candidate operator is

\[
S^{\rm cand}_{t+1}
=
\left[
\mathcal O_{\Pi}
\circ
\mathcal O_{\Theta}
\circ
\mathcal D
\circ
\mathcal B_{\rm spec}
\circ
\operatorname{Fix}_{\Phi}
\circ
\mathcal R_{\rm expert}
\circ
\mathcal A_{\rm sparse}
\circ
\mathcal E
\right](S_t,U_{t+1}),
\]

followed by the unchanged ADR-016 authority law

\[
S_{t+1}
=
V_t\,\Pi_\Lambda(S^{\rm cand}_{t+1})+(1-V_t)S_t,
\]

interpreted structurally. No sparse, speculative or adaptive mechanism may bypass verification or mutate authoritative state before commit.

## 1. Sparse active support

Let the declared logical geometry be \(\mathbb V\) and the physically materialized active set be

\[
\mathcal A_t\subseteq\mathbb V.
\]

Each active element receives an information/relevance score

\[
a_i
=
\lambda_p p_i
+\lambda_e |E_i|
+\lambda_\Omega |\Omega_i|
+\lambda_u u_i,
\]

where terms may encode transition probability, residual magnitude, memory significance and uncertainty or task utility.

The execution set is

\[
\mathcal A_t
=
\{i:a_i>\tau_a\}
\]

or an equivalent bounded top-k rule.

A backend SHALL report both logical extent and materialized support. Sparse execution may reduce work; it does not change the declared logical geometry.

## 2. Conditional expert routing

Let \(F_1,\ldots,F_K\) be bounded expert transforms and let

\[
g(z)=\operatorname{softmax}(W_gz+b_g).
\]

Only a bounded subset

\[
\mathcal K_t=\operatorname{TopK}(g(z_t),k_e),
\qquad k_e\ll K,
\]

is executed. The expert transition may be

\[
z'_t
=
z_t+
\sum_{j\in\mathcal K_t}g_j(z_t)F_j(z_t).
\]

The runtime SHALL account separately for router cost, expert execution cost, expert weight residency/transfer cost and any reuse/cache benefit. A nominally sparse expert design is not considered accelerated unless measured wall-clock/resource telemetry demonstrates the effect.

## 3. Adaptive inward depth

Latent refinement follows

\[
Z_t^{(k+1)}
=
\Phi_{\Theta_t,\Pi_t,\mathcal K_t}
\left(Z_t^{(k)},\Omega_t\right).
\]

A region or latent state may halt when

\[
\frac{\|Z_t^{(k+1)}-Z_t^{(k)}\|}
{\|Z_t^{(k)}\|+\epsilon}
<\tau_Z
\]

or when its bounded iteration budget is exhausted.

Easy/stable regions SHOULD consume fewer refinement steps than unresolved regions. Reported speedup must include the cost of convergence testing and any divergent/rejected candidate work.

## 4. Speculative branch execution

The runtime MAY produce a bounded branch set

\[
\mathcal B_t=\{b_1,\ldots,b_B\}
\]

from the converged or provisional latent state. Branches remain provisional and isolated.

Each branch receives a verification/utility record such as

\[
Q_b
=
q_{\rm quality}(b)
-\lambda_c C_b
-\lambda_e E_b
-\lambda_r R_b,
\]

where \(C_b\) is execution/resource cost, \(E_b\) is prediction/reconstruction discrepancy and \(R_b\) is instability or policy risk.

Only a branch that satisfies all hard gates may be selected:

\[
b_t^*
=
\operatorname*{arg\,max}_{b\in\mathcal B_{\rm valid}}Q_b.
\]

Rejected branches cannot mutate `Omega_mem`, `Theta_model`, `Pi_runtime`, authoritative spatial state or audit lineage except for explicit non-authoritative evaluation telemetry.

## 5. Runtime policy state

The bounded runtime policy MAY include

```text
Pi_runtime = (
    active_support_budget,
    expert_top_k,
    recursion_depth,
    speculative_width,
    numerical_precision,
    batch_or_tile_size,
    cache_or_memory_placement,
    temperature_or_sampling_policy
)
```

Each component has an explicit admissible domain. Optimization is policy search, not unrestricted source-code mutation.

## 6. Self-optimization law

For candidate runtime policies \(\Pi^{(j)}\), evaluate a fixed validation workload or immutable evaluation anchor and measure a vector

\[
M_j=
[Q_j,D_j,L_j,M_j^{\rm mem},B_j,S_j,C_j],
\]

where the terms represent output quality, divergence/error, latency, resident memory, bandwidth/data movement, stability and evaluation cost.

Hard constraints are evaluated before scalar or Pareto optimization. A reference admissibility predicate is

\[
\operatorname{valid}(j)
\iff
Q_j\ge Q_{\min}
\land D_j\le D_{\max}
\land S_j\le S_{\max}
\land \operatorname{resources\_ok}(j)
\land \operatorname{integrity\_ok}(j).
\]

Only valid candidates may compete under a runtime objective, for example

\[
J_{\rm run}(j)
=w_DD_j+w_LL_j+w_MM_j^{\rm mem}+w_BB_j+w_SS_j+w_CC_j.
\]

Then

\[
\Pi_{t+1}^{\rm cand}
=
\operatorname*{arg\,min}_{j:\operatorname{valid}(j)}J_{\rm run}(j).
\]

Promotion still occurs only inside the enclosing ADR-016 transaction.

The optimizer SHALL NOT modify the evaluation set, acceptance thresholds, benchmark baseline or hard verification gates while optimizing against them.

## 7. Evidence hierarchy

The runtime distinguishes four evidence classes:

1. **internal numerical coherence** — finite state, normalization, fixed-point residuals, reconstruction/cycle metrics;
2. **resource evidence** — measured latency, resident memory, data movement, active support and branch/expert counts;
3. **external correspondence** — task/world evidence independent of the system's own self-consistency when such claims are made;
4. **comparative benchmark evidence** — matched baseline results required for SOTA or beyond-SOTA claims.

Absence of an external evidence source is never encoded as `passed`; it is `not_applicable` or `not_implemented` under ADR-016 typed verification status.

## 8. Telemetry contract

Canonical runtime telemetry SHOULD expose at least

```text
geometry_profile
logical_elements
active_elements
active_fraction
experts_total
experts_active
expert_reuse_or_cache_hits
fixed_point_steps
fixed_point_residual
speculative_branches
accepted_branch_length_or_depth
rejected_branches
reconstruction_or_task_error
transition_or_step_latency
throughput
resident_memory_bytes
transfer_or_bandwidth_proxy
policy_candidate_id
verification_status
transaction_id
```

Synthetic visualization values MUST be labeled as simulated. A browser UI may visualize compute topology, but values such as PFLOPS, VRAM, hardware temperature, token/transition rate or accelerator count are authoritative only when supplied by a measured backend or instrumentation source.

## 9. Browser/Three.js binding

A browser visualization is a Layer-6 control plane unless separately promoted as an authoritative compute backend. Its nodes SHOULD represent actual logical/active states where possible:

```text
size/brightness  -> measured activity/probability
visible support  -> active set
cluster grouping -> selected experts/modalities
inward rings     -> contraction/refinement depth
branch paths     -> speculative candidates
red/collapse     -> rejected candidate
cyan/green path  -> accepted/committed transition
```

Random animation may remain for decorative effects but SHALL NOT be presented as measured ANN activity.

## 10. Beyond-SOTA research criterion

"Beyond SOTA" is an experimental target. Jarvis-X MAY claim superiority only after a matched benchmark demonstrates a declared dominance relation such as

\[
Q_{\rm DM}\ge Q_{\rm base}
\quad\land\quad
L_{\rm DM}<L_{\rm base}
\quad\land\quad
M_{\rm DM}<M_{\rm base},
\]

or another pre-registered quality/resource Pareto criterion, under the same dataset/task distribution, hardware class, precision policy, batch/sequence constraints and output-quality threshold.

A valid comparison SHALL include repetitions, uncertainty or variance reporting where relevant, complete baseline configuration, warmup methodology, measurement window and failure/rejection accounting.

Virtual geometry, nominal parallelism, simulated PFLOPS or reduced active-node count alone are not evidence of SOTA performance.

## 11. Canonical compact form

The sparse speculative 3D runtime is

\[
\boxed{
\begin{aligned}
Z_t^{(0)} &= \mathcal E_{\Theta_t}(X_t,U_t),\\
\mathcal A_t &= \operatorname{SelectActive}(Z_t,E_t,\Omega_t;\Pi_t),\\
\mathcal K_t &= \operatorname{TopExperts}(G_{\Theta_t}(Z_t);\Pi_t),\\
Z_t^* &= \operatorname{FixPoint}(\Phi_{\Theta_t,\Pi_t,\mathcal K_t}),\\
\mathcal B_t &= \operatorname{Speculate}(Z_t^*,\Pi_t),\\
b_t^* &= \operatorname{VerifySelect}(\mathcal B_t),\\
\hat X_t &= \mathcal D_{\Theta_t}(b_t^*),\\
E_t &= X_t-\hat X_t,\\
(\tilde\Omega,\tilde\Theta,\tilde\Pi) &= \operatorname{StageAdapt}(E_t,S_t),\\
S^{\rm cand}_{t+1} &= \operatorname{AssembleCandidate}(\hat X_t,\tilde\Omega,\tilde\Theta,\tilde\Pi),\\
S_{t+1} &= V_t\Pi_\Lambda(S^{\rm cand}_{t+1})+(1-V_t)S_t.
\end{aligned}
}
\]

This equation is subordinate to ADR-016 authority semantics and extends the ADR-017 operational autoencoding law with conditional computation, adaptive depth and speculative candidate evaluation.

## 12. Non-goals

This ADR does not establish:

- external SOTA or beyond-SOTA performance;
- unrestricted autonomous source modification;
- dense physical execution of large logical geometries;
- semantic multimodal understanding from byte-level adapters alone;
- production safety from the existence of a verification gate;
- hardware acceleration without measured backend results.

## Validation

A conforming implementation should demonstrate:

1. explicit logical and active support;
2. bounded expert routing with routing telemetry;
3. adaptive fixed-point depth and residual receipts;
4. isolated speculative branches and rollback-complete rejection;
5. immutable evaluation anchors during runtime-policy search;
6. measured rather than invented runtime telemetry;
7. typed verification receipts inherited from ADR-016;
8. atomic promotion of any accepted adaptive/runtime state;
9. deterministic replay or an explicitly documented stochastic protocol;
10. matched baselines before any comparative performance claim.

## Provenance

This ADR belongs to the Dr Moagi family and inherits attribution/provenance from ADR-015 and `docs/attribution/PERMEATION_MANIFEST.md`.
