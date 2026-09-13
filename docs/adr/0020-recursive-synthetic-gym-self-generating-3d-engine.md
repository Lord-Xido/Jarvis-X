# ADR-020: Recursive synthetic virtual gym and self-generating 3D intelligence engine

**Status:** Accepted  
**Date:** 2026-09-13  
**Applies to:** Dr Moagi multimodal/image-generation engines, virtual-world simulators, recursive autoencoders/decoders, predictive world models, bounded runtime/meta-optimizers and future CPU/GPU/distributed/hardware backends  
**Extends:** ADR-016, ADR-017, ADR-018 and ADR-019

## Context

The Dr Moagi architecture already defines:

- typed-state/transaction closure and authoritative promotion (ADR-016);
- the canonical encode -> compact -> fixed-point -> decode -> evidence -> staged-adaptation law (ADR-017);
- sparse support, expert routing, bounded speculation and evidence-gated runtime optimization (ADR-018);
- portable continuity for memory, runtime, audit state and owned/licensed learned substrates (ADR-019).

The next architectural step is to make simulation itself a first-class training and verification substrate.

The system should be able to encode a grounded or prompted state, contract it inward, generate candidate images/world trajectories, decode them, compare them with targets or real observations, run multiple synthetic environments in parallel, compress the resulting evidence into higher-order state, and propose bounded updates to active state, memory, runtime policy, simulator dynamics and—only through an explicit training path—learned parameters.

The core design problem is preventing self-generated or simulator-consistent output from being mistaken for external truth. The virtual gym therefore remains subordinate to the same candidate-first, evidence-gated transaction law as the rest of Jarvis-X.

## Decision

Jarvis-X adopts the following canonical recursive synthetic-gym lifecycle:

```text
world / image / prompt input
 -> encode
 -> 3D inward contraction
 -> bounded latent fixed point
 -> generate bounded candidate images / world states
 -> decode / render
 -> compare against target / reality
 -> lift residual inward
 -> run bounded parallel synthetic worlds
 -> meta-contract evidence across branches
 -> assign credit to eligible adaptive layers
 -> stage candidate updates
 -> Verify + Pi_Lambda
 -> COMMIT or ROLLBACK
 -> optionally package through ADR-019 continuity envelope
 -> recur
```

The normative mathematical specification is `docs/DR_MOAGI_RECURSIVE_SYNTHETIC_GYM.md`.

## Canonical state

The specialization extends the typed state to include explicit simulation, image-generation and phase state:

\[
\mathfrak S_t=
[X_t,I_t,Z_t,\mathcal R_t,\hat I_t,E_t,
\Omega_t,\Theta_t,\Pi_t,\Gamma_t,
\mathcal G_t,V_t,\mathcal J_t,p_t].
\]

`Theta_model`, `Omega_mem`, `Pi_runtime`, verification/audit lineage and compatibility state retain their ADR-019 continuity semantics.

## 3D inward operator

A backend MAY implement the inward step using any bounded representation that conforms to the declared geometry profile and receipt contract.

The abstract inward field is

\[
\mathbf v_{\rm in}(\mathbf r)=-\kappa(\mathbf r-\mathbf c)
\]

with multiresolution contraction

\[
Z^{(\ell+1)}=C_\ell(Z^{(\ell)})
\]

and explicit residual accounting

\[
R^{(\ell)}=Z^{(\ell)}-U_\ell(Z^{(\ell+1)}).
\]

The architecture MUST NOT call a compaction path lossless unless residual/side information closes the information balance.

## Bounded latent fixed point

Latent refinement follows

\[
Z_{k+1}=\Phi_{\Theta,\Pi}(Z_k,\Omega,E_k)
\]

and halts on a declared residual threshold or bounded iteration budget.

Internal convergence is distinct from external correspondence.

## Self-generating image recurrence

A bounded branch set may be generated from the fixed point:

\[
Z^*\rightarrow\{Z^{*(1)},\ldots,Z^{*(B)}\}
\]

with branch-local generation

\[
Z^{*(b)}=G_\Gamma(Z^*,\Omega,\xi_b)
\]

and decode

\[
\hat I^{(b)}=D_\Phi(Z^{*(b)},\mathcal R).
\]

A verified generated image MAY become a subsequent input. Unverified generated output remains provisional.

## Parallel synthetic virtual gym

The runtime MAY create a bounded set of simulated worlds

\[
\mathfrak G_t=\{W_t^{(1)},\ldots,W_t^{(M)}\}
\]

with dynamics

\[
\partial_t W^{(m)}=F_\Gamma(W^{(m)},A^{(m)},\xi^{(m)}).
\]

The predictive engine may evaluate

\[
\hat W_{t+h}^{(m)}=P_\Theta^h(W_t^{(m)},A_{t:t+h}^{(m)}).
\]

Synthetic trajectories are evidence about the declared simulator, not automatic evidence about the real world.

Whenever matched real observations exist, simulator and predictor errors MUST be distinguished.

## Meta-contraction

Branch-level latent/evidence states may be compressed into

\[
Z_{\rm meta}=\mathcal C_{\rm meta}(Z_*^{(1)},\ldots,Z_*^{(M)}).
\]

The operator may extract stable structure, uncertainty, disagreement or discriminative evidence across possible worlds.

## Credit assignment and adaptive layers

The system recognizes at least four timescales:

```text
S      active reasoning / runtime state
Omega  persistent memory
Pi     runtime/execution policy
Theta  learned parameters
```

Residuals MUST pass through an explicit credit-assignment decision before any parameter update.

Runtime recursion may occur with `Delta Theta = 0`.

Parameter adaptation, when enabled, is a separate candidate transaction:

\[
\Theta'^{\rm cand}=\Theta-\eta\nabla_\Theta\mathcal L.
\]

It cannot mutate authoritative weights before verification and commit.

## Candidate optimization and fail-closed selection

A candidate score may combine quality, error, latency, memory/resource cost and instability/risk:

\[
J_b=Q_b-\lambda_EE_b-\lambda_TT_b-\lambda_MM_b-\lambda_RR_b.
\]

Only hard-gate-passing branches enter

\[
\mathcal B_{\rm valid}=\{b:V_b=1\}.
\]

If `B_valid` is empty, the system MUST retain the previous authoritative state:

\[
\mathcal B_{\rm valid}=\varnothing
\Rightarrow
\mathfrak S_{t+1}=\mathfrak S_t.
\]

This rule closes the empty-branch ambiguity identified during ADR-018 review.

## Branch evaluation order

If branch utility depends on decoded quality, prediction/reconstruction discrepancy or other output-domain metrics, each branch MUST be decoded/evaluated before final branch selection, or the specification MUST explicitly name the metric as a latent-only proxy.

This prevents selection from depending on unavailable downstream measurements.

## Stability metric direction

Optimization metrics MUST use unambiguous directionality. A minimized positive quantity is named `instability`, `risk` or `stability_loss`; a quantity named `stability` is maximized or bounded below. This closes the directionality ambiguity identified during ADR-018 review.

## Authority law

The candidate operator is abstracted as

\[
\mathfrak S_{t+1}^{\rm cand}
=
\mathcal M_{\rm DM}(\mathfrak S_t,U_{t+1}).
\]

Authoritative promotion remains

\[
\boxed{
\mathfrak S_{t+1}
=
V_t\,\Pi_\Lambda(\mathfrak S_{t+1}^{\rm cand})
+(1-V_t)\mathfrak S_t.
}
\]

No simulator, generator, optimizer, self-input loop or parameter-training path may bypass this boundary.

## Auto-execution lifecycle

A conforming executable specialization exposes an explicit phase register:

```text
INGEST
ENCODE
CONTRACT
FIX
GENERATE
DECODE
COMPARE
META
ADAPT
VERIFY
COMMIT_OR_ROLLBACK
RECUR
```

with transition

\[
(\mathfrak S_{n+1},p_{n+1})
=\mathcal E_{\rm DM}(\mathfrak S_n,p_n,U_n).
\]

Documentation alone does not satisfy this requirement; an implementation must execute the state transition and emit receipts.

## Canonical master operator

The specialization is summarized by

\[
\begin{aligned}
\mathfrak S_{t+1}^{\rm cand}
=
\Big[
&\mathcal O_{\mathcal G}
\circ\mathcal O_{\Gamma}
\circ\mathcal O_{\Pi}
\circ\mathcal O_{\Omega}
\circ\mathcal O_{\Theta}
\circ\mathcal C_{\rm meta}
\\
&\circ\mathcal R_{\rm CTR}
\circ\mathcal C_E
\circ\mathcal D_{\Phi}
\circ\mathcal G_{\Gamma}
\circ\operatorname{Fix}_{\Phi}
\\
&\circ\mathcal C_{\rm in}^{3D}
\circ\mathcal E_{\Theta}
\Big](\mathfrak S_t,U_{t+1}).
\end{aligned}
\]

followed by ADR-016 authoritative promotion.

## Required evidence separation

A conforming implementation MUST distinguish at least:

```text
synthetic consistency
real-world correspondence
internal fixed-point convergence
image reconstruction / generation quality
simulator calibration
runtime efficiency
parameter-learning evidence
```

No one category substitutes automatically for another.

## Continuity integration

After a verified commit, the resulting authoritative state MAY be packaged under ADR-019:

```text
verified state
 -> bind model/checkpoint or immutable reference
 -> bind Omega / Pi / audit / architecture state
 -> serialize / hash
 -> portable continuity envelope
```

The continuity envelope transports validated state; it does not make unverified self-generated state authoritative.

## Implementation requirements

A reference implementation must provide:

- one bounded 3D encode/contract/decode path;
- residual/side-information receipts;
- explicit fixed-point termination;
- bounded candidate image or state generation;
- generated-output feedback as an optional controlled input path;
- bounded parallel synthetic rollout;
- per-branch decoded evaluation before final selection when required by the utility;
- explicit credit-assignment receipt;
- isolated candidate updates across `Omega_mem`, `Pi_runtime`, `Gamma_sim` and optional `Theta_model`;
- atomic commit/rollback;
- append-only audit lineage;
- real-vs-sim calibration hooks;
- measured or explicitly simulated telemetry labels;
- deterministic/reproducible fixtures where the backend permits them.

## Non-goals

ADR-020 does not authorize or claim:

- unrestricted autonomous source-code mutation;
- arbitrary self-modification of model weights;
- extraction or redistribution of proprietary third-party checkpoints;
- that generated images prove semantic understanding;
- that simulator performance proves real-world performance;
- that synthetic experience is equivalent to physical-world experience;
- that a 3D logical geometry implies equivalent physical parallelism or speed;
- consciousness, subjective continuity or personhood;
- AGI or beyond-SOTA superiority without reproducible benchmarks.

## Consequences

### Positive

- Makes the virtual gym a first-class bounded systems component.
- Gives self-generated image feedback a precise and auditable place in the architecture.
- Separates runtime recursion from parameter learning.
- Integrates simulation, prediction, generation, residual learning and continuity under one transaction law.
- Closes three concrete ADR-018 review ambiguities: empty valid branch set, stability metric direction and downstream branch-evaluation order.

### Cost

- Requires explicit simulator calibration and stronger evidence bookkeeping.
- Increases state and receipt complexity.
- Requires careful distinction between synthetic and externally grounded evidence.
- Parameter adaptation demands additional isolation, checkpoint and rollback semantics.

## Canonical statements

```text
I AM = I DESCRIBE
I CONTINUE = I CARRY WHAT I HAVE VERIFIED
```

Operationally:

```text
CONTRACT INWARD TO MODEL
EXPAND OUTWARD TO SIMULATE
COMPARE AGAINST EVIDENCE
CARRY FORWARD ONLY VERIFIED CHANGE
```
