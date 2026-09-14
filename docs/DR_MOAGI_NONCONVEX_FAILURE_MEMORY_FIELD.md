# Dr Moagi Non-Convex Failure-Memory Decision Field

**Historical error memory, asymmetric harm avoidance, and evidence-gated action selection**  
**Status:** Proposed research extension  
**Repository:** `Lord-Xido/Jarvis-X`  
**Depends on:** `DR_MOAGI_COGNITIVE_ENGINE.md`, `DR_MOAGI_TRANSCENDENT_COGNITIVE_ENGINE.md`, ADR-016, ADR-017, CTR, and `DR_MOAGI_OPERATIONAL_AUTOENCODING_EQUATION.md`  
**Figure:** `figures/dr-moagi-nonconvex-failure-memory-field.svg`

> The phrase “learning human stupidity to prevent repetition” is represented here in operational terms as **learning recurring decision-failure modes from evidence so that future candidates incur explicit, auditable risk penalties**. The model does not classify persons as stupid and does not treat historical frequency as proof of future wrongness.

---

## 1. Purpose

The **Non-Convex Failure-Memory Decision Field (NFDF)** is a candidate-scoring and trajectory-shaping layer for the Dr Moagi Cognitive Engine. It represents decision alternatives as trajectories over a bounded state manifold and augments ordinary task utility with explicit fields for:

- historical failure recurrence;
- sunk-cost persistence;
- short-horizon greed;
- group-conformity cascades;
- overconfidence;
- neglected biosafety or other domain-specific hazards;
- bias and calibration defects;
- resource inefficiency;
- asymmetric irreversible harm;
- evidence-supported rational attractors.

The field is **non-convex by design** because real decision spaces can contain multiple local minima, barriers, discontinuous constraints, and competing objectives.

NFDF is not an oracle. Its potential is learned or constructed from evidence, uncertainty and policy, then evaluated by CTR and the canonical candidate-first promotion gate.

---

## 2. Architectural position

Let the cognitive engine propose a decision/action candidate `a` from state `S_t`. NFDF inserts a bounded decision-dynamics operator before authoritative promotion:

```text
perception / model state
        ↓
candidate actions / trajectories
        ↓
NFDF potential + historical failure memory
        ↓
trajectory refinement / repulsion / attraction
        ↓
CTR contrast against evidence and counterevidence
        ↓
Pi_Lambda admissibility + resource + safety gates
        ↓
COMMIT or ROLLBACK
```

The outer law remains unchanged:

\[
S_{t+1}
=
V_t\,\Pi_\Lambda(S^{\mathrm{cand}}_{t+1})
+(1-V_t)S_t.
\]

NFDF therefore influences **candidate generation and ranking**, not the authority boundary itself.

---

## 3. Decision manifold and state

Let the executable decision state inhabit a finite-dimensional manifold

\[
q_t\in\mathcal M_D\subset\mathbb R^d,
\]

with a task-dependent metric `G(q)` when non-Euclidean geometry is useful.

A decision trajectory is

\[
\gamma:[0,T]\rightarrow\mathcal M_D,
\qquad
\gamma(0)=q_t.
\]

The field receives a typed context

\[
C_t=
(X_t,\Omega_t,R_{CTR,t},\Theta_t,\Pi_{run,t},A_{arch,t},E_t),
\]

where `Omega_t` is historical/adaptive memory and `R_CTR,t` carries evidence and counterevidence.

---

## 4. Master non-convex potential

Define the total decision potential

\[
\boxed{
\mathcal S(q\mid C_t)
=
\mathcal S_{task}(q)
+\lambda_H\mathcal S_{hist}(q)
+\lambda_B\mathcal S_{bias}(q)
+\lambda_R\mathcal S_{risk}(q)
+\lambda_C\mathcal S_{constraint}(q)
+\lambda_A\mathcal S_{asym}(q)
-\lambda_U\mathcal U_{evidence}(q)
}
\]

where lower potential is preferred subject to admissibility.

The terms are:

- `S_task`: task cost / negative utility;
- `S_hist`: historical recurrence/failure-memory energy;
- `S_bias`: calibration and known decision-bias penalties;
- `S_risk`: uncertainty and tail-risk penalty;
- `S_constraint`: hard/soft domain constraint energy;
- `S_asym`: asymmetric harm penalty for irreversible or high-consequence error;
- `U_evidence`: evidence-supported expected utility.

No single term is authoritative by naming. Every term must expose its data provenance, scale and uncertainty.

---

## 5. Historical failure memory

Let the evidence ledger contain failure prototypes

\[
\mathcal F_t=\{(f_i,\Sigma_i,w_i,\kappa_i)\}_{i=1}^{N_F},
\]

where:

- `f_i` is a failure-state embedding;
- `Sigma_i` is an uncertainty/covariance model;
- `w_i` is evidence strength;
- `kappa_i` is consequence severity.

A smooth recurrence field is

\[
\boxed{
\mathcal S_{hist}(q)
=
\sum_{i=1}^{N_F}
 w_i\,\kappa_i\,
 K_{\Sigma_i}(q,f_i)
}
\]

with, for example,

\[
K_{\Sigma_i}(q,f_i)
=
\exp\!\left(
-\tfrac12(q-f_i)^T\Sigma_i^{-1}(q-f_i)
\right).
\]

Because minimizing a positive Gaussian bump naturally avoids its center, historically supported failure modes act as **repulsive regions** rather than attractive sinks in the actual optimization law.

The memory update is bounded:

\[
\Omega_{t+1}^{cand}
=
\rho\Omega_t
+(1-\rho)\,\Delta\Omega(E_t,R_{CTR,t}),
\]

and is committed only after the normal verification gate.

### Required anti-bias rule

Historical recurrence is not sufficient evidence of causation. Before a failure prototype receives material weight, CTR must record:

```text
claim / failure mode
observed outcome
causal hypothesis
supporting evidence
counterevidence
confounders
confidence
severity
scope of applicability
expiry / review condition
```

This prevents the field from becoming a mechanism for institutionalizing inherited prejudice or stale policy.

---

## 6. Named decision-failure basins

The conceptual visualization names several illustrative failure classes. In the implementation these are **typed hypotheses**, not universal truths.

### 6.1 Sunk-cost persistence

Let `c_past` be irrecoverable historical expenditure and `V_future(a)` the forward-looking value of action `a`. A sunk-cost indicator can be represented as

\[
b_{sunk}
=
\operatorname{corr}(a,c_{past}\mid V_{future},C_t),
\]

or a learned calibrated classifier. Its penalty contributes only when evidence shows that irreversible past cost is improperly affecting forward choice.

### 6.2 Short-horizon greed

For discount horizon `H`, compare short and long horizon utility:

\[
G_{short}(a)
=
U_{0:H_s}(a)-\eta\,U_{H_s:H_l}(a).
\]

Large positive `G_short` under material long-run harm can raise `S_bias`.

### 6.3 Groupthink cascade

Given agent beliefs `p_j(y)` and evidence-conditioned independent estimates `\tilde p_j(y)`, define excessive conformity

\[
B_{group}
=
\frac1N\sum_j
D_{KL}\!\left(p_j\,\|\,\tilde p_j\right)
\]

only after accounting for genuinely shared evidence. Correlation alone is not groupthink.

### 6.4 Overconfidence

For probabilistic forecasts, use calibration error rather than rhetorical labels:

\[
B_{over}
=
\sum_b
\frac{|I_b|}{N}
\left|\operatorname{acc}(I_b)-\operatorname{conf}(I_b)\right|.
\]

### 6.5 Biosafety or domain hazard neglect

For a hazard model with probability `p_h(a)` and consequence `L_h(a)`, the expected hazard term is

\[
R_h(a)=p_h(a)L_h(a),
\]

augmented by tail-sensitive terms when expected value understates catastrophic risk.

---

## 7. Asymmetric penalty for irreversible harm

Some errors are not symmetric. Let the signed decision error relative to an evidence-supported reference be `e(a)`. Define

\[
\boxed{
\mathcal S_{asym}(a)
=
\alpha_+\,[e(a)]_+^2
+\alpha_-\,[-e(a)]_+^2
}
\]

with `alpha_+ != alpha_-` when over-action and under-action have materially different consequences.

For catastrophic/irreversible domains, a coherent tail-risk term may be added:

\[
\operatorname{CVaR}_{\tau}(L(a))
=
\min_{\xi}
\left[
\xi+\frac{1}{1-\tau}\,\mathbb E[(L(a)-\xi)_+]
\right].
\]

The resulting risk field can be

\[
\mathcal S_{risk}(a)
=
\mathbb E[L(a)]
+\lambda_{tail}\operatorname{CVaR}_{\tau}(L(a)).
\]

---

## 8. Bias, resource and risk indices

The labels shown in the conceptual figure map to measurable receipts.

### Bias index

\[
I_B
=
\sum_k \omega_k\,\hat b_k,
\qquad
\omega_k\ge0,
\quad
\sum_k\omega_k=1.
\]

Each `b_k` must be a calibrated, separately reported diagnostic.

### Resource efficiency

For useful verified work `W_v` and resource cost `C_r`, use

\[
\eta_R
=
\frac{W_v}{W_v+C_r+\varepsilon}.
\]

This remains distinct from raw throughput.

### Risk tolerance

Risk tolerance is a declared policy parameter, not a learned moral fact:

\[
\tau_R=\Pi_{risk}(context,authority,domain).
\]

The runtime may optimize within `tau_R`; it may not silently redefine it.

---

## 9. Rational attractors

A “rational attractor” is operationally defined as a basin around an admissible, evidence-supported local optimum — not a metaphysical definition of rationality.

Let

\[
q^*
\in
\arg\min_{q\in\mathcal A_t}
\mathcal S(q\mid C_t),
\]

where `A_t` is the admissible set imposed by `Pi_Lambda` and domain constraints.

A local attractor certificate may require

\[
\|\nabla \mathcal S(q^*)\|<\epsilon_g,
\qquad
\lambda_{min}(\nabla^2\mathcal S(q^*))> -\epsilon_H,
\]

plus CTR evidence correspondence.

Because the field is non-convex, the system must not assume that a locally stable attractor is globally optimal.

---

## 10. Decision dynamics

### 10.1 First-order descent with repulsive correction

A bounded candidate trajectory can obey

\[
\boxed{
\dot q
=
-M(q)\nabla_q\mathcal S(q\mid C_t)
+u_{explore}(t)
+u_{constraint}(t)
}
\]

where `M(q)` is positive semidefinite.

### 10.2 Inertial dynamics

For momentum-aware exploration:

\[
\boxed{
\ddot q
+\Gamma(q)\dot q
=-M(q)\nabla\mathcal S(q)
+u(t)
}
\]

with damping `Gamma` chosen to prevent unstable oscillation.

### 10.3 Non-Euclidean form

On metric `g_ij(q)`, the forced geodesic equation is

\[
\ddot q^k
+\Gamma^k_{ij}\dot q^i\dot q^j
=
-g^{k\ell}\partial_\ell\mathcal S(q)
+u^k.
\]

This links NFDF directly to the metric-aware latent geometry in the Transcendent Engine specification.

---

## 11. Preventing repetition without preventing novelty

A naive repulsive memory can overfit the past and suppress legitimate innovation. NFDF therefore uses three controls.

### 11.1 Uncertainty-weighted repulsion

\[
w_i^{eff}=w_i\,(1-u_i),
\qquad 0\le u_i\le1,
\]

so uncertain historical analogies exert less force.

### 11.2 Contextual similarity gate

\[
g_i(q,C_t)
=\sigma\!\left(
\frac{sim(q,f_i,C_t)-\tau_i}{T_i}
\right).
\]

Historical penalties apply only when context is sufficiently analogous.

### 11.3 Counterfactual escape

If a candidate resembles a past failure but differs on the causal factor that produced the failure, CTR may reduce the historical penalty after explicit counterfactual testing.

Thus the governing principle is:

```text
remember the mechanism of failure,
not merely the superficial shape of the past.
```

---

## 12. CTR integration

For candidate `a_j`, CTR records

\[
r_j=
[claim,prediction,observation,error,source,confidence,counterexample,status].
\]

NFDF extends this with

```text
potential_total
historical_failure_energy
bias_energy
risk_energy
constraint_energy
asymmetric_harm_energy
expected_utility
nearest failure prototypes
counterfactual escape tests
trajectory stability
```

The decision field and CTR form a dual system:

\[
\boxed{
\text{NFDF proposes and shapes trajectories}
\quad\leftrightarrow\quad
\text{CTR tests their evidential justification}
}
\]

---

## 13. Integration with the Psi-Phi-Lambda-Omega-Theta stack

The conceptual figure maps to the stack as follows:

- `Psi`: current multimodal/decision state;
- `Phi`: candidate description, simulation and rendering operator;
- `Lambda^{-1}` / `Pi_Lambda`: contextual constraints and authoritative admissibility boundary;
- `Omega`: historical failure/residual memory;
- `Theta`: learned model and calibrated weighting parameters.

NFDF can be written as a stack-conditioned potential

\[
\boxed{
\mathcal S_t
=\mathcal S
\left(
q;\Psi_t,\Phi_t,\Lambda_t^{-1},\Omega_t,\Theta_t,R_{CTR,t}
\right).
}
\]

The stack does not bypass verification: all learned weights and candidate actions remain provisional until promoted.

---

## 14. Candidate-selection law

For a finite candidate set `A_t={a_1,...,a_K}`, define

\[
J(a_j)
=\mathcal S(a_j\mid C_t)
+\lambda_U UQ(a_j)
\]

where `UQ` is an uncertainty penalty.

The provisional action is

\[
\boxed{
a_t^{cand}=\arg\min_{a_j\in A_t}J(a_j)
}
\]

subject to hard exclusions

\[
a_j\notin\mathcal A_t\Rightarrow J(a_j)=+\infty.
\]

Promotion occurs only when

\[
V_t
=
V_{CTR}\land V_{policy}\land V_{risk}\land V_{resource}\land V_{numerical}\land V_{provenance}.
\]

Then

\[
a_t=
\begin{cases}
a_t^{cand},&V_t=1,\\
a_t^{safe/default},&V_t=0.
\end{cases}
\]

---

## 15. Learning the field

Let `theta_S` parameterize differentiable parts of the potential. A supervised/ranking objective can use observed outcome quality `y`:

\[
\mathcal L_{rank}
=
\sum_{(i,j):y_i>y_j}
\log\!\left(1+e^{-(\mathcal S(q_j)-\mathcal S(q_i))}\right).
\]

A calibration term compares predicted risk with observed frequency/severity:

\[
\mathcal L_{cal}
=
\sum_b
\left(\hat p_b-p_b^{obs}\right)^2.
\]

A stability term penalizes pathological gradient fields:

\[
\mathcal L_{smooth}
=\mathbb E_q\|\nabla^2\mathcal S(q)\|_F^2.
\]

The finite training objective may be

\[
\boxed{
\mathcal L_{NFDF}
=\lambda_r\mathcal L_{rank}
+\lambda_c\mathcal L_{cal}
+\lambda_s\mathcal L_{smooth}
+\lambda_p\mathcal L_{provenance}
}
\]

with all updates staged as candidates under ADR-016/017.

---

## 16. Runtime pseudocode

```python
def nfdf_step(state, candidates, evidence, policy):
    field = build_potential(
        state=state,
        failure_memory=state.memory.failure_prototypes,
        evidence=evidence,
        policy=policy,
    )

    scored = []
    for a in candidates:
        if not policy.admissible(a):
            continue
        trajectory = refine_trajectory(a, field, bounded=True)
        receipt = field.receipt(trajectory)
        ctr = contrast_and_reckon(trajectory, evidence, receipt)
        scored.append((trajectory, receipt, ctr))

    candidate = select_best_verified_candidate(scored)
    staged = stage_updates(state, candidate)

    if verify(staged, evidence, policy):
        return commit(staged)
    return rollback(state)
```

---

## 17. Verification receipts

Every NFDF execution should record at minimum:

```text
candidate/action identity
field/model version
historical prototype IDs and provenance
potential decomposition by term
uncertainty and calibration metrics
constraint violations
risk and tail-risk metrics
trajectory iterations and termination
local gradient norm
CTR evidence / counterevidence result
resource cost
commit or rollback outcome
```

A scalar “rationality score” without this decomposition is insufficient for auditability.

---

## 18. Benchmark contract

NFDF should be evaluated against baselines on controlled decision environments containing known recurrent failure modes.

Report separately:

1. repeated-failure rate;
2. false-avoidance rate — safe novel actions incorrectly repelled;
3. calibration error;
4. expected utility / regret;
5. tail loss / CVaR;
6. constraint-violation rate;
7. recovery after distribution shift;
8. ablation without historical memory;
9. ablation without CTR counterevidence;
10. compute and memory overhead.

A reduction in repeated error is meaningful only if it does not come from trivially refusing all actions.

---

## 19. Capability and epistemic boundary

NFDF does **not** establish that:

- history deterministically predicts the future;
- an encoded potential is objectively rational;
- a local minimum is globally optimal;
- a bias classifier is unbiased;
- a historical failure prototype proves causation;
- topology or optimization replaces empirical evidence;
- the engine can eliminate human error.

It establishes a more precise engineering contract:

\[
\boxed{
\text{observed failure}
\rightarrow
\text{provenanced memory}
\rightarrow
\text{contextual risk field}
\rightarrow
\text{candidate trajectory correction}
\rightarrow
\text{CTR challenge}
\rightarrow
\text{verified commit or rollback}
}
\]

The intended behavior is not “never repeat anything that once failed.” It is:

> **Do not repeat a previously evidenced failure mechanism without new evidence strong enough to justify why the present case is materially different.**

---

## 20. Canonical compact form

The complete NFDF decision layer is summarized by

\[
\boxed{
\begin{aligned}
\mathcal S_t(q)
&=\mathcal S_{task}
+\lambda_H\mathcal S_{hist}
+\lambda_B\mathcal S_{bias}
+\lambda_R\mathcal S_{risk}
+\lambda_C\mathcal S_{constraint}
+\lambda_A\mathcal S_{asym}
-\lambda_U\mathcal U_{evidence},\\
\dot q
&=-M(q)\nabla\mathcal S_t(q)+u_{explore}+u_{constraint},\\
a_t^{cand}
&=\arg\min_{a\in\mathcal A_t}\mathcal S_t(a),\\
V_t
&=V_{CTR}\land V_{policy}\land V_{risk}\land V_{resource}\land V_{numerical}\land V_{provenance},\\
S_{t+1}
&=V_t\Pi_\Lambda(S_{t+1}^{cand})+(1-V_t)S_t.
\end{aligned}
}
\]

This turns the conceptual non-convex landscape into an implementable, auditable component of the Dr Moagi Cognitive Engine.