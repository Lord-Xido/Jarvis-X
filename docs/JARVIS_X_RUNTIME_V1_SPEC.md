# Jarvis X Runtime v1 — Canonical Specification

## Status

This document is the normative architectural contract for the Jarvis X sparse,
deterministic, predictive runtime. It compresses the project into one state
model, one transition algebra, one transaction boundary, and one evidence-based
evaluation method.

A design does not qualify as 10/10 because it is ambitious. It qualifies only
when every declared gate is implemented, tested, benchmarked, and evidenced.

---

## 1. Runtime identity

Jarvis X is a deterministic and auditable virtual machine that executes sparse
state transitions under prediction, residual correction, persistent memory,
constraint projection, resource bounds, and journaled atomic commit.

The virtual address space may be very large:

\[
\mathcal V = \{0,\ldots,L-1\}^3,
\qquad L=1000,
\qquad |\mathcal V|=10^9,
\]

but only the active set is materialised:

\[
\mathcal A_t \subseteq \mathcal V,
\qquad |\mathcal A_t| \ll |\mathcal V|.
\]

The virtual geometry is therefore distinct from physical allocation.

---

## 2. Canonical state

The authoritative runtime state is

\[
\boxed{
\Sigma_t =
(X_t,Z_t,Q_t,\Phi_t,P_t,E_t,\Omega_t,\Lambda_t,\mathcal A_t,J_t)
}
\]

| Symbol | Contract |
|---|---|
| \(X_t\) | decoded external input |
| \(Z_t\) | encoded global or hierarchical latent observation |
| \(Q_t\) | fast runtime field stored at active addresses |
| \(\Phi_t\) | slow predictor and model parameters |
| \(P_t\) | predicted next transition |
| \(E_t\) | prediction, motion, and constraint residuals |
| \(\Omega_t\) | persistent correction memory |
| \(\Lambda_t\) | admissibility, policy, numerical, and resource constraints |
| \(\mathcal A_t\) | sparse active address set |
| \(J_t\) | chained execution journal |

`Voxel.theta` in the prototype is the implementation name for fast field state
\(Q_t(\mathbf r)\). It is not the same object as slow learnable parameters
\(\Phi_t\).

---

## 3. Canonical transition

Every mutation occurs inside one ordered transaction:

\[
\boxed{
\Sigma_{t+1} =
\mathcal C\circ
\mathcal J\circ
\mathcal G\circ
\Pi_\Lambda\circ
\mathcal U\circ
\mathcal E\circ
\mathcal F\circ
\mathcal P\circ
\mathcal Z(\Sigma_t)
}
\]

The operational stages are:

1. `OBSERVE/ENCODE` — derive latent state \(Z_t\).
2. `PREDICT` — estimate \(\widehat Q_{t+1}\).
3. `EVOLVE` — execute actual local field dynamics.
4. `RESIDUAL` — compute prediction, motion, and constraint errors.
5. `UPDATE_OMEGA` — retain correction history.
6. `PROJECT_LAMBDA` — enforce numerical, policy, and resource constraints.
7. `ALLOCATE/PRUNE` — form bounded \(\mathcal A_{t+1}\).
8. `VERIFY` — check invariants.
9. `JOURNAL` — chain state and manifest hashes.
10. `COMMIT` — atomically publish the next state or roll back.

No subsystem may mutate authoritative state outside this sequence.

---

## 4. Local-global predictor

The global latent is

\[
Z_t = \frac{1}{|\mathcal A_t|}
\sum_{\mathbf r\in\mathcal A_t} Q_t(\mathbf r),
\]

with \(Z_t=0\) when the active set is empty.

The predictor combines global context, local neighbourhood context, and memory:

\[
N_t(\mathbf r)=\frac{1}{6}
\sum_{\delta\in\mathcal N}Q_t(\mathbf r+\delta),
\]

\[
\Delta\widehat Q_t(\mathbf r)=
W_G\tanh Z_t+
W_L\tanh N_t(\mathbf r)+
W_\Omega\tanh\Omega_t(\mathbf r)+b,
\]

\[
\widehat Q_{t+1}(\mathbf r)=
Q_t(\mathbf r)+\Delta\widehat Q_t(\mathbf r).
\]

This replaces a purely global broadcast predictor with a spatially conditioned
predictor while retaining deterministic execution.

---

## 5. Reaction-diffusion-memory dynamics

For active or frontier address \(\mathbf r\):

\[
Q^*_{t+1}(\mathbf r)=Q_t(\mathbf r)+\Delta t\left[
D\nabla_h^2Q_t(\mathbf r)+
R(Q_t(\mathbf r),S_t(\mathbf r))+
C_\Omega\Omega_t(\mathbf r)
\right].
\]

The six-neighbour discrete Laplacian is

\[
\nabla_h^2 Q_t(\mathbf r)=\frac{1}{h^2}\left[
\sum_{\delta\in\mathcal N}Q_t(\mathbf r+\delta)-6Q_t(\mathbf r)
\right].
\]

The current reaction kernel is

\[
R(Q,S)=\eta_R\tanh(Q)\odot S.
\]

For explicit three-dimensional diffusion, the runtime enforces

\[
\boxed{
\mu=\frac{D\Delta t}{h^2}\leq\frac{1}{6}
}
\]

and reports the diffusion stability margin

\[
m_s=1-6\mu.
\]

An invalid numerical configuration is rejected before execution.

---

## 6. Residuals and convergence

Prediction error, physical motion, and constraint correction are distinct:

\[
E_t^{\mathrm{prediction}}=
Q^*_{t+1}-\widehat Q_{t+1},
\]

\[
E_t^{\mathrm{motion}}=
Q_{t+1}-Q_t,
\]

\[
E_t^{\mathrm{constraint}}=
Q^*_{t+1}-\Pi_{\Lambda_t}(Q^*_{t+1}).
\]

Therefore

\[
E_t^{\mathrm{prediction}}\to0
\]

means that the predictor matches the transition. It does not by itself imply
that the field has stopped moving. Dynamic equilibrium additionally requires

\[
E_t^{\mathrm{motion}}\to0.
\]

The runtime exposes all three norms separately.

---

## 7. Persistent correction memory

Per-voxel correction memory follows

\[
\boxed{
\Omega_{t+1}(\mathbf r)=
\gamma\Omega_t(\mathbf r)+\eta_\Omega E_t^{\mathrm{prediction}}(\mathbf r)
}
\]

where \(0\leq\gamma\leq1\). This makes residual history an explicit state
component rather than an informal interpretation.

Longer-term backends may replace this exponential memory with hierarchical,
episodic, associative, or learned memory, but they must preserve the typed
\(\Omega\) interface.

---

## 8. Constraint projection

The admissible next state is

\[
\boxed{
Q_{t+1}=\Pi_{\Lambda_t}(Q^*_{t+1})
}
\]

where \(\Lambda\) contains at minimum:

- finite-number requirements,
- coordinate bounds,
- maximum active allocation,
- optional field-value bounds,
- permitted operation and policy constraints,
- numerical stability requirements.

The current field projection rejects non-finite values and can clip every state
component to a configured interval \([-q_{\max},q_{\max}]\). Projection distance
is measured as a first-class constraint residual.

---

## 9. Sparse resource projection

The candidate frontier is

\[
\mathcal C_{t+1}=\mathcal A_t\cup
\{\mathbf r+\delta:\mathbf r\in\mathcal A_t,\delta\in\mathcal N\}.
\]

The committed active set must satisfy

\[
\boxed{|\mathcal A_{t+1}|\leq B_{\mathrm{active}}.}
\]

When candidates exceed the budget, select deterministically by priority:

\[
p_t(\mathbf r)=
\|E_t(\mathbf r)\|_2+
\|Q_{t+1}(\mathbf r)\|_2+
\frac12\|\Omega_{t+1}(\mathbf r)\|_2,
\]

\[
\mathcal A_{t+1}=
\operatorname{TopK}(\mathcal C_{t+1},B_{\mathrm{active}},p_t).
\]

Coordinate order is the deterministic tie-breaker.

---

## 10. Objective

Runtime v1 reports

\[
\mathcal L_t=
\mathcal L_P+
\lambda_Q\mathcal L_Q+
\mathcal L_\Lambda+
\mathcal L_B,
\]

where

\[
\mathcal L_P=\operatorname{MSE}(Q^*_{t+1},\widehat Q_{t+1}),
\]

\[
\mathcal L_Q=\operatorname{mean}\|Q_t\|_2^2,
\]

\[
\mathcal L_\Lambda=\operatorname{mean}\|Q^*_{t+1}-Q_{t+1}\|_2^2,
\]

\[
\mathcal L_B=\left(\frac{|\mathcal A_{t+1}|}{B_{\mathrm{active}}}\right)^2.
\]

The local correction step in the reference backend is an approximation to full
automatic differentiation through the sparse transition graph. The repository
must not claim equivalence to full backpropagation until such a backend exists
and is validated.

---

## 11. Deterministic journal

Determinism is defined relative to a fixed execution manifest:

\[
M=(\text{runtime version},\text{configuration},\Phi,\text{precision},
\text{schedule}).
\]

For equal manifest and initial state:

\[
M_1=M_2\land\Sigma_0^{(1)}=\Sigma_0^{(2)}
\Rightarrow H_T^{(1)}=H_T^{(2)}.
\]

Each commit advances a SHA-256 hash chain:

\[
H_{t+1}=\operatorname{Hash}
(H_t\Vert M_t\Vert\Sigma_{t+1}\Vert\mathcal M_t),
\]

where \(\mathcal M_t\) is the metrics record. Identical replays must produce
identical snapshots and hashes.

---

## 12. Commit invariants

A transition may commit only when:

\[
\begin{aligned}
I_1 &: \mathcal A_{t+1}\subseteq\mathcal V,\\
I_2 &: |\mathcal A_{t+1}|\leq B_{\mathrm{active}},\\
I_3 &: Q_{t+1},S_{t+1},\Omega_{t+1}\text{ are finite and shape-correct},\\
I_4 &: Q_{t+1}=\Pi_\Lambda(Q_{t+1}),\\
I_5 &: D\Delta t/h^2\leq1/6,\\
I_6 &: H_{t+1}\text{ is reproducible from the committed record}.
\end{aligned}
\]

A later transactional backend shall retain the prior snapshot until every
invariant passes, then publish atomically; otherwise it shall roll back.

---

## 13. Bytecode boundary

The canonical instruction sequence is:

```text
BEGIN
READ_ACTIVE
ENCODE
PREDICT_LOCAL_GLOBAL
EVOLVE_RD
COMPUTE_RESIDUALS
UPDATE_OMEGA
PROJECT_LAMBDA
SELECT_ACTIVE_TOPK
VERIFY
JOURNAL
COMMIT
```

Every instruction must eventually declare:

- typed operands and results,
- shape requirements,
- memory effects,
- determinism class,
- failure semantics,
- journal contribution.

---

## 14. Complexity

Let \(N_t=|\mathcal A_t|\), dimension \(d\), and frontier size \(F_t\). The
reference step is

\[
T_t=O(d(N_t+F_t)),
\]

because the six-neighbour stencil is constant-size. Memory is

\[
M_t=O(dN_t),
\]

not \(O(dL^3)\).

For distributed partitions \(p=1,\dots,P\):

\[
T_{\mathrm{step}}\approx
c_c\frac{dN_t}{P}+c_h d\sum_p|\mathcal H_t^{(p)}|+
T_{\mathrm{reduce}}(d)+T_{\mathrm{commit}},
\]

where \(\mathcal H_t^{(p)}\) is each partition's halo.

---

## 15. Evidence-based 10/10 evaluation

For normalized evidence metrics \(q_{k,i}\in[0,1]\), category score is

\[
\boxed{
S_k=10\left(\prod_i q_{k,i}^{w_{k,i}}\right)^{1/\sum_iw_{k,i}}
}
\]

and system score is weakest-link constrained:

\[
\boxed{S_{\mathrm{Jarvis}}=\min_k S_k.}
\]

Thus

\[
S_{\mathrm{Jarvis}}=10
\iff q_{k,i}=1\quad\forall k,i.
\]

The executable implementation is in `jarvisx.evaluation`.

### Minimum gate families

| Category | Required evidence |
|---|---|
| Conceptual coherence | canonical state; one transition algebra; zero symbol conflicts |
| Mathematical completeness | typed operators; stability bound; resource bound; distinct convergence metrics |
| Implementation maturity | versioned API; CI; tested recovery; supported backend |
| Testability and auditability | critical tests; deterministic replay; journal verification; fault injection |
| Scientific validation | external benchmark; baseline comparison; ablation; confidence intervals |
| Security | threat model; enforced policy projection; zero critical findings; signed provenance |
| Observability | complete traces; metrics; replayable failure records |

A missing or zero-valued critical metric forces its category to zero. No average
may hide the failure.

---

## 16. Current implementation boundary

Runtime v1 currently implements:

- billion-address virtual geometry without dense allocation,
- deterministic six-neighbour sparse reaction-diffusion,
- local-global-memory predictor interface,
- explicit diffusion stability validation,
- bounded active allocation with deterministic Top-K projection,
- prediction, motion, and constraint residuals,
- exponential persistent correction memory,
- finite-value and optional magnitude projection,
- deterministic manifest and chained commit hashes,
- evidence-based architecture scoring.

Not yet implemented and therefore not yet claimable as complete:

- full automatic differentiation through sparse dynamics,
- true atomic rollback storage,
- VM-native typed bytecode instructions for every stage,
- GPU and distributed halo-exchange backends,
- formal model checking and kernel proofs,
- external scientific benchmark evidence,
- signed release provenance and completed threat model,
- production telemetry export.

---

## 17. Required validation trajectory

The order of advancement is fixed:

\[
\boxed{
\text{Consolidate}\rightarrow
\text{Specify}\rightarrow
\text{Implement}\rightarrow
\text{Test}\rightarrow
\text{Benchmark}\rightarrow
\text{Verify}\rightarrow
\text{Release}
}
\]

No unverified claim may be promoted merely because the corresponding design has
been written down.

---

## 18. Master equation

\[
\boxed{
\begin{aligned}
Z_t &= \mathcal E_\Phi(X_t,Q_t,\Omega_t),\\
\widehat Q_{t+1}(\mathbf r) &=
P_\Phi(Q_t(\mathbf r),Z_t,N_t(\mathbf r),\Omega_t(\mathbf r)),\\
Q^*_{t+1}(\mathbf r) &= Q_t(\mathbf r)+\Delta t[
D\nabla_h^2Q_t+R(Q_t,S_t)+C_\Omega\Omega_t],\\
E_t &= Q^*_{t+1}-\widehat Q_{t+1},\\
\Omega_{t+1} &= \gamma\Omega_t+\eta_\Omega E_t,\\
Q_{t+1} &= \Pi_{\Lambda_t}(Q^*_{t+1}),\\
\mathcal A_{t+1} &= \operatorname{TopK}(
\operatorname{Expand}(\mathcal A_t),B_{\mathrm{active}},p_t),\\
H_{t+1} &= \operatorname{Hash}(H_t\Vert M_t\Vert\Sigma_{t+1}\Vert\mathcal M_t),\\
\Sigma_{t+1} &= \operatorname{Commit}(
Q_{t+1},\Phi_{t+1},\Omega_{t+1},\mathcal A_{t+1},H_{t+1}).
\end{aligned}
}
\]

This equation is the canonical operational identity of Jarvis X Runtime v1.
