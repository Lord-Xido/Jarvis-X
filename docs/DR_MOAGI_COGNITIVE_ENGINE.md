# The Dr Moagi Cognitive Engine

**Architectural Specification & Theoretical Framework**  
**Status:** Proposed canonical research specification  
**Repository:** `Lord-Xido/Jarvis-X`  
**Architectural dependencies:** `docs/ARCHITECTURE.md`, ADR-016, ADR-017, and `docs/DR_MOAGI_OPERATIONAL_AUTOENCODING_EQUATION.md`  
**Capability boundary:** This document specifies an architecture and evaluation contract. It does not by itself establish deployed cognition, autonomy, clinical validity, superhuman performance, or beyond-SOTA empirical results.

---

## Executive Summary

The **Dr Moagi Cognitive Engine (DMCE)** is a hybrid neuro-symbolic research architecture for spatial reasoning, relational inference, multimodal state estimation, pattern synthesis, and bounded closed-loop adaptation.

It combines four computational regimes:

1. **Continuous representation learning** for compressing high-dimensional multimodal state into latent variables.
2. **Spatial and relational graph reasoning** for representing entities, dependencies, causal hypotheses, and constraints explicitly.
3. **Recursive latent refinement** for iteratively reducing prediction/reconstruction error under bounded fixed-point criteria.
4. **Candidate-first verification and rollback** so learned or inferred states remain provisional until they pass evidence, admissibility, and resource gates.

The engine is therefore not a single neural network. It is a typed processing system whose candidate state evolves through

```text
multimodal input
 -> encode
 -> compact / preserve residuals
 -> fuse
 -> relational graph synthesis
 -> recursive latent refinement
 -> decode / predict
 -> contrast against evidence
 -> stage memory / model / runtime updates
 -> verify
 -> COMMIT or ROLLBACK
 -> recur
```

This specification extends the canonical Dr Moagi operational auto-encoding/decoding law rather than replacing it.

---

# 1. Conceptual and Theoretical Foundations

## 1.1 Functional interpretation of cognition

Within Jarvis-X, cognition is treated operationally: a software system may be described as behaving cognitively when its state transitions reliably implement context-sensitive perception, representation, inference, prediction, error correction, planning, adaptation, and goal-directed control under uncertainty.

The term **cognitive engine** therefore denotes a systems abstraction over measurable software behavior. It does not imply a separate substance, consciousness, sentience, or capability unsupported by tests.

## 1.2 High-dimensional latent compression

Let the observed multimodal state at time `t` be

\[
X_t = \{X_t^{text},X_t^{audio},X_t^{image},X_t^{video},X_t^{code},X_t^{sensor},\ldots\}.
\]

For a spatial tensor modality,

\[
X_t^{(m)} \in \mathbb{R}^{C_m\times D_m\times H_m\times W_m}.
\]

Each modality is encoded by a declared encoder

\[
Z_t^{(m)} = E_{m,\Theta}(X_t^{(m)}),
\]

then projected into a shared representational manifold

\[
H_t^{(m)} = P_{m,\Theta} Z_t^{(m)}.
\]

Cross-modal fusion produces

\[
\boxed{
Z_t^{(0)} = \Phi_{fusion,\Theta}(H_t^{(1)},\ldots,H_t^{(M)})
}
\]

where `Z` is compact relative to the raw observation space.

Compression is not assumed to be lossless. Structure not retained in the bottleneck must be represented through explicit residuals, side information, memory, or declared distortion.

## 1.3 Relational and epistemic graph synthesis

Latent state alone does not encode an auditable relational model. The engine therefore constructs a graph

\[
G_t=(V_t,E_t,A_t)
\]

where:

- `V_t` contains entities, regions, concepts, hypotheses, or micro-states;
- `E_t` contains typed relations;
- `A_t` stores dynamic affinities, causal weights, constraints, or evidence strengths.

For node embeddings `h_i`, an attention-derived relation may be

\[
A_{ij}
=\operatorname{softmax}_j\left(
\frac{q_i^T k_j}{\sqrt d}+b_{ij}^{rel}
\right),
\]

with

\[
q_i=W_Qh_i,\qquad k_j=W_Kh_j.
\]

A graph update is then

\[
\boxed{
h_i^{(\ell+1)}
=\operatorname{LN}\left[
h_i^{(\ell)}+
F_\Theta\left(
\sum_{j\in\mathcal N(i)}A_{ij}W_Vh_j^{(\ell)}
\right)
\right]
}
\]

subject to declared relation types, admissibility masks, and domain constraints.

The graph is not treated as ground truth merely because the model generated it. Claims, predictions, observations, counterevidence, source identity, and confidence are routed into the Contrasting Tabularised Reckoner (`R_CTR`) before promotion.

## 1.4 Closed-loop recurrent metacognition

The engine recursively compares decoded predictions against observation and uses the residual to refine the next candidate state.

Given

\[
\hat X_t = D_\Theta(Z_t^*)
\]

and

\[
e_t = X_t-\hat X_t,
\]

a bounded latent refinement may use

\[
\boxed{
Z_t^{(k+1)}
=F_\Theta(Z_t^{(k)},\Omega_t,G_t,C_t,e_t)
}
\]

until either

\[
\frac{\|Z_t^{(k+1)}-Z_t^{(k)}\|_2}
{\|Z_t^{(k)}\|_2+\varepsilon}<\tau_Z
\]

or a declared iteration/resource ceiling is reached.

The result is a **candidate fixed point**, not an automatically authoritative state.

---

# 2. System Architecture Overview

The Dr Moagi Cognitive Engine is organized as seven logical modules.

```text
┌──────────────────────────────────────────────────────────────┐
│ 1. MULTIMODAL INGEST                                        │
│ text | image | video | audio | code | sensors | structured  │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ 2. ENCODING + MULTIRESOLUTION COMPACTION                    │
│ modality encoders | sparse tiles | residual hierarchy       │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ 3. SHARED LATENT + CROSS-ATTENTION FUSION                   │
│ Z(0) | topology | semantic alignment | uncertainty          │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ 4. RELATIONAL / EPISTEMIC GRAPH ENGINE                      │
│ entities | relations | hypotheses | constraints | evidence  │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ 5. RECURSIVE LATENT REASONER                                │
│ fixed-point iteration | Ω memory | attention | candidate set│
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ 6. DECODER / PREDICTOR / ACTION PROPOSER                    │
│ reconstruction | forecast | plan | rendered intervention    │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ 7. CTR + VERIFY + TRANSACTION                               │
│ contrast | evidence | constraints | COMMIT / ROLLBACK       │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               └───────────────→ recur
```

---

# 3. Canonical Typed Cognitive State

Define the engine state as

\[
\boxed{
C_t=
[X_t,\mathcal H_t,\mathcal R_t,Z_t,Z_t^*,G_t,\hat X_t,e_t,
\Omega_t,\Theta_t,\Pi_{run,t},A_{arch,t},R_{CTR,t},L_t]
}
\]

where:

| Symbol | Meaning |
|---|---|
| `X_t` | current observed/reference state |
| `H_t` | multiresolution representation hierarchy |
| `R_t` | residual hierarchy / side information |
| `Z_t` | encoded latent state |
| `Z_t*` | bounded recursive latent fixed point |
| `G_t` | relational / epistemic graph |
| `X_hat_t` | decoded reconstruction or prediction |
| `e_t` | residual/error field |
| `Omega_t` | adaptive temporal memory |
| `Theta_t` | model parameters |
| `Pi_run,t` | bounded runtime policy |
| `A_arch,t` | slower architectural/orchestration policy |
| `R_CTR,t` | contrast/evidence/reckoning state |
| `L_t` | audit and lineage state |

This typed separation is mandatory. Latent state, graph state, prediction state, evidence state, runtime policy, and authoritative world state are not interchangeable merely because they can all be represented numerically.

---

# 4. Core Mathematical Formulation

## 4.1 Neural latent encoding and decoding

For a flattened or tokenized representation `x_t`,

\[
z_t = E_\Theta(x_t),
\]

\[
\hat x_t = D_\Theta(z_t).
\]

For a deterministic autoencoder, the reconstruction residual is

\[
\boxed{e_t=x_t-\hat x_t.}
\]

For denoising training, corrupt the observation using a declared corruption process

\[
\tilde x_t = \mathcal C_\sigma(x_t),
\]

then reconstruct the clean target

\[
\hat x_t=D_\Theta(E_\Theta(\tilde x_t)).
\]

## 4.2 Spatial multi-resolution representation

For a volumetric hierarchy,

\[
H_t^{(0)}=X_t
\]

and

\[
H_t^{(\ell+1)}
=E_\Theta^{(\ell)}(H_t^{(\ell)};\mathcal A_t^{(\ell)}).
\]

A corresponding residual is

\[
\mathcal R_t^{(\ell)}
=H_t^{(\ell)}-
D_\Theta^{(\ell)}(H_t^{(\ell+1)}).
\]

The compact representation is therefore

\[
\mathcal Z_t=
[H_t^{(L)},\mathcal R_t^{(L-1)},\ldots,\mathcal R_t^{(0)}].
\]

This preserves the Jarvis-X rule that large logical spaces must remain distinct from physically resident working sets.

## 4.3 Relational graph operator

Let

\[
G_t=\mathcal G_\Theta(Z_t^*,X_t,\Omega_t).
\]

Graph reasoning produces a context update

\[
\tilde Z_t=
\mathcal R_G(Z_t^*,G_t).
\]

The graph operator may include message passing, typed constraints, causal hypothesis scoring, symbolic predicates, or differentiable logic, provided the representation contract is explicit.

## 4.4 Cognitive reconstruction objective

A general training/evaluation objective is

\[
\boxed{
\mathcal L_{cog}
=\lambda_{rec}\mathcal L_{rec}
+\lambda_{den}\mathcal L_{den}
+\lambda_{lat}\mathcal L_{lat}
+\lambda_{graph}\mathcal L_{graph}
+\lambda_{fp}\mathcal L_{fp}
+\lambda_{ctr}\mathcal L_{ctr}
+\lambda_{ctrl}\mathcal L_{ctrl}
}
\]

with terms such as

\[
\mathcal L_{rec}=\|x_t-\hat x_t\|_2^2,
\]

\[
\mathcal L_{den}=\|x_t-D_\Theta(E_\Theta(\mathcal C_\sigma(x_t)))\|_2^2,
\]

\[
\mathcal L_{lat}=\|z_t\|_2^2
\]

or another explicitly declared latent prior/regularizer,

\[
\mathcal L_{fp}=\|Z_t^*-F_\Theta(Z_t^*,\Omega_t,G_t,C_t)\|_2^2,
\]

and an evidence-consistency term

\[
\mathcal L_{ctr}=d(\hat X_t,X_t;R_{CTR,t}).
\]

`L_ctrl` represents bounded compute, policy, safety, sparsity, or resource penalties.

## 4.5 Gradient update with momentum

For trainable parameter vector `Theta`, define

\[
v_{t+1}=\mu v_t+\nabla_\Theta \mathcal L_{cog}(\Theta_t),
\]

\[
\boxed{
\Theta_{t+1}^{cand}=\Theta_t-\eta v_{t+1}.
}
\]

This remains a candidate parameter update. It becomes active only after the declared shadow-evaluation and promotion gates pass.

---

# 5. Canonical End-to-End Cognitive Operator

The cognitive engine is integrated into the existing Dr Moagi transaction law as

\[
\boxed{
C_{t+1}^{cand}
=
\Big[
\mathcal U_{\Omega,\Theta,\Pi}
\circ
\mathcal R_{CTR}
\circ
\mathcal D_{\mathcal R}
\circ
\mathcal R_G
\circ
\mathcal G
\circ
\operatorname{Fix}_{F_\Theta}
\circ
\Phi_{fusion}
\circ
\mathcal C_{exp}
\circ
\mathcal E
\Big](C_t,U_{t+1})
}
\]

where:

- `E` performs typed multimodal encoding;
- `C_exp` performs sparse/multiresolution compaction;
- `Phi_fusion` forms the shared latent manifold;
- `Fix_FTheta` performs bounded recursive latent refinement;
- `G` constructs the relational/epistemic graph;
- `R_G` performs graph-conditioned reasoning;
- `D_R` performs residual-aware selective decoding;
- `R_CTR` contrasts prediction against evidence and counterevidence;
- `U_Omega,Theta,Pi` stages memory, model, and runtime-policy adaptation.

The authoritative promotion law remains

\[
\boxed{
C_{t+1}=
V_t\,\Pi_\Lambda(C_{t+1}^{cand})
+(1-V_t)C_t
}
\]

interpreted structurally, with `V_t` equal to the conjunction of the relevant verification gates.

Operationally:

```text
candidate != authoritative
```

until the candidate passes verification.

---

# 6. Key Functional Modules

## 6.1 Spatial-temporal pattern memory

`Omega_t` stores bounded temporal information such as residual trends, novelty estimates, uncertainty statistics, recent latent trajectories, or domain-specific state summaries.

A generic update is

\[
\boxed{
\Omega_{t+1}^{cand}
=\rho\Omega_t+(1-\rho)\,\Gamma(e_t,Z_t^*,G_t)
}
\]

with explicit capacity and retention rules.

For spatial domains, state may be represented through active 3D tiles, octrees, factorized grids, or graph embeddings. Dense allocation must not be inferred from logical coordinate extent.

## 6.2 Dynamic denoising

Robustness is evaluated by injecting controlled corruption

\[
\tilde X_t=\mathcal C_{\sigma,p}(X_t)
\]

where the corruption contract may include Gaussian noise, masking, dropout, missing tiles, bit flips, occlusion, temporal packet loss, or modality dropout.

The system must report reconstruction error as a function of corruption level rather than only a single aggregate score.

## 6.3 Relational logic engine

The relational engine maps latent representations to explicit contextual structures and evaluates constraints over them.

Examples include:

- anatomical adjacency and trajectory constraints;
- object occupancy and collision relations;
- causal dependency graphs;
- policy/resource constraints;
- code dependency graphs;
- hypothesis/evidence/counterevidence tables.

Symbolic outputs remain proposals unless their semantics are validated against the relevant domain source.

## 6.4 Recurrent feedback loop

The decoder or predictor feeds the next contrast stage:

\[
Z_t^*\rightarrow \hat X_t\rightarrow e_t\rightarrow R_{CTR,t}\rightarrow
(\Omega,\Theta,\Pi)_{t+1}^{cand}.
\]

The next cycle then re-encodes new observation and committed state. This makes the loop recurrent without permitting unrestricted self-modification.

## 6.5 Attention and active support

For region `i`, define a computational salience score

\[
s_i(t)=
\alpha_e\epsilon_i
+\alpha_h H_i
+\alpha_u U_i
+\alpha_n N_i
+\alpha_g G_i
+\alpha_a A_i.
\]

Possible terms are reconstruction error, entropy/information, uncertainty, novelty, graph dependency, and task salience.

The active set is

\[
\boxed{
\mathcal A_{t+1}=\{i:s_i(t)>\tau\}.
}
\]

The engine therefore spends additional computation where error, uncertainty, information density, or dependency structure justifies it.

---

# 7. Metacognitive and Epistemic Verification

The engine must distinguish:

1. **internal convergence** — the latent loop has stabilized numerically;
2. **external correspondence** — the decoded prediction agrees with independently grounded evidence;
3. **transactional admissibility** — the proposed update is safe, typed, bounded, and policy-compliant.

A minimal acceptance conjunction is

\[
\boxed{
V_t=
V_{type}\land
V_{resource}\land
V_{fp}\land
V_{reconstruction}\land
V_{evidence}\land
V_{policy}\land
V_{integrity}
}
\]

where applicable.

A fixed point is insufficient evidence of truth. A model can converge stably to a wrong representation; therefore `R_CTR` must compare candidate conclusions with observations, counterexamples, competing hypotheses, and source provenance.

---

# 8. Primary Application Domains

## 8.1 Medical and diagnostic spatial modeling

Potential research uses include multimodal imaging representation, anatomical topology modeling, diagnostic prioritization, simulation support, and surgical trajectory planning.

Clinical use requires independent validation, calibrated uncertainty, traceable evidence sources, domain-specific safety review, and human clinical oversight. Research reconstruction accuracy alone is not clinical validation.

## 8.2 Autonomous system planning

The engine can model occupancy, object relations, trajectory candidates, residual prediction error, and bounded replanning.

A deployed autonomous controller must preserve hard safety constraints outside the learned latent representation and must fail safely when confidence or observability is inadequate.

## 8.3 Complex relational decision systems

Potential applications include resource routing, infrastructure optimization, multi-agent coordination, policy simulation, and socio-technical dependency modeling.

Outputs must distinguish model assumptions from empirical observations and should expose competing hypotheses where causal identification is uncertain.

---

# 9. Evaluation and Benchmark Contract

A DMCE implementation should be benchmarked across at least the following axes.

| Axis | Required measurement |
|---|---|
| Reconstruction | MSE/MAE/PSNR or domain-appropriate distortion |
| Denoising | error-vs-corruption curve |
| Latent stability | fixed-point residual and iteration count |
| Relational accuracy | edge/type/hypothesis precision and calibration |
| Evidence consistency | CTR-supported agreement with grounded observations |
| Robustness | missing-modality, occlusion, perturbation, and distribution-shift tests |
| Sparsity | active support / logical state-space ratio |
| Resource use | resident memory, wall time, FLOPs/ops estimate, device profile |
| Determinism | replay consistency under declared deterministic mode |
| Adaptation safety | shadow candidate acceptance/rejection statistics |
| Rollback integrity | proof that rejected candidates do not alter authoritative state |
| Calibration | confidence vs empirical correctness |

Any claim of superiority over another architecture must identify the dataset, task, baseline, hardware, metric, statistical protocol, and confidence interval.

---

# 10. Implementation Mapping

A reference implementation should preserve the following interfaces:

```text
Encoder.encode(input) -> EncodedState
Compactor.contract(encoded, active_set) -> Hierarchy + Residuals
Fusion.fuse(modal_states) -> LatentState
FixedPoint.refine(latent, memory, context) -> LatentCandidate
GraphBuilder.build(latent, observation, memory) -> RelationGraph
GraphReasoner.step(latent, graph) -> ReasonedLatent
Decoder.decode(reasoned_latent, residuals) -> Prediction
CTR.evaluate(observation, prediction, graph, evidence) -> EvidenceState
Adaptation.stage(...) -> CandidateUpdates
Verifier.verify(candidate, evidence, limits) -> VerificationReceipt
Transaction.commit_or_rollback(receipt) -> AuthoritativeState
```

All authoritative transitions should emit lineage sufficient to reconstruct why a candidate was accepted or rejected.

---

# 11. Roadmap

- [ ] Implement high-order tensor autoencoders for temporal sequences.
- [ ] Integrate transformer cross-attention between modality-specific latent spaces.
- [ ] Implement typed relational graph construction with explicit edge semantics.
- [ ] Couple graph reasoning to the canonical `R_CTR` evidence state.
- [ ] Add denoising benchmarks over controlled entropy/bit corruption levels.
- [ ] Add missing-modality and partial-observation recovery tests.
- [ ] Implement bounded latent fixed-point receipts and spectral/Jacobian diagnostics.
- [ ] Add shadow parameter adaptation with explicit rollback.
- [ ] Benchmark sparse active-support scheduling against dense baselines.
- [ ] Add calibrated uncertainty and abstention behavior.
- [ ] Define medical-imaging research adapters with explicit non-clinical capability boundaries.
- [ ] Define autonomous-planning adapters with external hard-safety constraints.
- [ ] Add reproducible benchmark manifests containing dataset, hardware, seeds, metrics, and confidence intervals.

---

# 12. Canonical Engineering Invariants

The Dr Moagi Cognitive Engine inherits the Jarvis-X architecture rules and adds the following cognitive-system-specific invariants:

1. **Representation is not reality.** Latent, graph, and decoded states are internal models until externally checked.
2. **Convergence is not truth.** Stable recurrence does not establish empirical correctness.
3. **Attention is allocation, not authority.** High salience changes compute priority, not epistemic status.
4. **Graph edges require semantics.** A numerical affinity does not automatically imply causation.
5. **Compression must expose distortion.** Information discarded from the bottleneck must be measured or represented through residuals/side information.
6. **Adaptation is candidate-first.** Learned updates are staged, evaluated, and promoted only after verification.
7. **Memory is bounded and typed.** Temporal memory must have explicit capacity, update, retention, and reset semantics.
8. **Uncertainty must remain visible.** The engine must not silently collapse ambiguity into a single asserted answer.
9. **Virtual scale is not physical throughput.** Logical dimensionality and address-space extent must remain distinct from resident memory and measured compute.
10. **Domain claims require domain evidence.** Medical, scientific, autonomous, financial, legal, or policy use requires evaluation specific to that domain.

---

# 13. Compact System Law

The complete cognitive cycle can be summarized as

\[
\boxed{
\begin{aligned}
&X_{t+1}\xrightarrow{\mathcal E}\mathcal Z
\xrightarrow{\mathcal C_{exp}}\mathcal H,\mathcal R
\xrightarrow{\Phi_{fusion}}Z^{(0)}
\xrightarrow{\operatorname{Fix}_{F_\Theta}}Z^*\\
&\xrightarrow{\mathcal G}G
\xrightarrow{\mathcal R_G}\tilde Z
\xrightarrow{\mathcal D_{\mathcal R}}\hat X
\xrightarrow{\mathcal R_{CTR}}(e,E,C,confidence)\\
&\xrightarrow{\mathcal U_{\Omega,\Theta,\Pi}}
C_{t+1}^{cand}
\xrightarrow{\Pi_\Lambda,V_t}
\{\text{COMMIT},\text{ROLLBACK}\}
\xrightarrow{}\text{recur}.
\end{aligned}
}
\]

This is the architectural closure of the Dr Moagi Cognitive Engine: **encode, relate, reason, reconstruct, contrast, verify, correct, and recur — without allowing an internal candidate to become authoritative merely because the system generated it.**


---

# 14. Computational Geometry Verification Extension

ADR-028 adds an optional, non-authoritative geometry receipt to the cognitive
verification path.

For a three-axis latent chart, define a software metric (g^{(Z)}), a symmetric
curvature proxy (mathcal G^{(Z)}), and an informational source tensor
(mathcal T^{(mathrm{info})}). The local computational residual is

[
oxed{
mathcal R^{(Z)}
=
mathcal G^{(Z)}
+
Lambda_Z g^{(Z)}
-
kappa_Zmathcal T^{(mathrm{info})}
}
]

with software-defined coupling parameters.

For a closed latent transport loop (P_1,ldots,P_n),

[
z' = P_ncdots P_1z,
qquad
h_Z=|z'-z|_2.
]

The geometry verifier emits

[
R_{m geom}
=
[
|mathcal R^{(Z)}|_F,
h_Z,
e_{m recon},
delta_{m fp},
A_Z
],
]

where (A_Z) is a combined telemetry norm. The receipt is accepted only when
each declared residual lies inside its own bound.

The cognitive verification path may therefore be extended as:

```text
latent candidate
 -> reconstruction / fixed-point evidence
 -> computational geometry receipt
 -> CTR evidence
 -> Pi_Lambda
 -> COMMIT or ROLLBACK
```

The geometry receipt is evidence only. It does not replace external-world
correspondence, domain validation, or the canonical transaction boundary.

The terms metric, curvature, source tensor and holonomy are computational
definitions. They are inspired by differential geometry but are not claims that
the cognitive engine obeys general relativity or that its internal tensors are
physical spacetime quantities.

NEXUS-3D, defined by ADR-028, is the associated GUI/media projection surface.
It may visualize multimodal outputs and measured browser/media telemetry, but it
is not an alternate authoritative cognitive-state channel.

## Recursive predictive state-space closure

The compact state-space closure for the cognitive loop is specified in
[`docs/architecture/recursive-predictive-3d-state-space.md`](./architecture/recursive-predictive-3d-state-space.md).

Its observable recurrence is

\[
S_{t+1}=\mathcal M_\Theta(S_t,X_t),
\qquad
S_t=[X_t,Z_t,\Omega_t,\hat X_t,E_t,\Pi_t],
\]

with

\[
E_t=X_t-\hat X_t.
\]

This does not replace the richer cognitive state or CTR transaction law. It is the
minimal interface joining encoder, bounded latent refinement, memory, decoder,
residual contrast, corrective policy and verification-gated recurrence. Candidate
memory, model and policy updates remain non-authoritative until admitted by the
existing verification boundary.
\n## Intrinsic geometric feedback interface

The recursive predictive closure admits a coordinate-free specialization on
Riemannian manifolds \((\mathcal X,g_{\mathcal X})\) and
\((\mathcal Z,g_t)\). The bounded cognitive correction path is

\[
Z_t
\xrightarrow{\varphi_T}
Z_t^\star
\xrightarrow{D_\phi}
\hat X_t
\xrightarrow{\log_{\hat X_t}(X_t)}
e_t
\xrightarrow{(dD_\phi)^\dagger}
\Pi_t
\xrightarrow{\operatorname{Exp}\ {\rm or}\ R}
Z_{t+1}^{\rm cand}.
\]

This gives CTR a typed geometric residual rather than assuming that all
observation errors are globally Euclidean vectors. If tangent-valued memory is
combined across different base points, the implementation must declare parallel
transport or another valid transport rule.

Optional metric adaptation

\[
\partial_t g_t=-2\operatorname{Ric}(g_t)+\mathcal S_t
\]

is candidate state and therefore remains subject to the same candidate-first
verification boundary as model, memory and policy updates. Geometric convergence
does not replace external evidence correspondence.
