# Recursive Predictive 3D State-Space Permeation

## Status

Repository-wide architectural integration note for the Jarvis-X / Dr Moagi recursive 3D runtime.

This document defines the system as a **closed-loop recurrent state-space inference architecture** with a Euclidean reference specialization and an intrinsic Riemannian formulation.

It complements:

- \`docs/DR_MOAGI_3D_OPERATIONAL_KINETIC_VISUALISATION.md\`;
- \`docs/CANONICAL_3D_GEOMETRIC_INTELLIGENCE.md\`;
- \`docs/DR_MOAGI_COGNITIVE_ENGINE.md\`;
- the transactional \`PERMEATE\` semantics used by the volumetric bytecode and verification layers.

The geometric representation is computational. Geometric contraction is not, by itself, information compression; a rendered vortex is not, by itself, measured throughput; and fixed-point convergence is not, by itself, empirical correctness.

---

## 1. Canonical compact state

For active coordinate \(\mathbf r=(x,y,z)\), define

\[
\boxed{
S_t(\mathbf r)
=
\left[
X_t,\,
Z_t,\,
\Omega_t,\,
\hat X_t,\,
E_t,\,
\Pi_t
\right](\mathbf r)
}
\]

where:

- \(X_t\): observed / ingested state;
- \(Z_t\): encoded latent state;
- \(\Omega_t\): bounded temporal / residual memory;
- \(\hat X_t\): decoded reconstruction or prediction;
- \(E_t\): externally anchored residual;
- \(\Pi_t\): runtime policy / corrective control state.

The complete recurrence is

\[
\boxed{
S_{t+1}=\mathcal M_\Theta(S_t,X_t)
}
\]

with candidate-first semantics: parameter, memory, metric, and policy proposals remain non-authoritative until verification.

---

## 2. Operational grammar

The system-level invariant is

\[
\boxed{
\text{Observe}
\rightarrow
\text{Encode}
\rightarrow
\text{Integrate}
\rightarrow
\text{Infer}
\rightarrow
\text{Decode}
\rightarrow
\text{Contrast}
\rightarrow
\text{Correct}
\rightarrow
\text{Verify}
\rightarrow
\text{Recur}
}
\]

A concrete cycle is

\`\`\`text
X_t
 -> E_theta(X_t)
 -> Z_t
 -> memory integration
 -> bounded latent refinement / flow
 -> Z_t*
 -> D_phi(Z_t*, Omega_t)
 -> Xhat_t
 -> residual / geometric logarithm
 -> tangent correction
 -> candidate state
 -> verification gate
 -> commit | rollback
 -> S_(t+1)
\`\`\`

---

# Part I — Euclidean reference specialization

## 3. Information transform and visualization transform are distinct

The encoder is an information map

\[
E_\theta:X\rightarrow Z
\]

while a 3D display transform may be

\[
\Phi_{\mathrm{viz}}:\mathbb R^3\rightarrow\mathbb R^3.
\]

They may be coupled, but they are not the same operator.

A simple Euclidean inward contraction around a display core \(\mathbf c\) is

\[
\Phi_\lambda(\mathbf r)
=
\mathbf c+\lambda(\mathbf r-\mathbf c),
\qquad
0<\lambda<1.
\]

A rotational-contractive display field is

\[
\dot{\mathbf r}
=
-\kappa(\mathbf r-\mathbf c)
+
\boldsymbol\omega\times(\mathbf r-\mathbf c)
+
\mathbf v_{\mathrm{res}}(E_t).
\]

This remains a visualization / coordinate projection unless it is explicitly tied to an intrinsic latent-space operator.

---

## 4. Euclidean fixed-point refinement

A bounded latent recurrence is

\[
Z^{(k+1)}
=
F_\Theta
\left(
Z^{(k)},
\Omega_t,
X_t
\right).
\]

Stop when

\[
\boxed{
\left\|Z^{(k+1)}-Z^{(k)}\right\|<\varepsilon_{\mathrm{fp}}
}
\]

or when the configured resource / iteration budget is exhausted.

An equilibrium satisfies

\[
\boxed{
Z^\star=F_\Theta(Z^\star,\Omega_t,X_t).
}
\]

A local sufficient stability condition is

\[
\boxed{
\rho\!\left(J_{F_\Theta}(Z^\star)\right)<1.
}
\]

---

# Part II — Intrinsic / Riemannian formulation

## 5. Geometric setting

Let the observation and latent spaces be Riemannian manifolds

\[
(\mathcal X,g_{\mathcal X}),
\qquad
(\mathcal Z,g_{\mathcal Z}).
\]

The latent metric may be time dependent:

\[
g_{\mathcal Z}=g_t.
\]

If the ambient observation space is Euclidean but the data occupy an intrinsic manifold \(\mathcal M\subset\mathcal X\), the geometric encoder should be interpreted on \(\mathcal M\).

If

\[
\dim \mathcal Z < \dim \mathcal X,
\]

the encoder cannot generally be an immersion of all of \(\mathcal X\) into \(\mathcal Z\). The appropriate object is a smooth dimensionality-reducing map

\[
E_\theta:\mathcal X\rightarrow\mathcal Z,
\]

or, when

\[
\dim\mathcal M\le\dim\mathcal Z,
\]

an immersion / embedding of the intrinsic data manifold:

\[
\boxed{
E_\theta|_{\mathcal M}:
(\mathcal M,g_{\mathcal M})
\rightarrow
(\mathcal Z,g_{\mathcal Z}).
}
\]

A controlled distortion target is

\[
(1-\varepsilon)
d_{\mathcal M}(x,x')
\le
d_{\mathcal Z}(E_\theta x,E_\theta x')
\le
(1+\varepsilon)
d_{\mathcal M}(x,x').
\]

For a stochastic encoder, \(q_\theta(\cdot\mid x)\) is a probability measure or density on \(\mathcal Z\) relative to the Riemannian volume measure.

---

## 6. Inward geometric flow

Use separate notation for the scalar potential and the induced flow.

Let

\[
U_t:\mathcal Z\rightarrow\mathbb R
\]

be a smooth latent potential.

The first-order Riemannian gradient flow is

\[
\boxed{
\dot Z(\tau)
=
-\operatorname{grad}_{g_t}U_t(Z(\tau)),
\qquad
Z(0)=Z_t.
}
\]

Let

\[
\varphi_\tau:\mathcal Z\rightarrow\mathcal Z
\]

denote the corresponding flow map. After bounded flow time \(T\),

\[
\boxed{
Z_t^\star=\varphi_T(Z_t).
}
\]

For second-order latent mechanics, use the covariant acceleration

\[
\boxed{
\nabla_{\dot Z}\dot Z
+
\Gamma(\dot Z)
+
\operatorname{grad}_{g_t}U_t(Z)
=
F_{\mathrm{control}}.
}
\]

This is the intrinsic analogue of damped kinetic latent evolution.

---

## 7. Intrinsic latent core

Reserve \(\Omega_t\) for temporal / residual memory. Denote the geometric latent core by

\[
\mathcal C_t\subset\mathcal Z.
\]

Examples include a potential sublevel set

\[
\boxed{
\mathcal C_t
=
\left\{
z\in\mathcal Z:
U_t(z)\le c_t
\right\},
}
\]

or an approximate stationary region

\[
\boxed{
\mathcal C_t
=
\left\{
z:
\left\|
\operatorname{grad}_{g_t}U_t(z)
\right\|_{g_t}
\le\varepsilon_{\mathrm{core}}
\right\}.
}
\]

Thus "inward" means movement toward a low-energy / stable region in intrinsic geometry, not merely toward the center of a Cartesian drawing.

---

## 8. Density transport

If a latent density \(\rho_t\) is maintained, and \(V_t\) is the latent vector field, then

\[
\boxed{
\partial_t\rho_t
+
\operatorname{div}_{g_t}(\rho_t V_t)
=
0.
}
\]

For pure inward gradient flow,

\[
V_t=-\operatorname{grad}_{g_t}U_t.
\]

This gives a coordinate-free continuity law for latent mass / probability transport.

---

## 9. Optional metric evolution

The latent state and latent geometry may evolve on different timescales.

A Ricci-type metric flow is

\[
\boxed{
\partial_t g_t
=
-2\operatorname{Ric}(g_t)
+
\mathcal S_t,
}
\]

where \(\mathcal S_t\) contains declared normalization, conditioning, task, or stabilizing terms.

The coupled state is therefore

\[
\boxed{
(Z_t,g_t)\longmapsto(Z_{t+1},g_{t+1}).
}
\]

Metric evolution is optional and must not be inferred from a static 3D visualization.

---

## 10. Decoder

The decoder is

\[
\boxed{
D_\phi:
(\mathcal Z,g_{\mathcal Z})
\rightarrow
(\mathcal X,g_{\mathcal X}).
}
\]

The reconstruction is

\[
\boxed{
\hat X_t=D_\phi(Z_t^\star).
}
\]

The composition \(D_\phi\circ E_\theta\) should approximate the identity on the supported data manifold to the declared distortion tolerance.

---

## 11. Intrinsic residual

A geometrically useful residual is the logarithmic displacement from prediction to observation:

\[
\boxed{
e_t
=
\log_{\hat X_t}(X_t)
\in
T_{\hat X_t}\mathcal X.
}
\]

This convention makes \(e_t\) point from the reconstruction toward the observation.

The Riemannian logarithm is locally unique only inside an appropriate normal neighborhood and away from the cut locus. Implementations must either enforce such a domain, choose a declared branch, or use a local approximation.

In a Euclidean chart,

\[
e_t=X_t-\hat X_t.
\]

---

## 12. Pulling the residual back to latent space

The decoder differential is

\[
dD_\phi|_{Z_t^\star}:
T_{Z_t^\star}\mathcal Z
\rightarrow
T_{\hat X_t}\mathcal X.
\]

Its metric adjoint maps observation-space tangent vectors back to latent tangent space:

\[
(dD_\phi|_{Z_t^\star})^\dagger:
T_{\hat X_t}\mathcal X
\rightarrow
T_{Z_t^\star}\mathcal Z.
\]

A natural geometric correction is

\[
\boxed{
\Pi_t^{\mathrm{geom}}
=
(dD_\phi|_{Z_t^\star})^\dagger e_t.
}
\]

A learned controller may augment it:

\[
\boxed{
\Pi_t
=
(dD_\phi)^\dagger e_t
+
C_\psi(Z_t^\star,\Omega_t,e_t),
\qquad
\Pi_t\in T_{Z_t^\star}\mathcal Z.
}
\]

This makes the correction vector intrinsically well typed.

---

## 13. Reinjection by exponential map or retraction

The corrected latent state is

\[
\boxed{
Z_{t+1}^{\mathrm{cand}}
=
\operatorname{Exp}_{Z_t^\star}
(\alpha\Pi_t).
}
\]

For computational efficiency, a declared retraction

\[
R_Z:T_Z\mathcal Z\rightarrow\mathcal Z
\]

may replace the exact exponential map:

\[
\boxed{
Z_{t+1}^{\mathrm{cand}}
=
R_{Z_t^\star}(\alpha\Pi_t).
}
\]

The Euclidean specialization is ordinary addition.

---

## 14. Controlled intrinsic latent ODE

The closed latent dynamics can be written

\[
\boxed{
\dot Z
=
-\operatorname{grad}_{g_t}U_t(Z)
+
(dD_\phi|_Z)^\dagger
\log_{D_\phi(Z)}X
+
C_\psi(Z,\Omega,X).
}
\]

The three terms have distinct roles:

\[
\underbrace{-\operatorname{grad}_{g_t}U_t}_{\text{intrinsic inward organization}}
+
\underbrace{(dD_\phi)^\dagger\log_{D_\phi(Z)}X}_{\text{reality-anchored correction}}
+
\underbrace{C_\psi}_{\text{learned control / memory}}.
\]

With evolving memory and metric, the larger dynamical system is

\[
\boxed{
\begin{aligned}
\dot Z
&=
-\operatorname{grad}_{g_t}U_t(Z)
+
(dD_\phi)^\dagger
\log_{D_\phi(Z)}X
+
C_\psi(Z,\Omega,X),
\\
\dot\Omega
&=
\mathcal M_\Omega(Z,e,\Omega),
\\
\partial_t g_t
&=
-2\operatorname{Ric}(g_t)
+
\mathcal S_t.
\end{aligned}
}
\]

---

## 15. Geodesic contraction and stability

If the potential is geodesically strongly convex in the operating region,

\[
\boxed{
\operatorname{Hess}_{g}U
\succeq
m\,g,
\qquad m>0,
}
\]

then the gradient flow is locally contractive under the corresponding assumptions, with behavior of the form

\[
d_{\mathcal Z}(Z_1(t),Z_2(t))
\lesssim
e^{-mt}
d_{\mathcal Z}(Z_1(0),Z_2(0)).
\]

This gives the visual notion of inward contraction an intrinsic interpretation: **geodesic contraction**, not merely Cartesian shrinkage.

The production convergence gate still requires both internal stability and external reconstruction / evidence checks.

---

## 16. Intrinsic memory

A bounded memory candidate may be

\[
\boxed{
\Omega_{t+1}^{\mathrm{cand}}
=
\rho\Omega_t
+
(1-\rho)\Gamma(e_t,Z_t^\star),
\qquad
0\le\rho<1.
}
\]

If memory values live in tangent spaces attached to different latent points, an implementation must use declared parallel transport or another typed transport mechanism before combining them.

---

## 17. Intrinsic objective

A coordinate-aware research loss may include

\[
\boxed{
\begin{aligned}
\mathcal L
={}&
\lambda_{\mathrm{rec}}
\mathbb E
\left[
d_{\mathcal X}
(X,D_\phi(E_\theta X))^2
\right]
\\
&+
\lambda_{\mathrm{dist}}
\mathbb E_{x,x'}
\left[
d_{\mathcal Z}(E_\theta x,E_\theta x')
-
d_{\mathcal M}(x,x')
\right]^2
\\
&+
\lambda_{\mathrm{fp}}
\mathbb E
\left[
\left\|
\operatorname{grad}_{g}U(Z^\star)
\right\|_g^2
\right]
\\
&+
\lambda_{\mathrm{ctrl}}
\mathbb E
\left[
\|\Pi\|_g^2
\right]
+
\lambda_{\mathrm{metric}}\mathcal R(g_{\mathcal Z}).
\end{aligned}
}
\]

Possible metric regularizers include:

- \(\|\operatorname{Ric}(g)\|_g^2\);
- \((\operatorname{Scal}(g)-S_0)^2\);
- metric condition-number penalties;
- volume-distortion penalties;
- injectivity-radius constraints;
- sectional-curvature bounds.

Direct minimization of scalar curvature alone is not assumed to produce a well-conditioned latent geometry.

For a stochastic encoder, the intrinsic KL term is

\[
\boxed{
D_{\mathrm{KL}}(q\|p)
=
\int_{\mathcal Z}
q(z)\log\frac{q(z)}{p(z)}
\,d\operatorname{vol}_{g_{\mathcal Z}}(z).
}
\]

---

## 18. Candidate-first verification law

The candidate state may now include metric and latent updates:

\[
S_{t+1}^{\mathrm{cand}}
=
\left(
Z_{t+1}^{\mathrm{cand}},
\Omega_{t+1}^{\mathrm{cand}},
g_{t+1}^{\mathrm{cand}},
\Theta_{t+1}^{\mathrm{cand}},
\Pi_{t+1}^{\mathrm{cand}}
\right).
\]

The authoritative transition remains

\[
\boxed{
S_{t+1}
=
V_t\,S_{t+1}^{\mathrm{cand}}
+
(1-V_t)\,S_t
}
\]

interpreted structurally.

A geometric verification conjunction may include

\[
V_t
=
V_{\mathrm{finite}}
\land
V_{\mathrm{resource}}
\land
V_{\mathrm{metric}}
\land
V_{\mathrm{fp}}
\land
V_{\mathrm{reconstruction}}
\land
V_{\mathrm{evidence}}
\land
V_{\mathrm{integrity}}.
\]

The metric gate may check positive definiteness, conditioning, bounded curvature proxies, and declared coordinate-chart validity.

---

# Part III — Visualization contract

## 19. 3D visualization as runtime instrumentation

The visualization must be driven by measured state rather than independent animation constants.

| 3D quantity | Runtime quantity |
|---|---|
| inward radial motion | projection of intrinsic gradient-flow progress |
| outward radial motion | decoder / reconstruction depth |
| particle coordinates | chart or embedding of latent coordinates |
| particle displacement | residual / tangent correction magnitude |
| ring / shell phase | memory state \(\Omega_t\) |
| core radius | fixed-point / gradient residual |
| angular velocity | measured iteration or phase rate |
| particle density | active-support / density proxy |
| shell deformation | reconstruction stress / uncertainty |
| local metric ellipsoid | \(g_t\) conditioning / anisotropy |
| geodesic traces | intrinsic latent trajectories |
| color / opacity | declared telemetry channel only |

Synthetic values must be labeled synthetic. Measured rates, residuals, iteration counts, memory occupancy, metric condition numbers, and convergence statistics must originate from the runtime that produced the frame.

---

## 20. Hourglass projection

A Euclidean display embedding may use

\[
r(z)
=
r_{\min}
+
(r_{\max}-r_{\min})
\left|\frac{z}{L}\right|^p,
\qquad
-L\le z\le L.
\]

Then

\[
r(0)=r_{\min},
\qquad
r(\pm L)=r_{\max}.
\]

This is a display geometry only. In the intrinsic formulation, the true notion of "inward" is reduction of the chosen geometric energy and/or geodesic distance to the declared core region \(\mathcal C_t\).

---

# Part IV — Compact intrinsic system law

## 21. Closed geometric operator

A compact discrete cycle is

\[
\boxed{
\begin{aligned}
Z_t &= E_\theta(X_t),
\\
Z_t^\star &= \varphi_T(Z_t),
\\
\hat X_t &= D_\phi(Z_t^\star),
\\
e_t &= \log_{\hat X_t}(X_t),
\\
\Pi_t &=
(dD_\phi)^\dagger e_t
+
C_\psi(Z_t^\star,\Omega_t,e_t),
\\
Z_{t+1}^{\mathrm{cand}}
&=
\operatorname{Exp}_{Z_t^\star}(\alpha\Pi_t),
\\
\Omega_{t+1}^{\mathrm{cand}}
&=
\mathcal M_\Omega(\Omega_t,e_t,Z_t^\star),
\\
S_{t+1}
&=
\operatorname{VerifyAndCommit}
\left(
S_t,
S_{t+1}^{\mathrm{cand}}
\right).
\end{aligned}
}
\]

The architecture therefore closes on

\[
\boxed{
\text{Encode}
\rightarrow
\text{Flow}
\rightarrow
\text{Decode}
\rightarrow
\text{Log-residual}
\rightarrow
\text{Tangent correction}
\rightarrow
\text{Exp/Retraction}
\rightarrow
\text{Verify}
\rightarrow
\circlearrowleft
}
\]

with explicit separation between:

- information transformation;
- intrinsic manifold dynamics;
- geometric visualization;
- measured telemetry;
- memory;
- candidate adaptation;
- authoritative state.
