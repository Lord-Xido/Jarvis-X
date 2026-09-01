# DMSO TPM continuum-field extension

**Status:** Proposed research specification  
**Date:** 2026-08-04  
**Owner:** Dr. Matladi Moagi  
**Related decision:** [`ADR-001-DMSO-INWARD-SELF-OPTIMIZATION.md`](adr/ADR-001-DMSO-INWARD-SELF-OPTIMIZATION.md)  
**Implementation status:** Not yet implemented

## 1. Purpose

This document extends the Dr. Moagi System of Operations (DMSO) from a finite augmented-state optimizer into a spatial continuum model. It formalizes a field \(\Psi\) that is encoded, transformed in a latent hierarchy, decoded, corrected by a task objective and spatially regularized by diffusion.

The specification preserves the intended operating structure:

```text
field observation
  → encode
  → latent hierarchical transform
  → decode
  → compare against task and reconstruction objectives
  → diffuse spatial inconsistencies
  → project into Λ constraints
  → shadow verify
  → commit or rollback
  → journal in Ω
```

The equations below are a mathematical proposal. They do not by themselves establish convergence, physical validity, production safety or implemented capability.

## 2. Domain and typed operators

Let

\[
\Omega=[0,1000]^3
\]

be the spatial domain and let

\[
\Psi:\Omega\times[0,T]\rightarrow\mathbb{R}^{c}
\]

be a scalar or vector-valued operational field. The number `1000` defines the coordinate extent; it does **not** require a dense \(1000^3\) in-memory grid.

The operators are typed as follows:

\[
\mathcal E:\mathcal X\rightarrow\mathcal Z,
\qquad
\mathcal H:\mathcal Z\rightarrow\mathcal Z,
\qquad
\mathcal D:\mathcal Z\rightarrow\mathcal X,
\qquad
\mathcal R:\mathcal X\rightarrow\mathcal Y,
\]

where:

- \(\mathcal X\) is an admissible field space, such as \(H^1(\Omega;\mathbb R^c)\);
- \(\mathcal Z\) is the finite latent state space;
- \(\mathcal H\) is the bounded latent hierarchy or latent propagator;
- \(\mathcal R\) is the task readout;
- \(\mathcal L:\mathcal Y\rightarrow\mathbb R\) is the task loss.

Let \(P_A\) be the projection onto currently active and mutable field coordinates. The Laplacian \(\Delta=\nabla^2\) acts componentwise when \(c>1\).

A boundary condition must be selected explicitly. Canonical reference implementations should use either:

- periodic boundaries; or
- homogeneous Neumann/no-flux boundaries, \(\nabla\Psi\cdot n=0\) on \(\partial\Omega\).

## 3. Corrected continuum evolution law

The submitted field equation was

\[
\frac{\partial \Psi}{\partial t}
=
\mathcal D\circ\mathcal H\circ\mathcal E(\Psi)
-
\eta\nabla_\Psi\mathcal L(\mathcal R(\Psi))
+
\kappa\Delta\Psi.
\]

For dimensional and fixed-point consistency, the decoded latent transform must enter as a **residual rate**, not as an unqualified state value. The canonical DMSO-TPM continuum equation is therefore

\[
\boxed{
\frac{\partial\Psi}{\partial t}
=
\frac{1}{\tau_H}
\left[
\mathcal D\!\left(\mathcal H(\mathcal E(\Psi))\right)-\Psi
\right]
-
\eta P_A\nabla_\Psi\mathcal L\!\left(\mathcal R(\Psi)\right)
+
\kappa\Delta\Psi
}
\]

with

\[
\tau_H>0,\qquad \eta\ge0,\qquad \kappa\ge0.
\]

Define the complete field generator

\[
\mathcal F(\Psi)
=
\frac{1}{\tau_H}
\left[
\mathcal D\mathcal H\mathcal E(\Psi)-\Psi
\right]
-
\eta P_A\nabla_\Psi\mathcal L(\mathcal R(\Psi))
+
\kappa\Delta\Psi.
\]

Then

\[
\partial_t\Psi=\mathcal F(\Psi).
\]

### Interpretation

- \(\tau_H^{-1}[\mathcal D\mathcal H\mathcal E(\Psi)-\Psi]\) is latent hierarchical reconstruction and correction;
- \(-\eta P_A\nabla_\Psi\mathcal L\) is task-directed adaptation restricted to authorized coordinates;
- \(\kappa\Delta\Psi\) is spatial smoothing or transport regularization;
- later transactional projection by \(\Pi_\Lambda\) enforces numerical, policy and coherence constraints.

The gradient notation includes the chain rule through \(\mathcal R\). Equivalently,

\[
\nabla_\Psi\mathcal L(\mathcal R(\Psi))
=
J_{\mathcal R}(\Psi)^*\nabla_y\mathcal L(y)
\big|_{y=\mathcal R(\Psi)}.
\]

## 4. TPM reconstruction volume

Let the pure autoencoder residual be

\[
r_E(\Psi)=\Psi-\mathcal D(\mathcal E(\Psi)).
\]

The submitted TPM volume was an unnormalized integral. Because \(|\Omega|=1000^3\), its magnitude scales directly with domain volume and resolution conventions. The canonical metric is the volume-normalized squared reconstruction error

\[
\boxed{
\mathcal V_{\mathrm{TPM}}(\Psi)
=
\frac{1}{|\Omega|}
\iiint_{\Omega}
\left\|
\Psi(x,y,z)-\mathcal D(\mathcal E(\Psi))(x,y,z)
\right\|_2^2
\,dx\,dy\,dz
\le
\varepsilon_{\mathrm{rec}}^2
}
\]

where \(\varepsilon_{\mathrm{rec}}\) has the same units as \(\Psi\).

A dimensionless relative form is preferred when the field scale varies:

\[
\boxed{
\widehat{\mathcal V}_{\mathrm{TPM}}(\Psi)
=
\frac{
\|\Psi-\mathcal D\mathcal E(\Psi)\|_{L^2}^2
}{
\|\Psi\|_{L^2}^2+\varepsilon_0
}
\le
\widehat\varepsilon_{\mathrm{rec}}^2
}
\]

with \(\varepsilon_0>0\) preventing division by zero.

If the intended measurement is consistency through the full latent hierarchy, define separately

\[
\mathcal V_{H}(\Psi)
=
\frac{1}{|\Omega|}
\|\Psi-\mathcal D\mathcal H\mathcal E(\Psi)\|_{L^2}^2.
\]

The autoencoder metric \(\mathcal V_{\mathrm{TPM}}\) and the hierarchical metric \(\mathcal V_H\) must not be conflated.

## 5. Stationary or locked field

A deterministic fixed point \(\Psi^*\) satisfies

\[
\partial_t\Psi^*=0.
\]

Therefore the corrected stationary equation is

\[
\boxed{
\frac{1}{\tau_H}
\left[
\mathcal D\mathcal H\mathcal E(\Psi^*)-\Psi^*
\right]
-
\eta P_A\nabla_\Psi\mathcal L(\mathcal R(\Psi^*))
+
\kappa\Delta\Psi^*
=0
}
\]

subject to the selected boundary condition and all \(\Lambda\)-constraints.

This is a force-balance condition. It does not require each term to vanish independently. A stronger decomposed lock may additionally require

\[
\|\Psi^*-\mathcal D\mathcal E(\Psi^*)\|_{L^2}\le\delta_{\mathrm{rec}},
\]

\[
\|P_A\nabla_\Psi\mathcal L(\mathcal R(\Psi^*))\|_{L^2}\le\delta_{\mathrm{task}},
\]

and

\[
\|\Delta\Psi^*\|_{L^2}\le\delta_{\mathrm{spatial}}.
\]

## 6. Mild integral solution

For the Markovian diffusion equation in Section 3, the mathematically equivalent integral representation uses the diffusion semigroup generated by \(\kappa\Delta\).

Let

\[
\mathcal G(\Psi)
=
\frac{1}{\tau_H}
\left[
\mathcal D\mathcal H\mathcal E(\Psi)-\Psi
\right]
-
\eta P_A\nabla_\Psi\mathcal L(\mathcal R(\Psi)).
\]

Then the mild solution is

\[
\boxed{
\Psi(t)
=
e^{\kappa t\Delta}\Psi(0)
+
\int_0^t
 e^{\kappa(t-s)\Delta}
\mathcal G(\Psi(s))\,ds
}
\]

for boundary conditions under which the semigroup is well-defined.

This form is the Duhamel representation of

\[
\partial_t\Psi=\mathcal G(\Psi)+\kappa\Delta\Psi.
\]

A scalar temporal kernel

\[
e^{-(t-s)/\lambda}
\]

is not interchangeable with the spatial diffusion semigroup. Using that kernel defines a different model with explicit temporal memory.

## 7. Optional exponential-memory extension

When a genuine fading-memory spatial term is intended, introduce an auxiliary memory field \(M\):

\[
\boxed{
\begin{aligned}
\partial_t\Psi
&=
\mathcal G(\Psi)+\kappa M,\\
\lambda\partial_tM+M
&=
\Delta\Psi,
\qquad \lambda>0.
\end{aligned}
}
\]

For \(M(0)=0\),

\[
\boxed{
M(t)
=
\frac{1}{\lambda}
\int_0^t
 e^{-(t-s)/\lambda}
\Delta\Psi(s)\,ds
}
\]

so the exponential kernel has a precise operational meaning. This memory model must be implemented and validated separately from ordinary diffusion.

As \(\lambda\to0^+\), formally \(M\to\Delta\Psi\), recovering the Markovian diffusion term.

## 8. Geode convergence criterion

The submitted criterion combined an \(L^2\) reconstruction norm and an \(L^1\) task-gradient norm directly. Those terms generally have different units and domain scaling. The canonical criterion normalizes each residual before aggregation.

Define

\[
c_{\mathrm{rec}}
=
\frac{
\|\Psi-\mathcal D\mathcal E(\Psi)\|_{L^2}
}{
\|\Psi\|_{L^2}+\varepsilon_0
},
\]

\[
c_{\mathrm{task}}
=
\frac{
\eta\|P_A\nabla_\Psi\mathcal L(\mathcal R(\Psi))\|_{L^2}
}{
F_{\mathrm{ref}}+\varepsilon_0
},
\]

and

\[
c_{\mathrm{dyn}}
=
\frac{
\|\mathcal F(\Psi)\|_{L^2}
}{
F_{\mathrm{ref}}+\varepsilon_0
},
\]

where \(F_{\mathrm{ref}}>0\) is a declared force or update-rate scale.

The DMSO Geode score is

\[
\boxed{
\mathcal C_{\mathrm{Geode}}(\Psi)
=
w_r c_{\mathrm{rec}}
+w_g c_{\mathrm{task}}
+w_f c_{\mathrm{dyn}}
<10^{-6}
}
\]

with

\[
w_r,w_g,w_f\ge0,
\qquad
w_r+w_g+w_f=1.
\]

The threshold \(10^{-6}\) is valid only after normalization and must be justified against numerical precision, discretization error and application tolerance.

A lock requires the criterion to hold for \(K\) consecutive accepted steps, together with:

- no \(\Lambda\)-constraint violation;
- no NaN or infinity;
- valid journal-chain integrity;
- deterministic replay agreement;
- no unauthorized mutation;
- bounded field and parameter updates.

## 9. Discrete transactional realization

Let \(\Psi_k\) be the authoritative field at time step \(k\). A reference implementation should compute a candidate in shadow state.

### 9.1 Candidate generator

For a first-order integrator,

\[
\widetilde\Psi_{k+1}
=
\Psi_k
+
\Delta t\left[
\frac{\mathcal D\mathcal H\mathcal E(\Psi_k)-\Psi_k}{\tau_H}
-
\eta P_A\nabla_\Psi\mathcal L(\mathcal R(\Psi_k))
+
\kappa\Delta_h\Psi_k
\right].
\]

The projected candidate is

\[
\Psi^{\Lambda}_{k+1}
=
\Pi_{\Lambda_k}(\widetilde\Psi_{k+1}).
\]

### 9.2 Verification

The shadow verifier checks:

- shape, dtype and boundary conditions;
- finite values;
- maximum update norm;
- reconstruction and task metrics;
- conservation laws declared by the subsystem;
- monotonicity conditions when required;
- memory and execution budgets;
- deterministic digest generation.

The authoritative state transition is

\[
\boxed{
\Psi_{k+1}
=
\begin{cases}
\Psi^{\Lambda}_{k+1}, & \operatorname{Verify}=\mathrm{accept},\\
\Psi_k, & \operatorname{Verify}=\mathrm{reject}.
\end{cases}
}
\]

Every decision is appended to the \(\Omega\) journal with prior, candidate and resulting state digests.

## 10. Numerical stability boundary

A dense explicit grid is not required. Implementations may use sparse blocks, octrees, spectral coefficients or finite elements, provided that the logical domain and discretization contract are explicit.

For an explicit finite-difference diffusion step on a uniform three-dimensional grid with spacing \(h\), a conservative stability restriction is

\[
\Delta t
\le
\frac{h^2}{6\kappa}
\]

when \(\kappa>0\), before accounting for additional restrictions introduced by the nonlinear latent and task terms.

Reference implementations should prefer one of:

- semi-implicit diffusion;
- operator splitting;
- an exponential integrator;
- a spectral step for periodic boundaries;
- an adaptively controlled solver.

Solver tolerances, precision and maximum iteration counts must be recorded in the journal.

## 11. Operational pseudocode

```text
observe Ψ_k
validate field schema, boundary contract and active mask
z_k ← E(Ψ_k)
z'_k ← H(z_k)
Ψ̂_k ← D(z'_k)
g_k ← P_A ∇_Ψ L(R(Ψ_k))
d_k ← Δ_h Ψ_k

candidate ← Ψ_k + Δt[(Ψ̂_k - Ψ_k)/τ_H - ηg_k + κd_k]
candidate ← Π_Λ(candidate)

metrics ← {
  TPM reconstruction,
  hierarchical consistency,
  task gradient,
  dynamic residual,
  Geode score,
  update norm
}

if Verify(Ψ_k, candidate, metrics):
    Ψ_{k+1} ← candidate
    decision ← commit
else:
    Ψ_{k+1} ← Ψ_k
    decision ← rollback

append Ω journal entry
lock only after K consecutive valid convergence steps
```

## 12. Required validation

A reference implementation must include:

1. shape and operator type tests;
2. boundary-condition fixtures;
3. finite-difference or automatic-gradient checks;
4. diffusion-only analytic decay fixtures;
5. identity encoder/decoder fixtures;
6. fixed-point residual tests;
7. TPM normalization tests independent of grid resolution;
8. Markovian versus memory-model separation tests;
9. explicit-step stability rejection;
10. projection and rollback tests;
11. deterministic replay under fixed seeds;
12. sparse-memory accounting;
13. Geode lock and false-lock tests;
14. journal hash-chain verification;
15. preservation of canonical VM behavior while the continuum subsystem is disabled.

## 13. Canonical compact form

The proposed Markovian DMSO-TPM field law is

\[
\boxed{
\partial_t\Psi
=
\tau_H^{-1}(\mathcal D\mathcal H\mathcal E(\Psi)-\Psi)
-
\eta P_A\nabla_\Psi\mathcal L(\mathcal R(\Psi))
+
\kappa\Delta\Psi
}
\]

with reconstruction admissibility

\[
\boxed{
\widehat{\mathcal V}_{\mathrm{TPM}}
=
\frac{\|\Psi-\mathcal D\mathcal E(\Psi)\|_{L^2}^2}
{\|\Psi\|_{L^2}^2+\varepsilon_0}
\le
\widehat\varepsilon_{\mathrm{rec}}^2
}
\]

and fixed-point condition

\[
\boxed{
\mathcal F(\Psi^*)=0.
}
\]

The equivalent mild solution is

\[
\boxed{
\Psi(t)
=e^{\kappa t\Delta}\Psi(0)
+
\int_0^t e^{\kappa(t-s)\Delta}
\left[
\tau_H^{-1}(\mathcal D\mathcal H\mathcal E(\Psi(s))-\Psi(s))
-
\eta P_A\nabla_\Psi\mathcal L(\mathcal R(\Psi(s)))
\right]ds.
}
\]

The normalized lock criterion is

\[
\boxed{
\mathcal C_{\mathrm{Geode}}(\Psi)
=w_rc_{\mathrm{rec}}+w_gc_{\mathrm{task}}+w_fc_{\mathrm{dyn}}
<10^{-6}
}
\]

for \(K\) consecutive verified transitions.

This field extension remains subordinate to DMSO's finite recursion, explicit authority boundaries, \(\Lambda\)-projection, shadow verification, atomic commit or rollback and auditable \(\Omega\) journaling.