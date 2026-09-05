# Bounded 3D Auto-Optimization for the Virtual Bitstream AE/AD

## Scope

`jarvisx.dr_moagi_virtual_3d_ae` is a deterministic sparse 3D bitstream codec
laboratory. It does **not** claim a learned neural representation. The refinement
adds a bounded self-tuning layer around the existing encode-couple-decode-feedback
recurrence while preserving finite materialization, deterministic replay and an
explicit capability boundary.

## 1. Active 3D coupling

For latent bit \(j\) at lattice point \(p\), map the bit to a spin
\(s_j(p)\in\{-1,+1\}\). With six-neighbour mean

\[
m_j(p)=\frac{1}{|\mathcal N_6(p)|}\sum_{q\in\mathcal N_6(p)}s_j(q),
\]

the synchronous coupled bit is

\[
\widetilde z_j(p)=
\mathbf 1\left[(1-\alpha)s_j(p)+\alpha m_j(p)\ge 0\right].
\]

When \(\alpha<0.5\), even unanimous opposing neighbours cannot flip the current
spin, so the operator is inert. The reference default is therefore moved to
\(\alpha=0.65\), where sufficiently strong six-neighbour consensus can alter a
latent bit.

## 2. Measured objective

Each candidate is evaluated from the same deterministic source tile. The score is

\[
J =
w_r L_{\rm rec}
+w_s L_{\rm spatial}
+w_b L_{\rm balance}
+w_g \bar{\Delta}.
\]

The terms are:

- \(L_{\rm rec}\): normalized Hamming reconstruction loss against the deterministic
  source stream;
- \(L_{\rm spatial}\): mean latent Hamming disagreement across positive x/y/z
  lattice edges;
- \(L_{\rm balance}=1-4p(1-p)\), where \(p\) is the latent one-bit fraction. This is
  zero at 50/50 occupancy and one at all-zero or all-one collapse;
- \(\bar{\Delta}\): mean reality gap across the bounded recurrence.

Default weights are \(w_r=1.0\), \(w_s=0.25\), \(w_b=0.25\), \(w_g=0.10\).

## 3. Deterministic bounded search

`DrMoagiVirtual3DAE.optimize()` evaluates a finite Cartesian grid of
\((\alpha,\beta)\) pairs. The current pair is always included as the baseline.
Candidate ordering and tie-breaking are deterministic. A new pair is committed
only when

\[
J_{\rm candidate} < J_{\rm baseline}-10^{-12}.
\]

Otherwise the engine keeps the baseline configuration. Rebuilding after a commit
clears transient materialized state so the subsequent run starts from the same
coordinate-derived source field.

Default candidate sets are:

\[
\alpha\in\{0.55,0.65,0.80\},\qquad
\beta\in\{0.35,0.50,0.65,0.80\}.
\]

This is twelve bounded candidates plus the baseline only when it is not already in
the grid.

## 4. CLI

Run the normal reference:

```bash
python -m jarvisx.dr_moagi_virtual_3d_ae --json
```

Run bounded auto-optimization before the recurrence:

```bash
python -m jarvisx.dr_moagi_virtual_3d_ae --auto-optimize --json
```

The JSON summary records the chosen parameters, baseline and final scores,
candidate count, component losses and whether an improvement was committed.

## 5. Safety and interpretation boundary

The optimizer searches two explicit scalar controls; it does not rewrite its own
source code, train weights by gradient descent, allocate the symbolic logical
universe, or guarantee improved reconstruction in isolation. It optimizes the
declared multi-objective score under finite candidate, pass and tile bounds.
