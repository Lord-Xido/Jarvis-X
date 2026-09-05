# Dr. Moagi 3D Continuum Kernel

## Canonical state equation

The runtime kernel operationalizes

\[
\Psi(t)=\int_{\Omega}\left(\Phi_{\mathrm{in}}(\mathbf r,t)\otimes
\Lambda_{\mathrm{in}}^{-1}(\mathbf r,t)\right)e^{-\gamma t}\,d\Omega
+\Theta_{\mathrm{core}}.
\]

For the executable 3D reference implementation, the operator product is made explicit as

\[
\Lambda_v^{+}\Phi_v,
\]

where \(\Lambda_v^{+}\) is a regularized left pseudoinverse. This avoids an undefined literal inverse when a voxel transform is singular or ill-conditioned.

## Discrete 3D form

For active voxels \(v=1,\ldots,N\) with voxel volume \(\Delta V\),

\[
\boxed{
\Psi(t)=\Theta_{\mathrm{core}}+
 e^{-\gamma t}\Delta V
 \sum_{v=1}^{N}\Lambda_v^{+}\Phi_v
}
\]

with

\[
\Lambda_v^{+}
=
\left(\Lambda_v^T\Lambda_v+\epsilon I\right)^{-1}\Lambda_v^T,
\qquad \epsilon>0.
\]

The state is three-dimensional:

\[
\Phi_v,\Psi,\Theta_{\mathrm{core}}\in\mathbb R^3,
\qquad
\Lambda_v\in\mathbb R^{3\times3}.
\]

## Runtime map

The implementation lives in `src/jarvisx/moagi_continuum.py` and exposes:

- `ContinuumConfig`: \(\gamma\), \(\Delta V\), and regularization \(\epsilon\).
- `regularized_pseudoinverse`: stable \(3\times3\) inverse operator.
- `continuum_step`: one complete domain integration.
- `homogeneous_recurrence`: inward recursive state evolution.

No NumPy dependency is required by the core implementation.

## Inward recurrence

For recursive execution, the output becomes the next global state:

\[
\Psi_{k+1}=\mathcal M(\Psi_k).
\]

The reference homogeneous field broadcasts a voxel-count-normalized state,

\[
\Phi_{v,k}=\frac{\Psi_k}{N},
\]

therefore

\[
\boxed{
\Psi_{k+1}
=
\Theta_{\mathrm{core}}
+
e^{-\gamma\Delta t}\frac{\Delta V}{N}
\sum_{v=1}^{N}\Lambda_v^{+}\Psi_k
}
\]

and a fixed point satisfies

\[
\boxed{\Psi^*=\mathcal M(\Psi^*)}.
\]

The normalization prevents the recurrence gain from increasing solely because the active sparse tile contains more voxels.

## Stability condition

Define

\[
A=
e^{-\gamma\Delta t}\frac{\Delta V}{N}
\sum_{v=1}^{N}\Lambda_v^{+}.
\]

Then

\[
\Psi_{k+1}=A\Psi_k+\Theta_{\mathrm{core}}.
\]

A sufficient convergence condition is

\[
\rho(A)<1,
\]

where \(\rho(A)\) is the spectral radius. Under that condition,

\[
\Psi^*=(I-A)^{-1}\Theta_{\mathrm{core}}.
\]

This connects the continuum kernel directly to the Jarvis-X fixed-point integrity contract.

## Verification

`tests/test_moagi_continuum.py` verifies:

1. the single-voxel arithmetic form;
2. discrete voxel-volume integration;
3. singular-transform regularization;
4. damping toward `Theta_core`;
5. a contractive inward recurrence; and
6. field-shape validation.

## Relationship to the virtual 3D AE/AD runtime

`dr_moagi_virtual_3d_ae.py` remains the bitwise sparse auto-encoding/decoding implementation. The continuum kernel is a continuous-valued state-transition primitive that can sit before, after, or around that codec without changing the codec's deterministic bitstream contract.

The intended composition is

\[
\text{3D input field}
\rightarrow \mathcal E
\rightarrow \Phi
\rightarrow \mathcal M_{\mathrm{continuum}}
\rightarrow \Psi
\rightarrow \mathcal D
\rightarrow \widehat X,
\]

with the recursive path

\[
\Psi_k\rightarrow\mathcal M_{\mathrm{continuum}}\rightarrow\Psi_{k+1}.
\]
