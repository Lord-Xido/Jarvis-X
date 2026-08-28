# Dr. Moagi Inward Self-Optimizing Render Engine

This document fixes a bounded, executable reference model for the pipeline

\[
\mathcal B \to \mathbf M \to \mathbf V_{raw} \to \mathbf z
\to \mathbf V_{recon} \to \mathbf I
\to \mathcal L_{Total} \to \text{inward update}.
\]

The implementation is `src/jarvisx/inward_render_runtime.py` and deliberately uses only the Python standard library.

## 1. Bytecode and VM transform

The reference ISA is intentionally small:

| Opcode | Encoding | Meaning |
|---|---|---|
| `0x01` | opcode + 3 little-endian `float32` | translate `(tx,ty,tz)` |
| `0x02` | opcode + 1 `float32` | rotate around Y |
| `0x03` | opcode + 1 `float32` | uniform scale |
| `0xFF` | opcode only | halt |

For column-vector homogeneous coordinates the VM accumulates

\[
\mathbf M_{k+1}=\mathbf M_k\mathbf T_k,\qquad \mathbf M_0=\mathbf I_4.
\]

A mathematical precision point is important: if the program contains only rotation and translation, then `M` lies in `SE(3)`. Once scale is allowed, the transform is no longer generally an element of `SE(3)`; the bounded reference uses uniform scale, so the transform lies in the 3-D similarity group `Sim(3)` (and more general scaling would belong to an affine group).

The VM maintains both `M` and `M^{-1}` so the rasterizer can transform world-space samples into canonical primitive space.

## 2. Volumetric SDF rasterization

For world-space voxel center `p_q` and canonical sphere radius `r`, the raw SDF is

\[
V_{raw}(q)=s\left(\left\|\mathbf M^{-1}p_q\right\|_2-r\right),
\]

where `s` is the accumulated positive uniform scale. The default model uses `D=64`, hence

\[
\mathbf V_{raw}\in\mathbb R^{64\times64\times64}.
\]

Tests use smaller grids so CI remains fast while exercising the same arithmetic.

## 3. 3-D autoencoding bottleneck

The standard-library reference divides the volume into a `G x G x G` spatial latent lattice. With default `G=4`,

\[
\dim z=G^3=64.
\]

For block `b`,

\[
\bar V_b=\frac{1}{|b|}\sum_{q\in b}V_{raw}(q),
\qquad
z_b=\tanh(\phi_g\bar V_b+\phi_b).
\]

The decoder reconstructs each voxel in block `b(q)` by

\[
V_{recon}(q)=\psi_g z_{b(q)}+\psi_b.
\]

The four trainable reference parameters are

\[
(\phi_g,\phi_b,\psi_g,\psi_b).
\]

This is a deliberately compact neural bottleneck, not a claim that a production convolutional network can be reduced to four scalars. It provides an auditable reference seam that can later be replaced by the existing C++/3-D convolutional backend while preserving the same forward and optimization contracts.

## 4. Sphere-tracing renderer

For pixel `(i,j)`,

\[
r_{ij}(t)=o+t d_{ij}.
\]

The implementation evaluates the reconstructed field by trilinear interpolation and advances with a conservative step

\[
t_{m+1}=t_m+\max(0.75|V_{recon}(r(t_m))|,\Delta t_{min}).
\]

A hit is accepted when

\[
|V_{recon}(r(t_m))|\le \epsilon.
\]

The surface normal is the normalized central-difference gradient

\[
N(p)=\frac{\nabla V_{recon}(p)}{\|\nabla V_{recon}(p)\|_2}.
\]

Telemetry records ray count, hit count, mean march steps and maximum observed steps.

## 5. Cook-Torrance PBR

The pixel shader evaluates a Cook-Torrance microfacet BRDF with GGX normal distribution, Smith masking-shadowing and Schlick Fresnel approximation:

\[
f_r = \frac{k_d a}{\pi}
+ \frac{D_{GGX}(N,H)G(N,V,L)F(V,H)}{4(N\cdot V)(N\cdot L)}.
\]

The final RGB intensity is ambient plus the BRDF contribution multiplied by `max(N dot L,0)` and clamped to `[0,1]`.

## 6. Composite inward objective

The executable objective is

\[
\boxed{
\mathcal L_{Total}=
\lambda_r\mathcal L_{recon}
+\lambda_e\mathcal L_{Eikonal}
+\lambda_b\mathcal L_{Bytecode}
+\lambda_t\mathcal L_{Telemetry}
}
\]

with:

### Reconstruction

\[
\mathcal L_{recon}=\frac1{D^3}\sum_q(V_{raw}(q)-V_{recon}(q))^2.
\]

### Eikonal validity

\[
\mathcal L_{Eikonal}=\frac1{|\Omega_i|}\sum_{q\in\Omega_i}
(\|\nabla V_{recon}(q)\|_2-1)^2.
\]

### Bytecode and latent stability

\[
\mathcal L_{Bytecode}=
\frac1P\|\theta_{\mathcal B}\|_2^2
+\frac1Z\|z_t-z_{t-1}\|_2^2.
\]

### Ray-march telemetry

\[
\mathcal L_{Telemetry}=\left(
\frac{\max(\bar m-M_{target},0)}{\max(M_{target},1)}
\right)^2.
\]

## 7. Inward continuous-time optimization

Collect all trainable parameters into

\[
\Theta=[\phi_g,\phi_b,\psi_g,\psi_b,\theta_{\mathcal B}].
\]

The conceptual continuous-time dynamics are

\[
\frac{d\Theta}{d\tau}=-\nabla_{\Theta}\mathcal L_{Total}.
\]

The bounded reference discretizes this with explicit Euler:

\[
\Theta_{k+1}=\Pi_{\mathcal C}\left[
\Theta_k-\Delta\tau\,\widehat{\nabla_{\Theta}\mathcal L_{Total}}
\right],
\]

where `Pi_C` projects parameters into finite admissible bounds. The current reference gradient estimator is symmetric finite difference across the complete end-to-end program. That choice keeps the renderer dependency-free and makes gradients available even through ray-march telemetry. A production tensor backend may replace this estimator with exact reverse-mode automatic differentiation without changing the state-transition interface.

Gradients are norm-clipped. The proposed Euler step then enters the canonical kinetic transaction:

`SNAPSHOT -> OBSERVE -> ENCODE -> PROPOSE -> SHADOW -> VERIFY -> COMMIT | ROLLBACK -> JOURNAL -> REENTER`.

A candidate is authoritative only if:

1. every parameter is finite; and
2. the measured total loss is non-regressing within the configured numerical tolerance.

The optimizer performs bounded backtracking before verification. Rejected candidates return the exact authoritative snapshot.

## 8. Master operational equation

The complete executable composition is

\[
\boxed{
\begin{aligned}
\mathbf M(\mathcal B_k)
&=\prod_{n=1}^{K}T_n(\theta_{n,k}),\\
\mathbf V_{raw,k}
&=\mathcal R_{SDF}(\mathbf M(\mathcal B_k)),\\
\mathbf z_k
&=f_{\phi_k}(\mathbf V_{raw,k}),\\
\mathbf V_{recon,k}
&=g_{\psi_k}(\mathbf z_k),\\
\mathbf I_k
&=\mathcal P_{CT}(\mathcal S_{ray}(\mathbf V_{recon,k})),\\
\mathcal L_k
&=\mathcal L_{Total}(\mathcal B_k,\phi_k,\psi_k,\mathbf z_k,\mathbf I_k),\\
\widetilde\Theta_{k+1}
&=\Pi_{\mathcal C}[\Theta_k-\Delta\tau\widehat\nabla_\Theta\mathcal L_k],\\
\Theta_{k+1}
&=\begin{cases}
\widetilde\Theta_{k+1}, & \mathcal L(\widetilde\Theta_{k+1})\le\mathcal L(\Theta_k)+\varepsilon,\\
\Theta_k, & \text{otherwise}.
\end{cases}
\end{aligned}
}
\]

This last conditional is the operational meaning of **turning the renderer inward onto itself**: the rendered system supplies its own measurable reconstruction, field-validity, program-complexity and runtime-telemetry errors; those errors propose the next state; the next state is accepted only after shadow evaluation proves bounded non-regression.

## 9. Run it

A portable PPM frame and JSON telemetry can be produced directly:

```bash
python -m jarvisx.inward_render_runtime --grid-size 24 --width 32 --height 24 --output inward_frame.ppm
```

Run one or more inward optimization transactions:

```bash
python -m jarvisx.inward_render_runtime --grid-size 12 --width 16 --height 12 --optimize-steps 2 --output inward_optimized.ppm
```

For the exact default volumetric dimension in the mathematical model, use `--grid-size 64`. Optimization at that setting is intentionally expensive in this pure-Python reference because each finite-difference parameter evaluation executes the full renderer.

## 10. Verification surface

`tests/test_inward_render_runtime.py` checks:

- bytecode serialization and VM transform execution;
- volumetric SDF sign structure;
- the complete forward pipeline and finite total loss;
- the inward optimizer's non-regression invariant; and
- portable PPM frame emission.
