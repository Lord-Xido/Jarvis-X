# Dr Moagi Unified Auto-Encoding Dynamics (D-MUAD)

## Status

**D-MUAD v2.0-C** is the canonical mathematical constitution for a compressive,
recurrent, physics-regularized, dual-branch geometric autoencoding system.

The system is a closed computational recurrence, not a closed-form global
solution. Its neural training core is differentiable almost everywhere; its
hard egress operations are discrete.

---

## 1. Constitutional classification

D-MUAD implements

\[
\mathcal X
\xrightarrow{\Pi}
\mathcal X_p
\xrightarrow{\mathcal E_{emb}}
\mathcal X_e
\xrightarrow{\mathcal E_{3D}}
\mathbf v_q
\xrightarrow{\mathcal L_{phy}}
(\mathbf h_q,\mathbf c_q)
\xrightarrow{\mathcal P}
\mathbf Z_q
\xrightarrow{\mathcal D_{vol},\mathcal R_{NeRF}}
(\mathcal G_{rgb},\mathcal G_{sdf},\mathcal G_\rho,\widehat{\mathbf C}).
\]

The canonical claim set is:

- **representation:** compressive and distributionally reconstructive;
- **inverse:** approximate decoder, not a global exact inverse;
- **differentiability:** training core differentiable almost everywhere;
- **determinism:** conditional on fixed parameters, recurrent state, kernels,
  statistics, seeds, hardware behavior, and reduction order;
- **optimization:** closed Adam recurrence, not a closed-form optimum;
- **arithmetic budget:** derived from concrete tensor dimensions and kernel
  implementation.

---

## 2. Spacetime domain and padding

Let

\[
\mathcal X\in\mathbb R^{B\times4\times T\times H\times W}.
\]

Define

\[
T_p=8\left\lceil\frac{T}{8}\right\rceil,\qquad
H_p=8\left\lceil\frac{H}{8}\right\rceil,\qquad
W_p=8\left\lceil\frac{W}{8}\right\rceil.
\]

The padded tensor is

\[
\mathcal X_p=\Pi_W\circ\Pi_H\circ\Pi_T(\mathcal X)
\in\mathbb R^{B\times4\times T_p\times H_p\times W_p}.
\]

A validity mask must accompany the tensor:

\[
M[b,t,i,j]=
\begin{cases}
1,&t<T,\ i<H,\ j<W,\\
0,&\text{otherwise}.
\end{cases}
\]

Losses are normalized over valid elements, not padded zeros.

---

## 3. Dr Moagi Transform I: separable spatiotemporal embedding

For output channel \(c\in\{0,\dots,63\}\),

\[
\mathcal X_e[b,c,t,i,j]
=
\beta_c+
\sum_{k=0}^{3}
\sum_{w=-1}^{1}
\sum_{u=-1}^{1}
\sum_{v=-1}^{1}
K_t[c,w]K_s[c,k,u,v]
\mathcal X_p[b,k,t+w,i+u,j+v].
\]

The boundary convention must be explicit: zero, reflection, replication, or
circular padding. The canonical output is

\[
\mathcal X_e\in\mathbb R^{B\times64\times T_p\times H_p\times W_p}.
\]

---

## 4. Dr Moagi Transform II: hierarchical 3D encoding

Using same-padded 3D convolutions,

\[
\begin{aligned}
\mathcal H_1 &= \operatorname{ReLU}(\operatorname{BN}(W_1*\mathcal X_e+b_1)),\\
\mathcal H_2 &= \operatorname{ReLU}(\operatorname{BN}(W_2*\mathcal H_1+b_2)),\\
\mathcal H_3 &= \operatorname{ReLU}(\operatorname{BN}(W_3*\mathcal H_2+b_3)).
\end{aligned}
\]

The canonical shapes are

\[
\begin{aligned}
\mathcal H_1&:[B,256,T_p/2,H_p/2,W_p/2],\\
\mathcal H_2&:[B,512,T_p/4,H_p/4,W_p/4],\\
\mathcal H_3&:[B,1024,T_p/4,H_p/4,W_p/4].
\end{aligned}
\]

Global average pooling gives

\[
\mathbf v_q[b,c]
=
\frac{1}{(T_p/4)(H_p/4)(W_p/4)}
\sum_{t,i,j}\mathcal H_3[b,c,t,i,j].
\]

The index \(q\) denotes the recurrent observation cycle. It is distinct from
frame time \(t\).

---

## 5. Dr Moagi Transform III: recurrent latent physics

For observation cycle \(q\),

\[
\begin{aligned}
\mathbf i_q&=\sigma(W_{ii}\mathbf v_q+b_{ii}+W_{hi}\mathbf h_{q-1}+b_{hi}),\\
\mathbf f_q&=\sigma(W_{if}\mathbf v_q+b_{if}+W_{hf}\mathbf h_{q-1}+b_{hf}),\\
\mathbf g_q&=\tanh(W_{ig}\mathbf v_q+b_{ig}+W_{hg}\mathbf h_{q-1}+b_{hg}),\\
\mathbf o_q&=\sigma(W_{io}\mathbf v_q+b_{io}+W_{ho}\mathbf h_{q-1}+b_{ho}),\\
\mathbf c_q&=\mathbf f_q\odot\mathbf c_{q-1}+\mathbf i_q\odot\mathbf g_q,\\
\mathbf h_q&=\mathbf o_q\odot\tanh(\mathbf c_q).
\end{aligned}
\]

The latent code is

\[
\boxed{
\mathbf Z_q=W_{proj}[\mathbf h_{q-1};\mathbf h_q]+b_{proj}
}
\qquad
\mathbf Z_q\in\mathbb R^{B\times1024}.
\]

---

## 6. Injectivity boundary

The padded input contains

\[
4T_pH_pW_p
\]

values per sample, while the latent contains \(1024\). Strided convolutions,
ReLU, global pooling, and the finite bottleneck are many-to-one. Therefore

\[
\boxed{
\mathcal E_{D\text{-}MUAD}\text{ is not globally injective.}
}
\]

The reconstruction contract is

\[
\boxed{
\mathcal D_\Theta(\mathcal E_\Theta(\mathcal X))\approx\mathcal X
}
\]

over the modeled data distribution.

---

## 7. Dr Moagi Transform IV: volumetric decoder

The original dense expansion to

\[
[B,1024,T_p/4,H_p/4,W_p/4]
\]

requires

\[
1024^2(T_p/4)(H_p/4)(W_p/4)
=
16\,384T_pH_pW_p
\]

weights, excluding bias. This term is resolution dependent and must be counted
explicitly.

A scalable implementation should use a fixed learned seed followed by
convolutional upsampling and final interpolation. The semantic decoder heads
must remain distinct:

\[
\begin{aligned}
\mathcal G_{rgb}&=\sigma(L_{rgb}),\\
\mathcal G_{sdf}&=L_{sdf},\\
\mathcal G_{\rho}&=\operatorname{softplus}(L_\rho).
\end{aligned}
\]

The output therefore has five semantic channels or three separate heads:
RGB, signed distance, and nonnegative density.

---

## 8. Dr Moagi Transform V: NeRF branch

For camera ray

\[
\mathbf r(s)=\mathbf o+s\mathbf d
\]

and samples \(s_k\), the radiance network maps

\[
(\mathbf Z_q,\gamma(\mathbf r(s_k)),\gamma(\mathbf d))
\mapsto
(\rho_k,\mathbf c_k).
\]

Use

\[
\rho_k=\operatorname{softplus}(W_\rho f_k+b_\rho),
\qquad
\mathbf c_k=\sigma(W_cf_k+b_c).
\]

Then

\[
\alpha_k=1-e^{-\rho_k\delta_k},
\qquad
T_k=\prod_{m<k}(1-\alpha_m),
\]

and

\[
\boxed{
\widehat{\mathbf C}(i,j)=\sum_kT_k\alpha_k\mathbf c_k.
}
\]

The branch requires a rendering loss or its parameters receive no gradient from
the volumetric losses.

---

## 9. Unified corrected loss

Normalize RGB to \([0,1]\). Define

\[
\mathcal L_{sdf}
=
\frac{\sum M(\mathcal G_{sdf}-\mathcal SDF_{gt})^2}{\sum M},
\]

\[
\mathcal L_{app}
=
\frac{\sum M\|\mathcal G_{rgb}-\mathcal X_{rgb}^{norm}\|_2^2}{3\sum M},
\]

\[
\mathcal L_{phy}
=
\frac{1}{1024B}\sum_b\|\mathbf h_q^{(b)}-\mathbf h_{q-1}^{(b)}\|_2^2,
\]

\[
\mathcal L_{eik}
=
\frac{1}{|\mathcal R|}
\sum_{\mathbf r\in\mathcal R}
(\|\nabla\mathcal G_{sdf}(\mathbf r)\|_2-1)^2,
\]

and

\[
\mathcal L_{render}
=
\frac{\sum M\|\widehat{\mathbf C}-\mathbf C_{gt}\|_2^2}{3\sum M}.
\]

The canonical aggregate is

\[
\boxed{
\mathcal L_{total}
=
\mathcal L_{sdf}
+0.5\mathcal L_{app}
+0.1\mathcal L_{phy}
+0.1\mathcal L_{eik}
+\mathcal L_{render}.
}
\]

---

## 10. Adam recurrence

For each parameter \(w\),

\[
\begin{aligned}
m_q&=\beta_1m_{q-1}+(1-\beta_1)g_q,\\
v_q&=\beta_2v_{q-1}+(1-\beta_2)g_q^2,\\
\widehat m_q&=m_q/(1-\beta_1^q),\\
\widehat v_q&=v_q/(1-\beta_2^q),\\
w_{q+1}&=w_q-\eta\frac{\widehat m_q}{\sqrt{\widehat v_q}+\epsilon}.
\end{aligned}
\]

This is a closed numerical recurrence. It does not guarantee a global optimum.

---

## 11. Arithmetic closure

For a dense 3D convolution,

\[
\operatorname{MAC}
=
BD_oH_oW_oC_oC_iK_tK_hK_w.
\]

For a dense layer,

\[
\operatorname{MAC}=Bn_{in}n_{out}.
\]

For NeRF,

\[
\operatorname{MAC}_{NeRF}
\propto
BTHWN_s\operatorname{MAC}_{MLP}.
\]

A universal fixed MAC count is invalid without fixed batch size, padded domain,
ray count, sample count, MLP widths, decoder implementation, and convolution
semantics. Backpropagation is implementation dependent and is not exactly twice
the forward MAC count as a universal law.

---

## 12. Differentiability and egress boundary

The training graph is differentiable almost everywhere. The following are hard,
discrete operations:

- occupancy thresholding;
- set construction;
- clipping and rounding;
- uint8 conversion;
- conventional marching cubes;
- mesh topology changes.

Use

\[
\mathcal Y_{rgb}=\operatorname{round}(255\mathcal G_{rgb})
\]

only at egress. Extract the mesh from the SDF zero level set:

\[
\mathcal Y_{mesh}=\operatorname{MarchingCubes}(\mathcal G_{sdf},0).
\]

---

## 13. Canonical closed recurrence

\[
\boxed{
\begin{aligned}
\mathcal X_p&=\Pi(\mathcal X),\\
\mathcal X_e&=\mathcal E_{emb}(\mathcal X_p),\\
\mathcal H&=\mathcal E_{3D}(\mathcal X_e),\\
\mathbf v_q&=\operatorname{GAP}(\mathcal H_q),\\
(\mathbf h_q,\mathbf c_q)&=\mathcal L_{phy}(\mathbf v_q,\mathbf h_{q-1},\mathbf c_{q-1}),\\
\mathbf Z_q&=W_{proj}[\mathbf h_{q-1};\mathbf h_q]+b_{proj},\\
(\mathcal G_{rgb},\mathcal G_{sdf},\mathcal G_\rho)&=\mathcal D_{vol}(\mathbf Z_q),\\
\widehat{\mathbf C}&=\mathcal R_{NeRF}(\mathbf Z_q,\mathcal C),\\
\Theta_{q+1}&=\operatorname{Adam}(\Theta_q,\nabla_\Theta\mathcal L_{total}).
\end{aligned}
}
\]

---

## 14. Executable contract

`src/jarvisx/d_muad.py` makes the following properties executable without a
machine-learning dependency:

- minimal padding to a multiple of eight;
- complete tensor-shape derivation;
- explicit non-injectivity classification;
- dense decoder parameter accounting;
- shape-dependent convolution MAC counts;
- corrected five-term loss aggregation;
- scalar Adam state recurrence;
- constitutional claim boundaries.

The executable contract is a validator and arithmetic constitution. A future
PyTorch, JAX, or TensorFlow backend must conform to it rather than silently
changing tensor semantics.
