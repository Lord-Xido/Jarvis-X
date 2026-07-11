# Dr. Moagi Unified Auto-Encoding Equation for GPT-3D

## Status

This document defines the formal research architecture for the GPT-3D
encoder-decoder family. It is a mathematical target specification, distinct
from the dependency-free reference runtime in `src/jarvisx/engine30d.py`.

The architecture combines:

- a 3D embedded byte grid;
- spherical causal axial attention;
- 3D rotary positional embeddings;
- spatially tiled feed-forward weights;
- depthwise 3D convolution;
- a volumetric bottleneck;
- a learned adjoint decoder;
- 3D reconstruction and total-variation losses;
- structured tile sparsity.

---

## 1. State space and notation

Let

\[
\mathbf X\in\mathbb R^{H\times W\times D\times d}
\]

be an embedded 3D byte grid. A voxel position is
\(p=(x,y,z)\), with radial coordinate

\[
r(p)=\lVert p\rVert_2.
\]

The encoder produces

\[
\mathbf Z\in
\mathbb R^{\lceil H/2\rceil\times\lceil W/2\rceil\times
\lceil D/2\rceil\times d'}.
\]

For a one-byte-per-voxel volume, a 2048-byte input means
\(HWD=2048\); the spatial factorisation is an implementation choice.

---

## 2. Three-dimensional rotary position operator

Split each query and key channel vector into three even-dimensional channel
blocks associated with the \(x\), \(y\), and \(z\) axes. Define

\[
\mathcal R_{3D}(p)
=
\operatorname{diag}
\bigl(
R_x(x),R_y(y),R_z(z)
\bigr),
\]

where each \(R_a(\cdot)\) is a block-diagonal matrix of planar rotations.
Because every block is orthogonal,

\[
\mathcal R_{3D}(p)^{-1}
=
\mathcal R_{3D}(p)^{\top}.
\]

RoPE is applied to queries and keys before attention scoring:

\[
\widetilde{\mathbf Q}^{a}_{p}
=
\mathcal R_{3D}(p)\mathbf Q^{a}_{p},
\qquad
\widetilde{\mathbf K}^{a}_{q}
=
\mathcal R_{3D}(q)\mathbf K^{a}_{q}.
\]

Values are not rotated unless a specific implementation explicitly chooses a
rotary value representation.

---

## 3. Spherical causal axial attention

For each axis \(a\in\{x,y,z\}\), let \(\mathcal L_a(p)\) be the set of
positions lying on the axial line through query position \(p\). The radial
causal mask is

\[
M_S(p,q)
=
\begin{cases}
0, & q\in\mathcal L_a(p)\ \land\ r(q)\le r(p),\\
-\infty, & \text{otherwise}.
\end{cases}
\]

This convention means that a query may attend to equal-or-earlier radial
shells. Reversing the inequality produces the opposite, outside-in causal
ordering. Radial order represents physical time only when the data mapping
explicitly identifies radius with temporal progression.

For axis \(a\),

\[
\operatorname{Attn}^{a}_{S}(\mathbf X)_p
=
\sum_{q\in\mathcal L_a(p)}
\alpha^{a}_{pq}\mathbf V^{a}_{q},
\]

with

\[
\alpha^{a}_{pq}
=
\operatorname{softmax}_{q}
\left(
\frac{
\widetilde{\mathbf Q}^{a}_{p}
(\widetilde{\mathbf K}^{a}_{q})^{\top}
}{\sqrt{d_k}}
+M_S(p,q)
\right),
\]

and

\[
\mathbf Q^{a}=\mathbf X\mathbf W_Q^{a},
\quad
\mathbf K^{a}=\mathbf X\mathbf W_K^{a},
\quad
\mathbf V^{a}=\mathbf X\mathbf W_V^{a}.
\]

The three axial outputs are concatenated and projected back to the model
width:

\[
\boxed{
\mathbb A_{S}(\mathbf X)
=
\left[
\operatorname{Attn}^{x}_{S}(\mathbf X)
\,\Vert\,
\operatorname{Attn}^{y}_{S}(\mathbf X)
\,\Vert\,
\operatorname{Attn}^{z}_{S}(\mathbf X)
\right]\mathbf W_O
}.
\]

Bidirectional decoding attention uses the same construction with
\(M_S=0\).

---

## 4. Three-dimensional tiled feed-forward operator

Partition the spatial volume into tiles \(\tau\in\mathcal T\). Each tile has
local parameters \(\mathbf W_{1,\tau}^{(l)}\) and
\(\mathbf W_{2,\tau}^{(l)}\), optionally generated from a smaller shared
parameter bank.

For voxel \(p\) in tile \(\tau(p)\),

\[
\operatorname{TiledFFN}^{(l)}(\mathbf U)_p
=
\mathbf W_{2,\tau(p)}^{(l)}
\,\sigma\!\left(
\mathbf W_{1,\tau(p)}^{(l)}\mathbf U_p
+\mathbf b_{1,\tau(p)}^{(l)}
\right)
+\mathbf b_{2,\tau(p)}^{(l)}.
\]

This provides local parameter specialisation while preserving an explicit tile
structure for pruning, routing, and provenance.

---

## 5. Dr. Moagi encoding transform

Define

\[
\mathbf U
=
\operatorname{TiledFFN}
\left(
\mathbb A_S(\mathbf X)
\right).
\]

Let \(\operatorname{DWConv3D}_{\mathbf C}\) be a channel-wise 3D convolution
with kernel \(\mathbf C\in\mathbb R^{3\times3\times3\times d'}\). To match
the explicit algebraic expansion below, the bottleneck is average pooling:

\[
\boxed{
\mathbf Z
=
\mathcal E_{\Theta_{enc}}(\mathbf X)
=
\operatorname{AvgPool3D}_{2,2,2}
\left[
\sigma\left(
\operatorname{DWConv3D}_{\mathbf C}(\mathbf U)
\right)
\right]
}.
\]

If max pooling is selected instead, the \(1/8\) average in the expanded form
must be replaced by a maximum over the eight samples.

For output voxel \(s=(i,j,k)\),

\[
\mathcal E(\mathbf X)_{ijk}
=
\frac{1}{8}
\sum_{a,b,c\in\{0,1\}}
\sigma\!\left(
\sum_{u,v,w=-1}^{1}
\mathbf C_{uvw}\odot
\mathbf U_{2i+a+u,\,2j+b+v,\,2k+c+w}
\right).
\]

---

## 6. Dr. Moagi decoding transform

The decoder is a learned approximate inverse. Transposed convolution and
transposed tiled weights are adjoint operators; they are not guaranteed exact
inverses unless additional orthogonality and perfect-reconstruction conditions
are imposed.

Let

\[
\mathbf V
=
\mathbb A_{\mathrm{bidir}}(\mathbf Z),
\qquad M_S=0.
\]

A tied-adjoint decoder is

\[
\boxed{
\widehat{\mathbf X}
=
\mathcal D_{\Theta_{dec}}(\mathbf Z)
=
\sigma\left[
\operatorname{ConvTranspose3D}_{2,2,2}
\left(
\operatorname{TiledFFN}^{\dagger}
\left(
\operatorname{DWConv3D}_{\mathbf C}^{\dagger}(\mathbf V)
\right)
\right)
\right]
}.
\]

Here \(\dagger\) denotes an adjoint or tied transpose. An untied decoder may
instead learn independent parameters. RoPE does not need to be globally
"undone" after attention because it acts inside the query-key score; when an
explicit rotated latent representation is stored, its inverse is
\(\mathcal R_{3D}^{\top}\).

---

## 7. Unified Dr. Moagi objective

Let

\[
\Theta
=
\{\mathbf W_{Q,K,V,O},\mathbf W_{\mathrm{tile}}^{(l)},
\mathbf C,\Theta_{dec}\}.
\]

Define the reconstruction residual

\[
\mathbf Y
=
\mathbf X-
\mathcal D_{\Theta_{dec}}
\left(
\mathcal E_{\Theta_{enc}}(\mathbf X)
\right).
\]

The anisotropy-aware 3D total variation is

\[
\operatorname{TV}_{3D}(\mathbf Y)
=
\sum_{x,y,z}
\sqrt{
\lVert\nabla_x\mathbf Y\rVert_2^2
+
\lVert\nabla_y\mathbf Y\rVert_2^2
+
\lVert\nabla_z\mathbf Y\rVert_2^2
+\varepsilon
}.
\]

For an \(8\times8\times8\) tile lattice, define the hierarchical mixed tile
norm

\[
\lVert\mathbf W\rVert_{(1,2,1),\mathrm{tile}}
=
\sum_{i=1}^{8}
\left[
\sum_{j=1}^{8}
\left(
\sum_{k=1}^{8}
\lVert\mathbf W_{ijk}\rVert_F
\right)^2
\right]^{1/2}.
\]

This is a structured mixed norm, not a tensor nuclear norm. It encourages
hierarchical block sparsity across tiled partitions.

The closed training objective is

\[
\boxed{
\Theta^*
=
\arg\min_{\Theta}
\mathbb E_{\mathbf X\sim\mathcal B}
\left[
\lVert\mathbf Y\rVert_F^2
+
\lambda\operatorname{TV}_{3D}(\mathbf Y)
\right]
+
\beta\sum_{l=1}^{L}
\lVert\mathbf W_l^{\mathrm{Tiled}}\rVert_{(1,2,1),\mathrm{tile}}
+
\gamma\sum_{l=1}^{L}\mathcal R_{3/2}(\mathbf W_l)
}.
\]

The optional smooth fractional stabiliser is

\[
\mathcal R_{3/2}(\mathbf W)
=
\frac{2}{3}\sum_n |w_n|^{3/2},
\]

whose gradient is

\[
\nabla_{\mathbf W}\mathcal R_{3/2}
=
\operatorname{sign}(\mathbf W)\odot|\mathbf W|^{1/2}.
\]

---

## 8. Moagi composite gradient-proximal flow

The fractional term is a differentiable regulariser, not by itself a proximal
operator. Structured sparsity is enforced by applying the proximal map of the
tile norm after the smooth gradient step:

\[
\widetilde{\mathbf W}^{(l)}_{t+1}
=
\mathbf W^{(l)}_t
-
\eta
\left[
\frac{\partial\mathcal L_{\mathrm{fidelity}}}
{\partial\mathbf W^{(l)}}
+
\gamma\,
\operatorname{sign}(\mathbf W^{(l)}_t)
\odot
|\mathbf W^{(l)}_t|^{1/2}
\right],
\]

followed by

\[
\boxed{
\mathbf W^{(l)}_{t+1}
=
\operatorname{prox}_{\eta\beta
\lVert\cdot\rVert_{(1,2,1),\mathrm{tile}}}
\left(
\widetilde{\mathbf W}^{(l)}_{t+1}
\right)
}.
\]

This separates smooth optimisation from explicit block-sparsity projection.
When the parameter space is constrained to a matrix manifold, the Euclidean
gradient must additionally be projected onto the tangent space and retracted;
without that manifold definition, the update is proximal gradient descent
rather than Riemannian gradient descent.

---

## 9. Closed system identity

\[
\boxed{
\mathbf X
\xrightarrow{\mathbb A_S+\mathcal R_{3D}}
\xrightarrow{\mathrm{TiledFFN}}
\xrightarrow{\mathrm{DWConv3D}}
\xrightarrow{\mathrm{Pool3D}}
\mathbf Z
\xrightarrow{\mathbb A_{bidir}}
\xrightarrow{\mathrm{adjoint\ decoder}}
\widehat{\mathbf X}
}
\]

The architecture is closed in the optimisation sense:

\[
\Theta^*
=
\arg\min_{\Theta}
\left(
\text{volumetric reconstruction}
+
\text{3D structural regularity}
+
\text{hierarchical tile sparsity}
\right).
\]

It does not claim that the decoder is an algebraic inverse for arbitrary
learned parameters. Instead, the end-to-end objective trains a constrained
approximate inverse whose reconstruction quality, spatial regularity, and tile
sparsity are jointly measurable.

---

## 10. Relationship to the current Jarvis-X runtime

The current standard-library engine implements the operational skeleton:

```text
normalise -> sparse route -> encode -> predict -> residual
          -> omega update -> correct -> decode -> verify -> commit
```

This document specifies the future tensor backend that may replace the scalar
reference transforms while preserving the same deterministic transaction,
policy, provenance, and rollback boundaries.
