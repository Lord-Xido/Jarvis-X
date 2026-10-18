# Dr Moagi Tri-Space Fixed-Point SDF / Differentiable Vision Specification

**Status:** Research specification  
**Date:** 2026-09-30  
**Applies to:** Jarvis-X geometric autoencoding, differentiable rendering, multimodal vision loops, fixed-point latent optimization  
**Related:** ADR-016, ADR-017, ADR-028, ADR-030

## 1. Purpose

This specification couples three differentiable state spaces into one bounded recursive operator:

1. a continuous 3D geometric manifold represented by a neural signed-distance field;
2. a 2D differentiable image plane produced by rendering that field;
3. a latent visual/Transformer state used to compare, refine and re-encode geometry.

The canonical fixed-point condition is

[
mathbf z^* = Phi(mathbf z^*).
]

The idempotence relation

[
Phi(Phi(mathbf z^*)) = Phi(mathbf z^*)
]

then follows immediately. It is not a substitute for the fixed-point definition.

The operator (Phi) is an internal candidate-state update. Convergence of this loop is evidence of internal numerical consistency only; it does not by itself establish physical truth, external correctness or production validity.

---

## 2. Coupled state

Let

[
mathbf z
=
[mathbf z_{mathrm{geom}},
 mathbf z_{mathrm{mat}},
 mathbf z_{mathrm{vis}}],
]

with

- (mathbf z_{mathrm{geom}}): geometry latent;
- (mathbf z_{mathrm{mat}}): material/lighting latent;
- (mathbf z_{mathrm{vis}}): visual semantic latent.

Define

[
Phi
=
mathcal U
circ
mathcal F
circ
mathcal V
circ
mathcal R
circ
mathcal G,
]

where

- (mathcal G) decodes latent geometry into an SDF;
- (mathcal R) differentiably renders the SDF;
- (mathcal V) tokenizes and embeds the rendered image;
- (mathcal F) fuses geometric, visual and target evidence;
- (mathcal U) performs one bounded latent update.

The recursive system is

[
mathbf z^{(k+1)} = Phi(mathbf z^{(k)};mathcal E_{mathrm{target}}).
]

A candidate fixed point is accepted only when its residuals, resource bounds and enclosing Jarvis-X verification gates pass.

---

## 3. Continuous 3D geometry

### 3.1 Neural signed-distance field

Let

[
f_	heta:
mathbb R^3 	imes mathcal Z_{mathrm{geom}}
ightarrow
mathbb R.
]

The surface is the zero level set

[
partialmathcal S
=
{mathbf x:f_	heta(mathbf x,mathbf z_{mathrm{geom}})=0}.
]

The ideal sign convention is

[
f_	heta(mathbf x)<0
Rightarrow
mathbf xinoperatorname{int}mathcal S,
qquad
f_	heta(mathbf x)>0
Rightarrow
mathbf x
otinmathcal S.
]

For a true Euclidean SDF,

[
|
abla_{mathbf x} f_	heta(mathbf x)|_2=1
]

almost everywhere away from non-differentiable medial-axis locations.

A learned field enforces this approximately through the Eikonal loss

[
mathcal L_{mathrm{eik}}
=
mathbb E_{mathbf x}
left(
|
abla_{mathbf x}f_	heta(mathbf x)|_2-1
ight)^2.
]

### 3.2 Surface normals

The differentiable unit normal is

[
mathbf n(mathbf x)
=
rac{
abla_{mathbf x}f_	heta(mathbf x)}
{|
abla_{mathbf x}f_	heta(mathbf x)|_2+arepsilon}.
]

### 3.3 Smooth solid occupancy

A hard Heaviside interior/exterior step is unsuitable for gradient-based optimization. During training or inverse rendering use

[
o_eta(mathbf x)
=
sigma!left(-eta f_	heta(mathbf x)ight),
]

where (sigma(cdot)) is the logistic sigmoid and (eta>0) controls boundary sharpness.

As (eta	oinfty),

[
o_eta(mathbf x)
ightarrow
mathbf 1[f_	heta(mathbf x)<0].
]

Electrical conductivity and optical extinction are separate quantities. For copper,

[
sigma_e(mathbf x)
=
sigma_{mathrm{Cu}},o_eta(mathbf x)
]

may represent an electrical material mask, while rendering should use a distinct optical extinction coefficient

[
alpha(mathbf x)
=
alpha_{max},o_eta(mathbf x).
]

This avoids conflating electrical conductivity with volume-rendering density.

---

## 4. Hairpin-wire geometry contract

Let the wire centerline be an arc-length-parameterized curve

[
mathbf r(s), qquad sin[0,L],
]

with total centerline length

[
L=1000 {m mm}.
]

The Frenet-Serret frame satisfies

[
rac{dmathbf T}{ds}=kappamathbf N,
]

[
rac{dmathbf N}{ds}
=
-kappamathbf T+	aumathbf B,
]

[
rac{dmathbf B}{ds}
=
-	aumathbf N.
]

### 4.1 Wire radius

Diameter

[
d=1.0 {m mm},
qquad
a=rac d2=0.5 {m mm}.
]

For a non-self-intersecting tubular neighborhood, the ideal wire SDF is locally

[
f(mathbf x)
=
d(mathbf x,mathcal C)-a,
]

where (mathcal C={mathbf r(s)}) and (d(mathbf x,mathcal C)) is the minimum Euclidean distance to the centerline.

### 4.2 Parallel-leg separation

Required inner gap:

[
g=2.0 {m mm}.
]

Therefore the centerline-to-centerline spacing of the two legs is

[
D_{mathrm{cc}}
=
g+2a
=
3.0 {m mm}.
]

For a semicircular planar U-bend,

[
R
=
rac{D_{mathrm{cc}}}{2}
=
1.5 {m mm},
]

so

[
kappa_{mathrm{bend}}
=
rac1R
=
rac23 {m mm}^{-1},
qquad
	au_{mathrm{bend}}=0.
]

The U-bend arc length is

[
L_{mathrm{bend}}
=
pi R
=
1.5pi {m mm}.
]

For equal straight legs,

[
L_{mathrm{leg}}
=
rac{1000-1.5pi}{2}
approx
497.6438055 {m mm}.
]

Thus the nominal analytical centerline is fully determined by the diameter, gap and total length.

### 4.3 Geometry losses

A practical geometry objective is

[
mathcal L_{mathrm{geom}}
=
lambda_{mathrm{eik}}mathcal L_{mathrm{eik}}
+
lambda_{mathrm{surf}}mathcal L_{mathrm{surf}}
+
lambda_{mathrm{rad}}mathcal L_{mathrm{rad}}
+
lambda_{mathrm{gap}}mathcal L_{mathrm{gap}}
+
lambda_{kappa}mathcal L_{kappa}
+
lambda_{	au}mathcal L_{	au}.
]

Representative terms include

[
mathcal L_{mathrm{rad}}
=
mathbb E_s
left(
hat a(s)-0.5
ight)^2,
]

[
mathcal L_{mathrm{gap}}
=
left(
hat g-2.0
ight)^2,
]

[
mathcal L_{kappa}
=
mathbb E_{sinmathrm{bend}}
left(
kappa(s)-rac23
ight)^2,
]

[
mathcal L_{	au}
=
mathbb E_{sinmathrm{bend}}
	au(s)^2.
]

---

## 5. Differentiable projection

### 5.1 Camera ray

For image coordinate ((u,v)),

[
mathbf r_{uv}(t)
=
mathbf o+tmathbf d_{uv},
qquad
tin[t_n,t_f].
]

### 5.2 Volume rendering

Define transmittance

[
T(t)
=
exp
left(
-int_{t_n}^{t}
alpha(mathbf r_{uv}(s)),ds
ight).
]

The rendered color is

[
hat{mathbf I}(u,v)
=
int_{t_n}^{t_f}
T(t),
alpha(mathbf r_{uv}(t)),
mathbf c(mathbf r_{uv}(t),mathbf d_{uv},mathbf n),dt
+
T(t_f)mathbf c_{mathrm{bg}}.
]

For an opaque metallic surface, a surface-oriented SDF renderer or SDF-derived narrow-band opacity may be preferable to a physically thick participating-medium model. The implementation must document which approximation it uses.

### 5.3 Copper BRDF

A Cook-Torrance microfacet term may be used:

[
f_r
=
rac{
D(mathbf h),
F(mathbf v,mathbf h),
G(mathbf n,mathbf v,mathbf l)
}{
4
(mathbf ncdotmathbf v)
(mathbf ncdotmathbf l)
+arepsilon
}.
]

The Fresnel term may use Schlick approximation

[
F
=
F_0
+
(1-F_0)(1-mathbf vcdotmathbf h)^5,
]

with wavelength/channel-dependent (F_0) for copper.

The shading model should preserve explicit material parameters rather than bury them in geometry latent coordinates.

### 5.4 Jacobian semantics

If the renderer is implemented as a smooth discrete computational graph, automatic differentiation produces

[
J_{mathcal R}
=
rac{partialhat{mathbf I}}
{partialmathbf z_{mathrm{geom}}}
]

for that discrete program.

This derivative is exact with respect to the implemented differentiable graph up to floating-point and autodiff semantics; it is not an assertion of an exact closed-form Jacobian for the physical continuum renderer.

---

## 6. Vision tokenization and attention

Let

[
hat{mathbf I}inmathbb R^{H	imes W	imes 3}.
]

Partition the image into (N) patches and linearly embed them:

[
mathbf h_i
=
operatorname{vec}(P_i)W_E+mathbf p_i.
]

Stack tokens into

[
H=[mathbf h_1,dots,mathbf h_N]^	op.
]

Then

[
Q=HW_Q,qquad
K=HW_K,qquad
V=HW_V,
]

and

[
operatorname{Attn}(H)
=
operatorname{Softmax}
left(
rac{QK^	op}{sqrt{d_k}}
ight)V.
]

The resulting visual latent is

[
mathbf z_{mathrm{vis}}
=
mathcal V(hat{mathbf I}).
]

If a target image or multimodal target is available, its embedding is computed independently and compared through an explicit feature metric.

---

## 7. Unified objective

Define

[
mathcal L_{mathrm{total}}
=
lambda_{mathrm{img}}mathcal L_{mathrm{img}}
+
lambda_{mathrm{feat}}mathcal L_{mathrm{feat}}
+
lambda_{mathrm{geom}}mathcal L_{mathrm{geom}}
+
lambda_{mathrm{fp}}mathcal L_{mathrm{fp}}
+
lambda_{mathrm{reg}}mathcal L_{mathrm{reg}}.
]

A fixed-point residual term is

[
mathcal L_{mathrm{fp}}
=
|Phi(mathbf z)-mathbf z|_2^2.
]

Example visual feature loss:

[
mathcal L_{mathrm{feat}}
=
1-
rac{
langle
mathbf z_{mathrm{vis}},
mathbf z_{mathrm{target}}
angle
}{
|mathbf z_{mathrm{vis}}|_2
|mathbf z_{mathrm{target}}|_2
+arepsilon
}.
]

The objective must keep geometry constraints separate from perceptual similarity so that a visually plausible but dimensionally incorrect result cannot silently satisfy the physical contract.

---

## 8. Bounded second-order optimization

A raw Newton update

[
mathbf z_{k+1}
=
mathbf z_k
-
H^{-1}
ablamathcal L
]

is not the default operational rule because the Hessian may be singular, indefinite, poorly conditioned or too large to materialize.

Use a damped step:

[
(H_k+lambda_k I)Deltamathbf z_k
=
-
ablamathcal L(mathbf z_k),
]

followed by

[
mathbf z_{k+1}
=
Pi_{mathcal Z}
left(
mathbf z_k+eta_kDeltamathbf z_k
ight),
]

where

- (lambda_kge0) is damping;
- (0<eta_kle1) is a line-search/trust-region step scale;
- (Pi_{mathcal Z}) is a bounded latent-domain projection.

For high-dimensional latent states, admissible approximations include Hessian-vector products with conjugate gradient, Gauss-Newton / Levenberg-Marquardt on residual-form losses, and L-BFGS.

---

## 9. Contractivity and convergence

For a declared domain (mathcal D), a sufficient contraction condition is

[
sup_{mathbf zinmathcal D}
|J_Phi(mathbf z)|_2
le q<1.
]

Under this condition and completeness assumptions, the iteration has a unique fixed point in the invariant domain and converges locally/globally according to the domain covered by the bound.

Operational stopping requires all of:

[
|mathbf z_{k+1}-mathbf z_k|_2
le
arepsilon_{mathrm{step}},
]

[
|Phi(mathbf z_k)-mathbf z_k|_2
le
arepsilon_{mathrm{fp}},
]

[
|mathcal L_{k+1}-mathcal L_k|
le
arepsilon_{mathcal L},
]

or termination at a configured iteration/resource bound.

A fixed-point receipt should record at minimum:

- iteration count;
- final objective;
- fixed-point residual;
- gradient norm;
- damping/trust-region state;
- geometry residuals;
- image/feature residuals;
- maximum allocated memory;
- wall-clock/runtime budget;
- termination reason.

---

## 10. End-to-end execution law

| Phase | Input | Core map | Output |
|---|---|---|---|
| 1. Latent geometry decode | (mathbf z_{mathrm{geom}}) | (f_	heta(mathbf x,mathbf z_{mathrm{geom}})) | Continuous SDF |
| 2. Physical constraint evaluation | SDF + centerline | Eikonal, radius, gap, curvature, torsion residuals | Geometry receipt |
| 3. Differentiable rendering | SDF + material + camera | (mathcal R) | (hat Iinmathbb R^{H	imes W	imes3}) |
| 4. Vision encoding | (hat I) | patch embedding + Transformer attention | (mathbf z_{mathrm{vis}}) |
| 5. Evidence fusion | geometry + image + target | (mathcal F) | total residual state |
| 6. Bounded optimizer step | residual state | damped Newton/GN/L-BFGS | candidate (mathbf z_{k+1}) |
| 7. Fixed-point test | candidate state | (|Phi(z)-z|) and contractivity receipts | converge / recur |
| 8. Jarvis-X verification | converged candidate | CTR / external checks / resource policy | commit or rollback |

---

## 11. Reference analytical hairpin

A convenient planar centerline can be embedded in the (xy)-plane with two long vertical legs and a bottom semicircle.

Let

[
R=1.5 {m mm},
quad
a=0.5 {m mm},
quad
L_{mathrm{leg}}approx497.6438055 {m mm}.
]

One admissible parameterization is:

Left leg:

[
mathbf r_1(s)
=
(-R, s, 0),
qquad
0le sle L_{mathrm{leg}}.
]

Semicircle, parameterized by (	hetain[pi,0]):

[
mathbf r_2(	heta)
=
(Rcos	heta, L_{mathrm{leg}}-Rsin	heta, 0).
]

Right leg:

[
mathbf r_3(s)
=
(R, L_{mathrm{leg}}-s, 0),
qquad
0le sle L_{mathrm{leg}}.
]

Equivalent translated or rotated parameterizations are valid. Implementations should test continuity of position and tangent at segment joins.

---

## 12. Verification boundary

This tri-space loop is subordinate to the Jarvis-X candidate-first transaction law.

The system may propose geometry, material and latent-state updates, but authoritative state is changed only after:

1. numerical convergence checks;
2. geometry-dimensional checks;
3. target/evidence checks;
4. resource-policy checks;
5. external verification when claims concern the physical world;
6. atomic commit, otherwise rollback.

Fixed-point convergence alone is never promoted to a claim of physical correctness.

---

## 13. Implementation roadmap

A production-oriented adapter should separate modules for:

```text
SDF decoder
 -> smooth occupancy / narrow-band opacity
 -> differentiable camera + renderer
 -> material BRDF
 -> vision encoder
 -> fusion loss
 -> bounded optimizer
 -> fixed-point / contractivity monitor
 -> verification receipt
 -> candidate commit/rollback
```

The preferred execution order remains:

```text
Working -> Robust -> Portable -> Elegant -> Advanced
```
