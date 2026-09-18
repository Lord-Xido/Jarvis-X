# Dr Moagi Recursive Autoencoding Equation

**Status:** Canonical research specification  
**Date:** 2026-09-18  
**Scope:** Recursive autoencoding, variational regularization, inward refinement, and fixed-point convergence  
**Related contract:** `docs/DR_MOAGI_OPERATIONAL_AUTOENCODING_EQUATION.md`

## 1. Compact canonical form

The Dr Moagi Recursive Autoencoding Equation is written compactly as

[
oxed{
M(x)
=
D_{	heta}!left(
E_{phi}!left(x-eta_x
abla_xmathcal L(x)ight)
ight)
}
quadcirclearrowleft
]

with recursive execution

[
oxed{
x_{n+1}=M(x_n),qquad
lim_{n	oinfty}M^n(x_0)=x^star
}
]

whenever (M) is contractive on the declared admissible domain.

At the fixed point,

[
oxed{
x^star=M(x^star).
}
]

The explicit negative sign is the canonical loss-minimizing form. If a runtime defines the supplied (
abla L) symbol itself as an already-negated corrective field, the original shorthand (x+
abla L) is equivalent by convention; implementations must state which convention they use.

## 2. Objective

Let

[
z=E_phi(x),
qquad
hat x=D_	heta(z).
]

The canonical reconstruction-plus-regularization objective is

[
oxed{
mathcal L(x;	heta,phi)
=
|x-hat x|_2^2
+
eta,
D_{mathrm{KL}}
!left(
q_phi(zmid x),|,p(z)
ight)
}
]

for a variational encoder, with (etage0).

For a deterministic autoencoder, the KL term may be disabled by setting (eta=0) or replaced by another explicitly declared latent regularizer.

## 3. Operational recurrence

Because (mathcal L) depends on the current reconstruction, the executable recurrence is evaluated at the current iterate:

[
oxed{
egin{aligned}
z_n &= E_phi(x_n),\
hat x_n &= D_	heta(z_n),\
mathcal L_n
&=
|x_n-hat x_n|_2^2
+
eta D_{mathrm{KL}}
!left(q_phi(zmid x_n),|,p(z)ight),\
	ilde x_n
&=
x_n-eta_x
abla_{x_n}mathcal L_n,\
x_{n+1}
&=
D_	heta(E_phi(	ilde x_n)).
end{aligned}
}
]

Thus the inward loop is

[
x_0
ightarrow
	ilde x_0
ightarrow
x_1
ightarrow
	ilde x_1
ightarrow
x_2
ightarrowcdotsightarrow
x^star.
]

This operator acts in input/reconstruction space. A latent-space variant may instead apply the corrective recurrence to (z), but the state space and gradient variable must not be conflated.

## 4. Contractive closure

The statement

[
lim_{n	oinfty}M^n(x_0)=x^star
]

is a theorem only when the recursion is contractive on a complete state space.

Let (mathcal X) be the declared admissible state space. Require

[
oxed{
|M(x)-M(y)|
le K_M|x-y|,
qquad
0le K_M<1.
}
]

Then Banach's fixed-point theorem gives

[
oxed{
orall x_0inmathcal X,qquad
M^n(x_0)ightarrow x^star
}
]

with a unique globally attractive fixed point (x^star).

A differentiable sufficient condition is

[
oxed{
sup_{xinmathcal X}
|J_M(x)|_2<1.
}
]

A spectral radius below one at a sampled point is not, by itself, a global contraction proof.

## 5. Encoder-decoder Lipschitz bound

If

[
operatorname{Lip}(E_phi)le L_E,
qquad
operatorname{Lip}(D_	heta)le L_D,
]

and the correction map

[
G(x)=x-eta_x
abla_xmathcal L(x)
]

has Lipschitz constant (L_G), then

[
oxed{
operatorname{Lip}(M)
le
L_D L_E L_G.
}
]

Therefore a sufficient global condition is

[
oxed{
L_D L_E L_G<1.
}
]

If (
ablamathcal L) is (L_
abla)-Lipschitz, a conservative norm bound is

[
L_G
le
1+eta_xL_
abla.
]

That bound alone does not produce contraction unless the encoder-decoder composition is sufficiently contractive. Stronger optimization assumptions can yield tighter bounds for (G).

## 6. Connection to the 3D equilibrium core

For the canonical 3D lattice equilibrium engine, let

[
A=
|W_{mathrm{self}}|_2
+
sum_{d=1}^{6}gamma_d|W_d|_2.
]

For

[
f(h;x)=
sigma!left(alpha(Lh+b+x)ight),
]

the driven-state contraction bound is

[
oxed{
K_flealphaell_sigma A<1.
}
]

For the autonomous inward fold

[
Phi(h)=
sigma!left(alpha((L+I)h+b)ight),
]

the safe global bound is

[
oxed{
K_Phi
le
alphaell_sigma(A+1)<1.
}
]

Hence a single sufficient scaling rule is

[
oxed{
alpha
=
rac{ho}
{ell_sigma(A+1)},
qquad
0<ho<1.
}
]

This is the preferred certification mechanism for the spatial equilibrium layer.

## 7. Conditioned inward refinement

An unconditional autonomous contraction has one global attractor and will eventually erase dependence on the initiating input.

For task-preserving recursive refinement, retain conditioning:

[
oxed{
Phi_x(h)
=
sigma!left(
alpha(Lh+eta_hh+eta_cP_xx+b)
ight).
}
]

The conditioning term does not alter contractivity with respect to (h). A sufficient state-space condition is

[
oxed{
alphaell_sigma(A+eta_h)<1.
}
]

This yields an input-dependent refined equilibrium rather than collapse toward one input-independent autonomous fixed point.

## 8. Complete recursive autoencoding cycle

The operational cycle is

[
oxed{
egin{aligned}
x_0 &= x,\
z_n &= E_phi(x_n),\
hat x_n &= D_	heta(z_n),\
mathcal L_n
&=
|x_n-hat x_n|_2^2
+
eta D_{mathrm{KL}}(q_phi(zmid x_n)|p(z)),\
	ilde x_n
&=
x_n-eta_x
abla_{x_n}mathcal L_n,\
x_{n+1}
&=
D_	heta(E_phi(	ilde x_n)),\
x^star
&=
operatorname{Fix}(M),\
hat x^star
&=
D_	heta(E_phi(x^star)).
end{aligned}
}
]

In system form:

[
oxed{
	ext{Encode}
ightarrow
	ext{Decode}
ightarrow
	ext{Measure residual/regularization}
ightarrow
	ext{Correct}
ightarrow
	ext{Re-encode}
ightarrow
	ext{Re-decode}
ightarrow
	ext{verify contraction}
ightarrow
	ext{recur}.
}
]

## 9. Implementation invariant

The runtime must preserve the distinction between three claims:

1. **The recursion is executable:** an implementation can evaluate (M).
2. **The recursion converges empirically:** a particular run reaches a numerical tolerance.
3. **The recursion is globally convergent:** a valid contraction bound (K_M<1) has been established on the declared domain.

Only the third supports the universal Banach fixed-point guarantee.

Where adaptive training changes (	heta) or (phi), the contraction certificate must be recomputed or enforced after the update. A valid implementation may use spectral normalization, projection into an admissible parameter set, or explicit operator rescaling.

## 10. Canonical shorthand

For diagrams, comments, and architecture summaries, the compact notation is

[
oxed{
M(x)
=
D_{	heta}
Big(
E_{phi}(x-eta_x
abla_xmathcal L)
Big)
circlearrowleft,
qquad
M^n(x)	o x^star,
qquad
x^star=M(x^star)
}
]

where

[
oxed{
mathcal L
=
|x-hat x|_2^2
+
eta D_{mathrm{KL}}
left(q_phi(zmid x)|p(z)ight).
}
]

The fixed-point limit is asserted as a theorem only when the declared contraction condition is satisfied.
