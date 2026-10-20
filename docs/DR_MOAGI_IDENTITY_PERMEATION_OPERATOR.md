# Dr Moagi Identity-Permeation Operator on the 11 x 6 x 4 3-Torus

**Status:** executable finite reference contract  
**Domain:** Lambda = Z_11 x Z_6 x Z_4  
**Cardinality:** |Lambda| = 264

## Canonical statement

The requested terminal operator is

\[
\boxed{\mathfrak M = \mathrm{id}_{\Lambda}}
\]

with the associated invariants

\[
\mathfrak M \circ \mathfrak M = \mathfrak M,
\]

\[
\forall p\in\Lambda:\quad C^*(p)=\mathfrak M[C^*](p),
\]

and periodic closure on

\[
\Lambda=\mathbb Z_{11}\times\mathbb Z_6\times\mathbb Z_4.
\]

The implementation in src/jarvisx/identity_permeation.py materializes the
entire finite domain because it contains only 264 cells and therefore does not
make a virtual-scale or throughput claim.

## Mathematical normalization of the gradient condition

The literal pair

\[
\nabla\mathfrak M=0,
\qquad
\mathfrak M=\mathrm{id}
\]

is not compatible if \(\mathfrak M\) is interpreted as a differentiable
coordinate-valued map on a non-trivial continuous torus: the identity map has
Jacobian \(I\), not zero.

The executable invariant therefore measures the identity defect

\[
\delta_{\mathfrak M}[C]
=
\mathfrak M[C]-C
\]

and requires

\[
\boxed{
\delta_{\mathfrak M}=0,
\qquad
\nabla\delta_{\mathfrak M}=0.
}
\]

On the discrete torus the gradient is the periodic forward difference in all
three axes. This preserves the intended conclusion while keeping the contract
mathematically falsifiable.

## Permeation predicate

For this finite contract,

\[
\mathfrak M\ \text{permeates}\ \Lambda
\Longleftrightarrow
\mathfrak M=\mathrm{id}_{\Lambda}.
\]

The verifier does not infer identity from idempotence alone. A zero projection,
for example, satisfies

\[
P\circ P=P
\]

but fails the pointwise fixed-point requirement for a nonzero field and
therefore does not permeate the lattice.

The executable acceptance predicate is

\[
\begin{aligned}
V_{\rm permeate}={}&
V_{\rm periodic}
\land
(r_{\rm idem}\le\epsilon)
\land
(r_{\rm fp}\le\epsilon)\\
&\land
(r_{\nabla\delta}\le\epsilon)
\land
(r_{\rm circ}\le\epsilon).
\end{aligned}
\]

For the canonical identity operator the default tolerance is exactly zero.

## Circulation preservation

Every fundamental generator loop is enumerated:

- 24 loops in the x homology class;
- 44 loops in the y homology class;
- 66 loops in the z homology class.

For unit lattice edge length, the reference discrete circulation check is the
cycle sum

\[
I_\gamma(C)=\sum_{p\in\gamma}C(p).
\]

The verifier requires

\[
\max_\gamma
\left|
I_\gamma(\mathfrak M[C])-I_\gamma(C)
\right|
\le\epsilon.
\]

For \(\mathfrak M=\mathrm{id}\), every cycle is preserved exactly.

## Runtime surface

~~~python
from jarvisx.identity_permeation import (
    IdentityPermeationVerifier,
    TorusLattice3D,
    canonical_field,
    identity_permeation,
)

lattice = TorusLattice3D()
field = canonical_field()
receipt = IdentityPermeationVerifier(lattice).verify(
    identity_permeation,
    field,
)

assert receipt.permeates
~~~

The machine-readable receipt reports shape, cells checked, idempotence residual,
fixed-point residual, identity-defect-gradient residual, circulation residual,
periodic closure and the final permeation verdict.

## Architectural role

This operator is a terminal fixed-point identity profile, not a replacement for
the adaptive Jarvis-X transition law. It can be used as a conformance boundary
or terminal state condition after an inward/refinement process:

~~~text
candidate dynamics
  -> fixed-point convergence
  -> identity-permeation verification on Lambda
  -> verified terminal profile
~~~

It does not authorize adaptive state to bypass the repository-wide
candidate-first verification and commit/rollback boundary.

## Verification

Focused tests in tests/test_identity_permeation.py cover exact geometry,
modular wrapping, idempotence, pointwise fixed-point closure, zero defect
gradient, all fundamental-cycle sums, rejection of the false implication
"idempotent therefore identity", malformed/non-finite input and geometry lock.
