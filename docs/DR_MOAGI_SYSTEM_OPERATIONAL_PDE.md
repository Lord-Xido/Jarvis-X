# Dr Moagi System Operational Mathematical Equation

## Status

This document defines a **bounded mathematical specification candidate** for the
Dr Moagi `1000^3 -> R^3 -> 1000^3` geometric engine.

Repository classification: **specification**.

It belongs to Phase 4, bounded model and mechanics adaptation, of the canonical
Jarvis-X systems consolidation programme. It does not replace the transactional
VM, the canonical sparse spatial substrate, the existing C++ autoencoder, or the
billion-field reference implementation.

The equation below is operational only after its state types, discretization,
boundary conditions, stochastic protocol, resource limits, verifier, and commit
boundary are fixed. A logical `1000^3` extent does not imply dense resident
allocation.

---

## 1. Logical and physical state

Let

\[
\Omega_h = \{0,1,\ldots,999\}^3
\]

be the logical voxel lattice. It contains

\[
|\Omega_h| = 1000^3 = 1,000,000,000
\]

addressable coordinates.

The physical runtime operates on an explicitly bounded active support

\[
\mathcal A_t \subseteq \Omega_h,
\qquad
|\mathcal A_t| \le M_{\max}.
\]

For every active coordinate \(x_i\in\mathcal A_t\), define

\[
S_i(t)\in\mathbb R^3,
\qquad
V_i(t)=\dot S_i(t)\in\mathbb R^3.
\]

The authoritative machine state is

\[
X_t=(S_t,V_t,\mathcal A_t,\Theta,\omega_t),
\]

where \(\Theta\) contains immutable run parameters and \(\omega_t\) is the
explicit random-generator state. A stochastic run is replayable only when the
seed, generator algorithm, draw order, and generator state are part of the
receipt.

---

## 2. Canonical DMSO-3D evolution law

The compact symbolic form is

\[
\boxed{
\ddot S
=
\alpha A[S]S
+
\beta\bigl(D_{\theta^*}(E_{\theta^*}[S])-S\bigr)
-
\gamma\dot S
+
\sigma_{\mathrm n}\Xi
-
\lambda g_R[S]
}
\]

where \(g_R[S]\in\partial R[S]\) is a gradient or subgradient of the chosen
regularizer.

The rigorous continuous-time stochastic form is

\[
\boxed{
\begin{aligned}
dS_t &= V_t\,dt,\\
dV_t &=
\left[
\alpha F_{\mathrm{att}}(S_t)
+
\beta\bigl(\widehat S_t-S_t\bigr)
-
\gamma V_t
-
\lambda g_R(S_t)
\right]dt
+
\sigma_{\mathrm n}\,dW_t,
\end{aligned}
}
\]

with

\[
\widehat S_t = D_{\theta^*}(E_{\theta^*}[S_t]).
\]

The symbol \(\sigma_{\mathrm n}\) is reserved for noise amplitude. Decoder
width is denoted \(\ell\) to avoid a collision between two unrelated
parameters.

Setting \(\sigma_{\mathrm n}=0\) gives the deterministic ODE.

---

## 3. Attractive attention force

For the force to pull active states together, use the displacement from state
\(i\) toward state \(j\):

\[
\boxed{
F_{\mathrm{att},i}(S)
=
\sum_{j\in\mathcal N(i)}
w_{ij}
\frac{S_j-S_i}
{\left(\|S_j-S_i\|^2+\varepsilon^2\right)^2}
}
\]

with

\[
w_{ij}=w_{ji}\ge0,
\qquad
\varepsilon>0.
\]

Using \(S_i-S_j\) with a positive coefficient produces repulsion, not
attraction. The softening length \(\varepsilon\) prevents a singular division
at coincident states.

The neighbourhood \(\mathcal N(i)\) must be bounded. A literal all-pairs
calculation costs \(O(M^2)\) interactions per step and is not an admissible
implementation for \(M=10^9\). Permitted approximations include:

- fixed-radius or k-nearest-neighbour graphs;
- block-local interactions;
- tree codes or fast multipole methods;
- convolutional approximations on regular active blocks.

The selected approximation, error tolerance, and maximum work must be recorded
in the run configuration.

---

## 4. Three-coordinate encoder

Let \(q_i(S)\ge0\) be an occupancy or density weight at logical coordinate
\(x_i\). Define total mass

\[
m(S)=\sum_{i\in\mathcal A}q_i(S).
\]

For \(m(S)>0\), the centroid encoder is

\[
\boxed{
E[S]
=
z
=
\frac{\sum_{i\in\mathcal A}x_i\,q_i(S)}
{\sum_{i\in\mathcal A}q_i(S)}
\in\mathbb R^3.
}
\]

For \(m(S)=0\), the encoder must fail closed or return an explicitly declared
empty-state code. It must not divide by zero.

This encoder retains only a weighted centroid. It is permutation invariant and
translation covariant, but it discards shape, topology, orientation, scale, and
multimodal structure unless those properties are encoded separately.

---

## 5. Gaussian decoder and typing boundary

The Gaussian decoder produces a scalar occupancy field:

\[
\boxed{
\widehat q_z(x)
=
K_{\ell}(x-z)
=
\frac{1}{(2\pi\ell^2)^{3/2}}
\exp\left(-\frac{\|x-z\|^2}{2\ell^2}\right),
\qquad \ell>0.
}
\]

Therefore its exact type is

\[
D_{\ell}:\mathbb R^3\rightarrow\mathbb R^{\Omega_h},
\]

not automatically

\[
D_{\ell}:\mathbb R^3\rightarrow(\mathbb R^3)^{\Omega_h}.
\]

A vector-state implementation must define an additional lifting map
\(L\), for example

\[
\widehat S(x)=L\bigl(x,z,\widehat q_z(x)\bigr),
\]

and test that the lifted field has the same shape and units as \(S\). The
lifting map is part of the decoder contract; it may not be left implicit.

The distributional statement

\[
K_{\ell}(x-z)\rightharpoonup\delta^{(3)}(x-z)
\quad\text{as}\quad \ell\to0
\]

is a weak-limit statement. A Dirac delta is not a finite floating-point voxel
array.

---

## 6. Reconstruction identity boundary

A three-coordinate centroid cannot losslessly encode an arbitrary billion-voxel
field. The identity

\[
D\circ E = I
\]

can hold only on a restricted decoder family such as

\[
\mathcal M_{\ell}
=
\{K_{\ell}(\cdot-z):z\in\Omega\}
\]

or on another explicitly defined three-dimensional manifold.

The correct fixed-manifold claim is

\[
\boxed{
(D\circ E)[q]=q
\quad\text{for}\quad q\in\mathcal M_{\ell},
}
\]

subject to boundary truncation and numerical tolerance.

For an arbitrary field, the three floats are a centroid descriptor, not a
lossless compressed representation.

---

## 7. Regularization

For the stated sparsity regularizer

\[
R[S]=\|S\|_1,
\]

the derivative is a subgradient:

\[
(g_R(S))_{ic}
\in
\begin{cases}
\{+1\},&S_{ic}>0,\\
[-1,+1],&S_{ic}=0,\\
\{-1\},&S_{ic}<0.
\end{cases}
\]

A deterministic implementation must choose and document a value at zero,
usually zero, or use a smooth approximation such as

\[
R_{\delta}[S]
=
\sum_{i,c}\sqrt{S_{ic}^2+\delta^2}.
\]

---

## 8. Variational boundary

A conservative force may be derived from a potential \(U[S]\) through

\[
F_{\mathrm c}[S]=-\nabla_S U[S].
\]

Damping and white noise are not produced by an ordinary conservative
Euler-Lagrange action. Damping is added through a Rayleigh dissipation function

\[
\mathcal Q(V)=\frac{\gamma}{2}\|V\|_F^2,
\]

and noise is an external stochastic forcing term.

The simple manifold drift

\[
\beta(D(E(S))-S)
\]

is not generally the negative gradient of

\[
\frac{\beta}{2}\|D(E(S))-S\|_F^2.
\]

The exact gradient of that reconstruction energy contains the Jacobian-adjoint
factor

\[
\bigl(J_{D\circ E}(S)-I\bigr)^T.
\]

Consequently this specification treats the simple manifold term as a
phenomenological restoration drift unless an energy-consistent variant is
selected explicitly.

---

## 9. Energy statement

Monotone energy decrease is not an unconditional invariant of the master
equation.

It can be asserted for a deterministic conservative variant

\[
\ddot S=-\nabla U(S)-\gamma\dot S,
\qquad
\gamma\ge0,
\]

where

\[
H(S,V)=\frac12\|V\|_F^2+U(S)
\]

satisfies

\[
\boxed{
\frac{dH}{dt}=-\gamma\|V\|_F^2\le0.
}
\]

When \(\sigma_{\mathrm n}>0\), stochastic work enters the system and sample-path
energy is not monotonically decreasing. The runtime must report measured energy
or Lyapunov values rather than assume monotonicity.

---

## 10. Fixed point and collapse

A deterministic equilibrium \(S^*\) satisfies

\[
V^*=0
\]

and

\[
\boxed{
\alpha F_{\mathrm{att}}(S^*)
+
\beta\bigl(D(E(S^*))-S^*\bigr)
-
\lambda g_R(S^*)
=0.
}
\]

The additional reconstruction condition

\[
D(E(S^*))=S^*
\]

is valid only when \(S^*\) belongs to the declared decoder manifold. It does not
follow automatically from zero acceleration.

A centered Dirac delta is one possible distributional limit under a compatible
encoder, decoder, boundary, regularizer, and zero-noise protocol. It is not the
unique or universal solution of the master equation.

---

## 11. Bounded discrete integrator

For a fixed active support and a candidate transaction, a semi-implicit Euler
step is

\[
\begin{aligned}
a_i^k &=
\alpha F_{\mathrm{att},i}(S^k)
+
\beta(\widehat S_i^k-S_i^k)
-
\gamma V_i^k
-
\lambda g_R(S_i^k)
+
\sigma_{\mathrm n}\xi_i^k,\\
V_i^{k+1} &= V_i^k+\Delta t\,a_i^k,\\
S_i^{k+1} &= S_i^k+\Delta t\,V_i^{k+1}.
\end{aligned}
\]

Reference pseudocode:

```text
checkpoint authoritative state and PRNG state
construct bounded active support and neighbourhood graph

for step in 0 .. max_steps - 1:
    z = weighted_centroid(logical_coordinates, occupancy)
    q_hat = gaussian_decoder(z, decoder_width)
    S_hat = lift(q_hat, z)

    for i in active_support:
        F_att = softened_attractive_neighbour_force(i)
        F_man = beta * (S_hat[i] - S[i])
        F_reg = -lambda * l1_subgradient(S[i])
        noise = sigma_noise * deterministic_normal_draw(prng)
        acceleration[i] = alpha * F_att + F_man - gamma * V[i] + F_reg + noise

    V_candidate = V + dt * acceleration
    S_candidate = S + dt * V_candidate

    verify finite values, support bound, force bound, residuals, and budgets
    reject and restore checkpoint on any failed verification

commit S_candidate, V_candidate, and the advanced PRNG state atomically
emit replayable receipt
```

An explicit or semi-implicit method still requires a tested step-size bound.
No single \(\Delta t\) is stable for every parameter set or neighbourhood graph.

---

## 12. Resource accounting

A dense state containing one billion three-component `float32` values requires

\[
10^9\times3\times4=12,000,000,000\text{ bytes}
\]

for \(S\) alone: 12 GB in decimal units, approximately 11.18 GiB.

Storing \(S\), \(V\), and acceleration densely requires at least 36 GB before
indices, occupancy, decoder buffers, neighbourhoods, receipts, and allocator
overhead.

Therefore:

- `4 GB -> 12 bytes` is not the raw size of the stated three-channel field;
- a 12-byte latent is only the centroid descriptor unless additional model or
  side information is retained;
- the canonical implementation must remain sparse, tiled, streamed, or
  otherwise bounded;
- virtual extent, resident bytes, materialized voxels, and operations per step
  must be reported separately.

---

## 13. Transactional authority contract

This mechanics layer must not mutate authoritative Jarvis-X state directly.
Every run is a candidate transaction:

\[
X_t
\xrightarrow{\text{propose}}
\widetilde X_{t+1}
\xrightarrow{\text{verify}}
\begin{cases}
X_{t+1}=\widetilde X_{t+1},&\text{accept},\\
X_{t+1}=X_t,&\text{reject}.
\end{cases}
\]

Verification must include at least:

1. finite state, velocity, force, latent, and metrics;
2. active-support and neighbour-count bounds;
3. force, velocity, and displacement bounds;
4. deterministic replay in zero-noise mode;
5. seeded replay in stochastic mode;
6. reconstruction residual and equilibrium residual;
7. elapsed-step and operation budgets;
8. no persistence side effects before commit.

The receipt must bind:

- specification version;
- input-state digest;
- parameter and boundary-condition digest;
- active-support and neighbourhood digest;
- PRNG algorithm, seed, and pre/post state digest;
- candidate-state digest;
- verifier results;
- commit or rollback outcome;
- prior-receipt digest.

---

## 14. Required conformance tests

A conforming implementation must test:

- empty occupancy fails closed without division by zero;
- one occupied voxel encodes to its coordinate;
- symmetric occupied voxels encode to their centroid;
- attention force is attractive and pairwise antisymmetric for symmetric weights;
- coincident states remain finite because of \(\varepsilon\);
- decoder mass and boundary truncation are measured;
- `D(E(q))` reproduces members of the declared decoder family within tolerance;
- arbitrary non-Gaussian fields are not falsely reported as exactly reconstructed;
- deterministic mode replays byte-for-byte under canonical serialization;
- stochastic mode replays from the recorded PRNG state;
- a failed verifier restores state and PRNG state;
- support, operation, velocity, force, and displacement limits reject before commit;
- dense billion-voxel allocation is not required by the reference implementation.

---

## 15. Canonical glyph

With all terms interpreted through the contracts above:

\[
\boxed{
\mathfrak D_m:
\quad
\ddot S
=
\alpha A[S]S
+
\beta(DE[S]-S)
-
\gamma\dot S
+
\sigma_{\mathrm n}\Xi
-
\lambda g_R[S].
}
\]

Read operationally:

> Propose a bounded state transition from attractive geometric coupling,
> restoration toward a declared three-coordinate decoder manifold, damping,
> optional replayable stochastic exploration, and explicit regularization;
> then verify and atomically commit or roll back.

---

## Capability boundary

This specification defines a reduced-order centroid-and-kernel dynamics model.
It does not establish lossless billion-voxel compression, a universal
three-dimensional autoencoder, monotone stochastic energy, a unique delta
attractor, artificial general intelligence, consciousness, or a production
implementation.

A future `dm_engine.cpp` may claim conformance only after it implements the
stated type contracts, bounded neighbourhood protocol, deterministic or seeded
stochastic protocol, transactional verifier, resource accounting, receipts, and
conformance tests.