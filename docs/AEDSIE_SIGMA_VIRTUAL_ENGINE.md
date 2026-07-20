# AEDSIE-Sigma End-to-End Virtual Engine

## Status

Executable deterministic reference model for the Auto-Encoding and Decoding
Electromagnetic Signalling Intelligence Engine.  The implementation turns the
corrected mathematical architecture into one auto-executing virtual pipeline.

It is a CPU reference engine for semantics, testing, provenance, and bounded
self-optimisation.  It is not a measured 1 GSPS radio, FPGA bitstream, lossless
universal codec, or deployment benchmark.

## Run

```bash
jarvisx aedsie 4
```

Disable the inward mechanics proposal:

```bash
jarvisx aedsie 4 --no-inward
```

The command emits JSON containing the final state metrics, expert routing,
angle estimate, mechanics decision, SHA3 state seal, ledger head, and complete
program trace.

## Virtual geometry

The default signal field is:

```text
4 antennas x 8 frequency bins x 4 temporal slices x 3 vector channels
```

The vector channels are:

```text
(real I, imaginary Q, magnitude)
```

The default latent state has 16 dimensions.  All dimensions are deliberately
small enough for deterministic CI execution while preserving the mathematical
contract needed by larger CPU, GPU, or FPGA backends.

## Auto-executing program

Every committed cycle executes:

```text
ACQUIRE_RF
  -> DDC_CHANNELIZE
  -> TENSORIZE_3D
  -> DR_MOAGI_OPERATOR
  -> ENCODE_RESIDUAL
  -> DECODE_RESIDUAL
  -> COMPARE
  -> UPDATE_OMEGA
  -> INWARD_SHADOW
  -> PROJECT_MANIFOLD
  -> ROUTE_EXPERTS
  -> ESTIMATE_AOA
  -> SEAL_SHA3
  -> COMMIT
```

The program is closed over an explicit state:

\[
\Sigma_t=(X_t,\phi_t,Z_t,\widehat\phi_t,E_t,\Omega_t,g_t,
          \mathcal M_t,J_t).
\]

## Deterministic RF source

The reference source synthesises two complex emitters with antenna-dependent
phase slopes and deterministic low-amplitude interference.  No wall-clock
value, system entropy, or unseeded random number enters the execution path.

For antenna \(a\) and sample \(n\):

\[
x_{a,n}=e^{j2\pi(f_1n+p_1a)}+
0.55e^{j2\pi(f_2n+p_2a+q t)}+\epsilon_{a,n}.
\]

This source is a repeatable test vector, not a substitute for an ADC interface.

## Channelisation

The engine performs deterministic digital downconversion, Hann windowing, and
a direct discrete Fourier transform:

\[
X_a[k]=\frac1N\sum_{n=0}^{N-1}
 x_a[n]w[n]e^{-j2\pi(f_c+k/N)n}.
\]

A production backend may replace the direct transform with FFT hardware while
preserving the numerical and state-transition tests.

## Dr Moagi differential operator

The finite-difference feature operator is:

\[
\mathcal M_{3d}(\phi)=\operatorname{swish}\left(
\alpha\nabla^2\phi+
\beta\nabla\times\phi+
\gamma\nabla(\nabla\cdot\phi)+
\delta(K_{local}\phi-\phi)
\right).
\]

Periodic boundaries make the reference stencil fully specified.  The operator
is a learned-style differential feature transform; it is not labelled an exact
Helmholtz-Hodge decomposition.

## Residual autoencoder

The encoder preserves a per-channel global baseband skip:

\[
b_c=\frac1{|V|}\sum_{v\in V}\phi_{v,c},
\qquad r=\operatorname{vec}(\phi-b).
\]

A finite orthonormal cosine basis \(Q\) produces:

\[
Z=Qr.
\]

The tied decoder is the exact adjoint of that projection:

\[
\widehat\phi=b+Q^T Z.
\]

Therefore:

\[
\widehat\phi=b+Q^TQr,
\]

which is a projection onto the declared latent subspace, not a claim that
\(Q^TQ=I\) over the complete field space.

The reconstruction residual is:

\[
E_t=\widehat\phi_t-\phi_t.
\]

## Field and Omega update

The persistent correction state evolves as:

\[
\Omega_{t+1}=\rho_\Omega\Omega_t+\eta_\Omega E_t.
\]

The bounded explicit field proposal is:

\[
\widetilde\phi_{t+1}=\phi_t+\Delta t\left[
\mathcal M_{3d}(\phi_t)+\lambda E_t+\Omega_{t+1}
\right].
\]

The implementation admits mechanics only when the conservative load proxy
satisfies:

\[
\Delta t(6|\alpha|+|\lambda|)\leq0.90.
\]

This is a reference policy bound for the six-neighbour explicit stencil, not a
universal nonlinear PDE stability theorem.

## Positive manifold metric

The virtual Riemannian metric is represented by a positive diagonal tensor:

\[
g_{i,t+1}=\max\left(\varepsilon,
(1-\eta_g)g_{i,t}+\eta_g(1+|Z_{i,t}|)
\right).
\]

This guarantees:

\[
g_t\succ0.
\]

The reference implementation does not claim Lorentzian geometry or geodesic
closure merely because recurrent state is used.

## Expert routing

Nine deterministic expert projections share one class-logit space.  Routing is:

\[
\omega_i=
\frac{\exp(\operatorname{cos}(Z,e_i)/\tau)}
{\sum_j\exp(\operatorname{cos}(Z,e_j)/\tau)},
\qquad \tau=0.80.
\]

The fused logits are:

\[
\ell_c=\sum_i\omega_i\langle W_{i,c},Z\rangle.
\]

A final softmax produces the virtual classification and confidence.  The
experts are deterministic reference operators, not empirical SOTA claims.

## Angle-of-arrival projection

The strongest frequency bin is selected from aggregate antenna magnitude.  The
mean adjacent-antenna phase difference \(\Delta\varphi\) is projected under a
half-wavelength spacing model:

\[
\widehat\theta=\arcsin\left(
\operatorname{clip}(\Delta\varphi/\pi,-1,1)
\right).
\]

This is a deterministic narrowband reference estimator.  Real arrays require
calibration, ambiguity handling, geometry metadata, and measured uncertainty.

## Bounded inward turn

The mechanics state is:

\[
\mathcal M_t=(\alpha,\beta,\gamma,\delta,\lambda,\Delta t,v_t).
\]

Each cycle generates at most one declared candidate.  Baseline and candidate
execute from the same field, reconstruction, and Omega snapshot:

\[
S_{base}=F_{\mathcal M_t}(S_t),
\qquad
S_{shadow}=F_{\mathcal M'_t}(S_t).
\]

The measured objective is:

\[
C(S)=
\operatorname{MSE}(S,\phi_{obs})+
0.2\operatorname{MSE}(S,\widehat\phi)+
0.001\|S\|_2^2.
\]

A candidate is committed only when:

\[
C(S_{shadow})<C(S_{base}),
\]

its state is finite, its mechanics pass the stability gate, and:

\[
\text{analysis share}=\frac{1}{1+1}=0.5
\leq B_{analysis}.
\]

Rejected candidates cannot mutate committed state.  Every accepted mechanics
version receives a parent-linked SHA3 manifest.

## SHA3 provenance

The authoritative state is canonically quantised before hashing.  The state
seal contains:

```text
logical cycle
canonical field
latent vector
positive metric
mechanics version
prediction
angle estimate
```

The ledger transition is:

\[
J_{t+1}=\operatorname{SHA3-256}
\left(J_t\parallel\operatorname{Canon}(\Sigma_{t+1})\right).
\]

No wall-clock value or external randomness enters the hash.  The chain is
tamper-evident relative to its trusted head; it is not described as physically
immutable without external anchoring.

## Output contract

One cycle returns:

```text
cycle
reconstruction_mse
field_energy
latent_energy
predicted_class
confidence
aoa_degrees
routing_weights
metric_min / metric_max
mechanics
inward decision
state_hash
ledger_head
program_trace
```

## Verified invariants

The test suite checks:

1. identical initial state and inputs produce identical state and ledger hashes;
2. every declared program stage executes exactly once per committed cycle;
3. routing weights sum to one;
4. metric values remain strictly positive;
5. ledger heads change once per commit and remain parent-linked;
6. an accepted inward candidate has lower measured cost than the baseline;
7. disabling inward execution leaves mechanics unchanged;
8. autoencoder output is finite and shape preserving;
9. invalid FFT and unstable mechanics configurations are rejected.

## Extension boundary

The reference interfaces can be replaced independently:

```text
SyntheticRFSource -> ADC / recorded IQ source
Direct DFT        -> FFT / FPGA channelizer
Python stencil    -> SIMD / CUDA / FPGA kernel
Cosine projection -> trained or invertible encoder
Reference experts -> calibrated task models
Local ledger      -> signed durable transparency log
```

Each replacement must preserve the transaction, deterministic replay, policy,
and provenance contracts before promotion.
