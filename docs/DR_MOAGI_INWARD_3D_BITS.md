# Inward Recursive 3D Bit AE/AD

## Scope

`jarvisx.dr_moagi_inward_3d_bits` is a deterministic, finite-materialization
reference for repeatedly feeding a decoded 3D bit volume back into its own encoder.
It is deliberately distinct from the anchored virtual 3D codec: the authoritative
feedback anchor is the current state `X_t`, not the original source `X_0`.

The runtime does not allocate the symbolic virtual universe and does not claim a
trained neural representation. It executes a bounded active tile.

## Recurrence

For each active 3D coordinate `p`,

\[
Z_t(p)=E(X_t(p)).
\]

The latent bit field is synchronously coupled across the six-neighbour lattice:

\[
\widetilde Z_t=C_{\alpha,\mathcal N_6}(Z_t).
\]

The decoder reconstructs a source-width bit word:

\[
\widehat X_t(p)=D(\widetilde Z_t(p)).
\]

The residual-memory plane is

\[
\Omega_{t+1}(p)=X_t(p)\oplus\widehat X_t(p).
\]

For a deterministic coordinate-specific reconstruction mask `M_beta(p)`, the
self-referential update is

\[
X_{t+1}(p)=
\left(X_t(p)\land\neg M_\beta(p)\right)
\lor
\left(\widehat X_t(p)\land M_\beta(p)\right).
\]

An optional, independently generated mask can inject selected residual-memory bits:

\[
X_{t+1}(p)\leftarrow
X_{t+1}(p)\oplus
\left(\Omega_{t+1}(p)\land M_\Omega(p)\right).
\]

The decoder output therefore does not terminate the computation:

\[
X_t\to E\to C_{3D}\to D\to\widehat X_t\to X_{t+1}\to E\to\cdots
\]

## Why this is different from the anchored codec

The anchored virtual codec uses original source bits outside the feedback mask. In
that form, majority-group reconstruction can become a one-shot projection followed
by a fixed-point check.

The inward runtime instead preserves unselected bits from `X_t`:

\[
X_{t+1}=(X_t\land\neg M)\lor(\widehat X_t\land M).
\]

Spatially changed latent majorities can therefore influence later encodes and later
neighbour interactions. The recurrence is genuinely state recursive rather than a
repeated projection against `X_0`.

## Authoritative state and atomicity

The authoritative state is the tuple

\[
S_t=(X_t,\Omega_t,Z_t).
\]

A step computes the complete candidate tuple first. Bit widths, coordinate support
and any external gate are validated before publication. Rejection leaves the prior
state untouched.

The runtime hashes the complete tuple using canonical JSON-compatible state records.
This makes replay and cycle detection independent of Python dictionary insertion
order.

## Fixed points and cycles

A zero source-state gap alone is insufficient when residual or latent state can
change future behavior. The reference therefore requires equality of the complete
authoritative tuple:

\[
S_{t+1}=S_t
\]

and the configured source-state tolerance

\[
\Delta_X\le\varepsilon.
\]

A previously observed candidate hash that differs from the current full-state hash
is classified as a non-fixed synchronous cycle. The candidate is not published.
This prevents a period-2 or longer oscillation from being reported as convergence.

## Telemetry

Each step reports:

- reconstruction loss against the original deterministic source;
- self-reconstruction loss against `X_t`;
- codec latent-cycle loss;
- 3D spatial disagreement;
- latent balance/collapse loss;
- source-state reality gap `Delta_X`;
- latent transition gap `Delta_Z`;
- raw-to-coupled latent gap;
- residual-memory density;
- changed source and latent bits;
- input and candidate authority hashes;
- commit, fixed-point and cycle status.

The current grouped majority codec satisfies `E(D(z)) = z` for valid latent words,
so its codec-cycle loss is a consistency check rather than evidence of learned cycle
quality.

## CLI

```bash
python -m jarvisx.dr_moagi_inward_3d_bits \
  --tile 8 \
  --bits 64 \
  --latent 16 \
  --iterations 32 \
  --alpha 0.65 \
  --beta 0.50 \
  --json
```

Optional residual feedback can be enabled explicitly:

```bash
python -m jarvisx.dr_moagi_inward_3d_bits --omega-feedback 0.15 --json
```

## Capability boundary

This module demonstrates deterministic recursive bit-state dynamics, spatial latent
coupling, bounded residual feedback, atomic state publication, full-state fixed-point
checks and non-fixed-cycle detection. It does not establish gradient-trained
representation learning, universal convergence, semantic intelligence, physical
allocation of the virtual address space, or superiority over conventional neural
codecs.
