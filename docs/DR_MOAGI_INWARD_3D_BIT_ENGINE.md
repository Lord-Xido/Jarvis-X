# Dr Moagi inward-looped 3D bit engine

This document specifies the bounded discrete reference implemented in
`src/jarvisx/dr_moagi_inward_bit_engine.py`.

The engine is a deterministic research recurrence over a finite cubic bit field.
It is not a claim of lossless compression, universal convergence, physical
resonance, or unrestricted self-modification.

## 1. State and recurrence

The complete dynamical state is

```text
S_t = (X_t, Z_t, Omega_t)
```

where:

- `X_t[x,y,z]` is a fixed-width source/state word;
- `Z_t[x,y,z]` is a smaller latent word;
- `Omega_t[x,y,z]` is bounded recursive reconstruction-error memory.

One synchronous cycle is

```text
X_t
-> E
-> Z_t
-> C_6N
-> Z~_t
-> D
-> X_hat_t
-> Omega_t+1
-> bounded inward feedback
-> X_t+1.
```

The executable fixed-point condition is therefore

```text
S* = M(S*)
```

rather than only `X* = M(X*)`.

## 2. Tie-neutral bit encoding

Each source word is partitioned into `latent_bits` contiguous groups. One latent
bit summarizes each group.

Strict majority is used when the group contains more zeros than ones or more
ones than zeros. For an even-width exact tie, the first source bit in that group
resolves the tie.

For a uniformly distributed four-bit group this produces exactly eight latent
zeros and eight latent ones over the complete 16-word domain. This removes the
one-bias caused by the earlier `ones >= width/2` rule.

The decoder expands a latent bit to an all-zero or all-one source group. For this
codec,

```text
E(D(z)) = z
```

for every valid latent word. Consequently latent-cycle loss is an algebraic
identity and is reported separately from the nontrivial six-neighbour coupling
loss.

## 3. Six-neighbour latent coupling

For latent bit `b` at lattice point `p`, map bits to spins in `{-1,+1}` and
compute

```text
h_b(p) = (1-alpha) s_b(p)
       + alpha * mean_{q in N6(p)} s_b(q).
```

The coupled bit is one when `h_b(p) >= 0` and zero otherwise.

Every cycle reads a frozen latent field and writes a new complete coupled field;
partial writes are never fed back during the same cycle.

## 4. Contractive error memory

The previous pure rotate/XOR memory could perpetually remix historical parity.
The refined reference separates retention from capture:

```text
Omega_rot     = ROTL_1(Omega_t)
Omega_keep    = Omega_rot & M_retention
error_t       = X_t XOR X_hat_t
error_capture = error_t & M_capture
Omega_t+1     = Omega_keep XOR error_capture.
```

In the absence of new reconstruction error, masking cannot increase the Hamming
weight of memory. This is a contractive unforced-memory property; it is not a
proof that the complete forced recurrence converges.

## 5. Precomputed deterministic masks

Reconstruction replacement, memory injection, memory retention and error capture
use deterministic BLAKE2b-ranked bit masks. These masks depend only on immutable
configuration and coordinate data, so the implementation computes them once at
materialization time instead of hashing and sorting on every recurrence step.

The hot path therefore contains primarily integer bit operations and the 3D
six-neighbour coupling.

## 6. Inward feedback

For each coordinate, the next source/state word is constructed as

```text
candidate = preserve_unselected(X_t)
candidate |= selected_reconstruction(X_hat_t)
candidate ^= selected_memory(Omega_t+1)
X_t+1 = candidate & full_width_mask.
```

`beta` controls the reconstruction replacement fraction and `omega_gain` controls
the memory-injection fraction.

## 7. Metrics

The reference separates metrics that were previously conflated:

```text
L_local  = Hamming(X_t, X_hat_t) / source_bits
L_anchor = Hamming(X_0, X_hat_t) / source_bits
D_anchor = Hamming(X_0, X_t+1) / source_bits
L_cycle  = Hamming(Z~_t, E(D(Z~_t))) / latent_bits
L_couple = Hamming(Z_t, Z~_t) / latent_bits
Delta_X  = Hamming(X_t, X_t+1) / source_bits
Delta_O  = Hamming(Omega_t, Omega_t+1) / source_bits.
```

The full-state gap is

```text
Delta_S = (dX + dZ + dOmega)
        / (2 * total_source_bits + total_latent_bits).
```

A fixed point is declared only when `Delta_S <= tolerance`.

## 8. FMDR bridge

`bitplane_field(bit, latent=False, spins=True)` exposes any source or latent
bitplane as a scalar 3D field with values in `{-1,+1}` by default. That field can
be passed directly into the Fourier-Markov-diffusion-resonance reference in
`dr_moagi_fmdr.py`.

The combined observation path is therefore

```text
bit recurrence
-> selected 3D bitplane
-> spatial Fourier modes
-> dominant-mode Markov transitions
-> diffusion attenuation
-> temporal resonance/coherence
-> bounded FMDR proposal.
```

FMDR remains an analysis/control layer. It does not bypass the bit engine's
finite-width projection or Jarvis-X transaction and validation boundaries.

## 9. Required invariants

The focused tests require:

1. exact unbiased tie resolution over the complete four-bit fixture domain;
2. exact latent cycle identity for the deterministic codec;
3. deterministic precomputed feedback masks with declared cardinality;
4. non-increasing unforced Omega Hamming weight after retention masking;
5. fixed-point detection over source, latent and memory state;
6. separated local reconstruction, anchor reconstruction and anchor drift metrics;
7. valid state and latent bitplane export for FMDR analysis;
8. deterministic bounded recurrence for identical configuration and seed;
9. malformed configuration rejection.

## 10. Boundary

The default recurrence may approach an attractor or limit cycle rather than an
exact fixed point. Contractive memory in the unforced case does not imply global
contraction of the coupled codec-memory-feedback map. Any convergence claim must
be established for a declared configuration with measured or analytical
evidence.
