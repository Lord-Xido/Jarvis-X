# MM3D-AED-BCE-Ω⁴-G50T-OPT Operational Audit

## Status

The submitted architecture is preserved as an operational reference kernel in
`src/jarvisx/mm3d_omega4.py`. The implementation keeps the intended closed loop
while correcting dimensional contradictions, unbounded allocations, invalid
rendering shapes, nondeterministic ledger semantics, and the broken distributed
merge path.

## Canonical loop

\[
\Xi_{t+1}
=
\Pi_{\Lambda}
\left[
\mathcal D_{\Phi}
\left(
\mathcal E_{\Phi}(\Xi_t)
+
\Phi_{QAS}(Z_t)
\right)
\right],
\]

followed by an immutable Ω record and bounded Θ projections.

The current `PhiQAS` class is explicitly classical. It performs deterministic,
bounded latent candidate exploration and does not claim quantum execution.

## Defects in the submitted program

### Voxel width

The declared voxel width was 384 bits or 48 bytes, but the four `int8` feature
arrays contained 336 elements. With the float and flags, `to_bytes()` emitted
342 bytes. The operational layout interprets 128/96/64/48 as feature bits and
stores 16/12/8/6 bytes respectively, followed by a float32 and two flag bytes:

\[
16+12+8+6+4+1+1=48\text{ bytes}.
\]

### Raw allocation floor

Before Python-object overhead, the original declarations required approximately
38.02 GiB:

| Structure | Raw size |
|---|---:|
| Serialized 256³ voxel payload at 342 bytes | 5.34 GiB |
| Python object-pointer cube | 0.125 GiB |
| QCA state | 0.016 GiB |
| Low-rank `L` | 0.25 GiB |
| Dense `S` | 16 GiB |
| Codebook | 0.25 GiB |
| Encoder projection | 8 GiB |
| Decoder projection | 8 GiB |
| Temporary decoded latent volume | 0.031 GiB |

The actual object-per-voxel implementation would consume substantially more and
would require 16,777,216 Python-level constructor iterations.

The corrected Xi-cube uses one contiguous 48-byte NumPy structured dtype. The
reference configuration has a hard allocation guard, reports actual allocated
bytes, and rejects an oversized profile before allocation.

### Encoder and decoder dimensions

The submitted encoder produced 32³ scalar values and reshaped them to
`(32,32,32,1)` while comparing each scalar against 256-dimensional codebook
vectors. The decoder then expanded the code to 32³×256 values but multiplied it
by a matrix expecting only 32³ inputs.

The corrected kernel uses a continuous latent matrix

\[
Z\in\mathbb R^{L\times C},
\qquad L=s^3,
\]

where `C` is exactly the codebook width. Both encoder and decoder use consistent
factorized maps.

## Factorized metric

The dense sparse-correction matrix is replaced with a positive diagonal plus
low-rank factor:

\[
g=D+LL^T.
\]

The inverse action uses Woodbury:

\[
g^{-1}v
=D^{-1}v
-D^{-1}L
\left(I+L^TD^{-1}L\right)^{-1}
L^TD^{-1}v.
\]

This preserves the geometric mechanism without allocating a 65,536² dense
matrix.

## Vector quantization

Continuous latent vectors are quantized by chunked matrix distance evaluation:

\[
\|z-c_j\|^2
=
\|z\|^2+\|c_j\|^2-2z^Tc_j.
\]

The implementation avoids a Python loop over every latent cell and every
codebook entry.

## Z8 substrate

The six-neighbour Laplacian is computed with periodic `numpy.roll` operations,
removing the undeclared SciPy dependency. Every update is performed in an
integer accumulator and projected back into

\[
\mathbb Z_8=\{0,1,\ldots,7\},
\]

so the QCA state remains `uint8` rather than drifting to floating point.

## Lambda projection

The runtime enforces policy before execution and validates that the mask has the
same dimension as the decoded state. Constrained coordinates are projected to
the feasible boundary. `CodexVM` also gates the entire operation under the named
`MM3D.CYCLE` action.

## Omega chain

Ω entries use logical sequence numbers and canonical JSON hashing. The chain
uses full SHA3-256 hashes. When the retained window reaches capacity, the dropped
entry hash becomes the new verification anchor, so the retained chain remains
verifiable.

## Theta projections

All projections are shape-bounded and deterministic:

- text returns deterministic numerical token IDs;
- image repeats finite state into a configured RGB grid;
- audio derives a bounded four-tone waveform from state values;
- video rolls the deterministic image over a bounded frame count.

These are reference projections, not production generative decoders.

## Distributed execution

The submitted distributed path encoded the full input on every node, selected a
`LatentCode`, and then passed that object into `cycle()`, which expected an array.

The corrected path partitions the factorized encoder's manifold columns. Each
node computes a partial hidden projection:

\[
h_n=B_{[:,I_n]}x_{I_n},
\qquad
h=\sum_n h_n.
\]

The authoritative node then quantizes, decodes, applies Lambda, commits Omega,
and renders Theta outputs.

## Parameter accounting

`50T` is retained as a conceptual scaling target rather than a false allocation
claim. The reference result reports:

- actual allocated parameter count;
- actual allocated bytes;
- conceptual total parameter target;
- conceptual active parameter target.

At 0.5% activity:

\[
50\times10^{12}\times0.005=250\times10^9,
\]

so the correct conceptual active count is 250 billion, not 500 billion.

## Performance contract

13.7 ms is a measured target. Every result reports the actual wall-clock cycle
time and a Boolean `target_met`; no Python reference run is declared compliant
without measurement.

## Verification

The tests cover:

1. exact 48-byte voxel round-trip;
2. deterministic mod-8 QCA evolution;
3. conceptual-versus-operational parameter accounting;
4. allocation-guard rejection of the submitted dense profile;
5. deterministic cycle equality and Omega verification;
6. Lambda rejection of empty input;
7. equivalence of sequential and partitioned encoder state;
8. retained-window Omega verification;
9. VM policy, ledger, trace, CLI, and API integration.

## Invariant

```text
Conceptual scale is metadata until physical allocation and measured execution
prove otherwise.
```
