# Inward Loop: analytic Fourier torus codec and replayable residual witness

## What is actually executed

`jarvisx.inward_loop_certified` evolves **N=8192** particles on a two-dimensional
torus in three-dimensional Euclidean space, embeds their positions using
16 Fourier features, reconstructs them, computes residual and finite-difference
projection metrics, and commits the arrays to a SHA-256 chain each frame.

This implementation is **an analytic encoder/decoder, not a trained neural network**.
It does not modify model weights, learn a codebook, or perform autonomous code
changes. It is a deterministic geometric baseline suitable for later integration.

## Coordinates and state

For angles `u,v` (modulo `2π`), with `R=5.2`, `r=0.85`:

`P(u,v) = ((R+r cos(v)) cos(u), r sin(v), (R+r cos(v)) sin(u))`.

The inverse at valid torus points is
`u=atan2(z,x)` and `v=atan2(y,sqrt(x²+z²)-R)`. Points on the revolution
axis are intentionally rejected because the angle `u` is undefined there.

For Fourier modes `j=1,2,3,4`, the encoder is

`phi(P) = [cos(j u)]_(j=1..4) || [sin(j u)] || [cos(j v)] || [sin(j v)]`

with layout indices `0:4, 4:8, 8:12, 12:16`. The decoder recovers
`u=atan2(z[4],z[0])`, `v=atan2(z[12],z[8])` and applies `P(u,v)`.
Because the first harmonic pairs for **both** angular coordinates survive,
`psi(phi(P))=P` on the torus up to floating point rounding.

The originally supplied example created eight modes in each of four blocks,
then discarded the final 16 features. As a consequence, its `v`-harmonics
were lost and its decoder accessed `u`-harmonics at indices intended for `v`.
This implementation retains four modes per block to obtain the advertised
16-dimensional representation.

## Projector geometry and measured invariants

`T = psi ∘ phi` is a local normal projection onto the torus (away from its
nonunique projection locus). At a point `P0` **on the surface**, the central
finite-difference Jacobian is

`J[:,k] = (T(P0+eps e_k)-T(P0-eps e_k))/(2 eps)`, `eps=1e-5`.

A smooth local torus projector has ideal eigenvalues `(1,1,0)` and
`J = I - n nᵀ`, with unit normal
`n=(cos(u)cos(v), sin(v), sin(u)cos(v))`.

Each frame reports:
- `residual_max`, `residual_mean`: Euclidean norm of `psi(phi(P_i))-P_i`.
- `residual_witness_index`: index attaining the maximum residual.
- `idempotence`: maximum norm of `T(T(P_i))-T(P_i)`.
- `eig_error`: difference between sorted real eigenvalues of J and `(1,1,0)`.
- `projector_error`: maximum absolute component of `J²-J`.
- `tangent_error`: maximum absolute component of `J-(I-n nᵀ)`.

The Jacobian is measured on an actual particle instead of at the **cloud
centroid**: the centroid lies away from the torus and can approach singular
projection locations where rank-two projector behavior is not applicable.

The particle dynamics are an angular shear field:
`du/dt=0.01`, `dv/dt=0.013+0.35 sin(u)`, integrated with a forward Euler
step and periodic wrap. This is not learning from residual feedback.

## Certificate specification

At initialization, the SHA-256 chain root commits to the schema version,
seed, particle count, timestep and geometric constants. At each frame,

`H_n = SHA256( len(H_(n-1))||H_(n-1) || len(n)||n || len(P)||P || ... || len(J)||J )`

where each component is length-prefixed, and the arrays are canonical
little-endian float64, C-order, with their shape encoded. In practice the
`_digest` function supplies each length prefix (uint64 LE), and `canonical_bytes`
includes its own shape metadata. The chain includes the full input cloud,
latent representation, reconstruction, residual vector and Jacobian.

`--verify FILE` **recomputes** all frames with the recorded inputs/configuration
and compares the entire exported manifest including every full 256-bit root.
This detects accidental or malicious modifications to a stored trace relative
to the code and configuration used for replay. This is **not** an authenticated
signature or an external proof that the claimed computation occurred. A party
controlling both the source and the trace can regenerate matching evidence.
For stronger provenance, anchor the final root outside the process or sign
it using a separately protected key, and pin the code revision and environment.

## Reproduce

```bash
python -m pip install -e '.[test]'
python -m pytest tests/test_inward_loop_certified.py -q --no-cov
python -m jarvisx.inward_loop_certified --particles 8192 --frames 600 --seed 7 --json artifacts/inward-loop-certificate.json
python -m jarvisx.inward_loop_certified --verify artifacts/inward-loop-certificate.json
```

All metrics and hashes come from numerical evaluation. The code does not draw
images; a downstream dashboard can visualize these independently measured values.
