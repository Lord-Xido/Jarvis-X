# Dr Moagi 3D Mandelbulb ANN — Inward Operational Verification

## Executive result

The supplied C++20 engine was compiled and independently audited by another C++20 translation unit that includes the **actual original source**. This is a measured proof-of-concept, not an assertion of AGI, a trillion-parameter system, or production-grade multimedia synthesis.

**Passed:** 3D voxel indexing, bounded Mandelbulb escape-time features, finite-difference gradient checks of 269 ANN parameters, the original Quine, three encode/archive/decode/re-encode passes per media type, decoder replay, edge case shapes, independent replay of the original four-stage spatial recursion, and bounded convergence to its frozen-input fixed point.

**Security gap:** no cryptographic authenticity or integrity verification in the archive. A one-bit flip in the final residual byte was accepted and changed the reconstructed content. Rejecting truncation and malformed magic does not detect data tampering.

## The faithful end-to-end mathematical model

1. Source information X = (x_i)_{i=0}^{N-1}. The original engine handles text/binary bytes, P6 PPM RGB data, and PCM16 RIFF/WAVE wrappers. Its input byte features normalize x_i into [-1,1].
2. Voxelization: S = ceil(N^(1/3)), i = a + S b + S² c, with (a,b,c) in [0,S-1]³, and p_i = (2a/(S-1)-1, 2b/(S-1)-1, 2c/(S-1)-1) for S > 1.
3. Mandelbulb: q_{k+1}=B_8(q_k)+p_i, q_0=0, radius escape > 2 or maximum seven steps. The scalar φ_i in [-1,1] encodes capped escape time, **not signed distance**.
4. Encoder E: v_i=[norm(x_i),φ_i,p_ix,p_iy,p_iz,neighbor_mean_i] in R^6. Shared tanh neural layers produce z_i^(0) in R^4 using 269 trainable parameters total.
5. Spatial mapping M: with B_i=z_i^(0) frozen, N_6(i) the occupied six-neighbor set, and A the normalized neighbor-averaging operator,

   T(Z)_i = 0.68 Z_i + 0.32 tanh( B_i + 0.18 (A Z)_i + 0.08 φ_i ).

   This mapping acts **jointly across the 3D voxel lattice**. The archive stores the state after four applications, Z^(4)=T⁴(Z^(0)); the auditor extends to 24.
6. Decoder D: predicted byte p_i = Quantize(tanh(w_o·tanh(W_d [z_i,φ_i,p_ix,p_iy,p_iz,neighbor_mean_i]+b_d)+b_o)).
7. Restoration R: residual r_i = x_i XOR p_i, hence (p_i XOR r_i)=x_i. **This is an identity from stored side information, not neural losslessness.**
8. Re-encoding: X_{k+1}=R(D(M(E(X_k))),stored residual_k), so X_{k+1}=X_k exactly for correctly stored archives. Repeating this recursion three times verified stability of input and archive output under deterministic seeds.
9. Generative branch: perturb Z and apply D plus modality-specific procedural synthesis; its output is not constrained to equal source or converge.
10. Quine: original C++ source can reproduce its own source bytes (through an embedded template) without opening its original source file. This is self-replication, **not autonomous model self-improvement**.

## Mathematical convergence proof and its scope

For any two latent fields U,V, frozen base B and geometry φ, using |tanh(a)-tanh(b)| ≤ |a-b| and ||A||_∞ ≤ 1,

||T(U)-T(V)||∞ ≤ (0.68 + 0.32·0.18)||U-V||∞ = 0.7376 ||U-V||∞.

The map T is a strict contraction, and therefore has a unique fixed point for each fixed B and φ. This does not show that the **outer** map from input to a newly trained network is a contraction, nor that generation is fixed-point stable.

## Measured C++ results

### Spatial and neural validation

- 269/269 gradients checked against central differences. Maximum observed absolute discrepancy: 0.000004 (reported rounded).
- Independently implemented frozen-base 3D recurrence matches original four-step output at tested bit precision (maximum absolute discrepancy 0).
- 24th recurrence update size and fixed-point residual are recorded in `audit_outputs/recurrence.csv`.
- Two perturbed trajectories obeyed the 0.7376 infinity-norm bound at every tested step.

### Three complete archive loops per modality

| Kind | Bytes in input | Bytes in archive | Expansion | Decoder-only RMSE (byte intensity) | Exact reconstruction |
|---|---:|---:|---:|---:|---|
| text | 91 | 3138 | 34.48× | 27.9297 | yes (3/3) |
| image | 6925 | 153213 | 22.12× | 12.6534 | yes (3/3) |
| audio | 4844 | 106780 | 22.04× | 11.0380 | yes (3/3) |
| bytes | 729 | 17174 | 23.56× | 10.6018 | yes (3/3) |

All tests ran on local CPU with `g++ -O2 -std=c++20`; reported benchmark durations are tiny sample sizes and **not representative of throughput at industrial scale**.

### Generative re-encoding

The generated image was re-encoded and generated again three times with fixed seed/strength. Absolute pixel differences were:
- Iteration 1: mean absolute byte change 19.716100.
- Iteration 2: mean absolute byte change 17.252900.
- Iteration 3: mean absolute byte change 14.665900.

The nonzero changes demonstrate that the generative pathway is not a fixed point of its own output. The decreasing changes over three steps do **not** establish eventual convergence.

## Known implementation limitations

- The ANN is trained on original per-byte features **before** recurrence, not by backpropagation through the spatial recurrence or across modalities.
- The precise source byte is included in each encoder feature, and the 3D layout is an indexing scheme rather than semantic 3D object understanding.
- Exact lossless restoration depends on the stored prediction byte and XOR residual byte. The archive expands significantly; it is not a compressed codec.
- No authentication, digital signatures, or authenticated encryption. Add SHA-256 with a trusted signature/MAC, or an AEAD container if security matters; a bare unkeyed hash only catches accidental corruption.
- Only basic text, PPM and PCM16 WAV formats; no video mux/demux, MP3/AAC, JPEG/PNG or learned text/image/audio generator.
- Current CPU data layout is not out-of-core, parallel sparse, or distributed; 8 MiB payload cap.
- Full self-rewriting/self-training Quine functionality has **not** been demonstrated.

## Reproduce

```sh
g++ -O2 -std=c++20 -Wall -Wextra -pedantic dm_mandelquine.cpp -o dm_mandelquine
g++ -O2 -std=c++20 -Wall -Wextra -pedantic dm_inward_verifier.cpp -o dm_inward_verifier
# Supply the original source's demo_outputs folder alongside original source:
./dm_inward_verifier audit_outputs path/to/original/dm_mandelquine.cpp
./dm_mandelquine quine > replica.cpp
cmp dm_mandelquine.cpp replica.cpp
```

The standalone interactive HTML reads data directly embedded from the C++ `spatial_trace.csv` and `recurrence.csv`; its canvas is a projected 3D visualization. Drag to orbit, scroll to zoom, select latent dimensions, and advance the 24 iterations.
