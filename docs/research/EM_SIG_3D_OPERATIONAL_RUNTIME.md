# EM-Sig 3D Operational Runtime

**Status:** Executable bounded research runtime  
**Module:** `python -m jarvisx.em_sig_3d_runtime`  
**Evidence class:** deterministic software simulation; not RF hardware telemetry

## Purpose

This runtime operationalises the repository's electromagnetic-signalling logic
as a complete materialized 3D software pipeline.

```text
bits
  -> QPSK
  -> spatial source program
  -> data-dependent 3D interference volume
  -> centered readout
  -> ternary voxel logic
  -> sparse sensor lattice
  -> six-bit spatial latent
  -> fixed-point refinement
  -> residual decoder
  -> recovered bits
  -> machine-readable verification receipt
```

It extends `jarvisx.em_sig_emulation` without changing the physical evidence
boundary. The field stage is a scalar interference proxy, not a Maxwell,
FDTD, FEM, antenna, circuit, or measured electromagnetic solution.

## 1. Source programming

Pairs of bits are Gray-style QPSK symbols

```text
00 -> (+1 + j)/sqrt(2)
01 -> (-1 + j)/sqrt(2)
11 -> (-1 - j)/sqrt(2)
10 -> (+1 - j)/sqrt(2)
```

Each of the configured source elements receives a 3D source instruction

```text
S_k = (x_k, y_k, z_k, A_k, phi_k)
```

where source position is deterministic and `A_k, phi_k` come from the QPSK
symbol. The materialized 3D field therefore depends on the encoded source bits.

## 2. 3D interference volume

At each voxel `r`, the software evaluates

```text
psi(r) = sum_k A_k exp(j (k |r-r_k| + phi_k)) / |r-r_k|
I(r)   = |psi(r)|^2
```

and normalizes the scalar intensity volume.

This is explicitly a reduced software proxy. It does not represent Poynting
flux, because no coupled `E` and `H` Maxwell fields are solved.

## 3. Volumetric logic decoding

The normalized intensity is centered into a software readout

```text
s(r) = 2 I_norm(r) - 1
```

and quantized with an explicit deadband

```text
B(r) = +1   if s(r) > +theta
        0   if |s(r)| <= theta
       -1   if s(r) < -theta
```

Thus every materialized voxel receives an explicit ternary logical state.

The runtime verifies

```text
N_negative + N_deadband + N_positive = extent^3
```

so the complete materialized volume is accounted for.

## 4. Sparse 3D sensing

A regular sparse sensor lattice samples

```text
(x, y, z, readout, logic)
```

at a configurable stride. This represents the software analogue of the EEIITL
observation boundary:

```text
field -> readout -> quantizer -> sensor state
```

without claiming a physical sensor implementation.

## 5. Spatial latent

The full ternary geometry plus sparse sensor coordinates are folded into a
deterministic six-bit spatial signature

```text
z_target in {0, ..., 63}
```

and the existing bounded fixed-point routine converges to that state.

The six-bit latent is deliberately lossy and is marked `invertible: false`.
It cannot reconstruct an arbitrary input bitstream by itself.

## 6. Residual decoder

Exact source recovery remains a latent-plus-residual problem.

The runtime uses the existing bounded residual-correction surrogate:

```text
r_i = b_i XOR b_hat_i
```

and corrects only the active error set until both

```text
residual_count == 0
reconstructed_(t+1) == reconstructed_t
```

hold.

A zero observed software BER is reported only for the finite materialized
bitstream and is not promoted into a physical BER claim.

## 7. Operational invariant

The executable runtime follows

```text
Encode
  -> Program Sources
  -> Materialize 3D Field
  -> Threshold
  -> Sense
  -> Compress
  -> Refine
  -> Decode Residual
  -> Verify
  -> Emit Receipt
```

The JSON receipt uses

```text
schema_version = jarvisx.em-sig-3d-runtime.v1
provenance     = simulated
```

and includes source-program state, field-volume metrics, voxel logic counts,
sensor observations, latent convergence, decoder convergence, and claim
boundaries.

## 8. Run

```bash
python -m pip install -e ".[test]"

pytest tests/test_em_sig_3d_runtime.py -q --no-cov

python -m jarvisx.em_sig_3d_runtime \
  --bit-count 65536 \
  --materialized-extent 12 \
  --field-sources 16 \
  --logic-deadband 0.20 \
  --sensor-stride 2 \
  --seed 42 \
  --output artifacts/em-sig-3d-runtime.json

python -m json.tool artifacts/em-sig-3d-runtime.json > /dev/null
```

## 9. Verification boundary

A passing runtime verifies only:

- deterministic bit and QPSK encoding;
- input-dependent 3D source programming;
- materialized scalar 3D interference arithmetic;
- complete ternary voxel-state accounting;
- deterministic sparse 3D sensing;
- six-bit spatial latent convergence;
- residual bitstream recovery under the configured software channel;
- stable machine-readable execution evidence.

It does **not** verify:

- physical RF transmission;
- Maxwell-field accuracy;
- antenna or photonic hardware;
- electromagnetic compatibility;
- a physical ternary logic substrate;
- measured electromagnetic sensing;
- hardware throughput, latency, energy, or BER.

Those remain separate circuit/device/field-solver and laboratory-validation
questions governed by the EEIITL research boundary.
