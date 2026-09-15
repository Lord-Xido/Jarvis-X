# Hierarchical 1000 GB 3D Engine

This module operationalizes the Moagi/Jarvis-X terabyte-scale geometry as a streaming software runtime rather than a dense 1000 GB allocation.

## Geometry

A single `1000^3 x 4` float32 volume is 16 GB. The logical 1000 GB dataset therefore corresponds to 62.5 such volumes, represented as 62,500 streamed `100^3 x 4` blocks of 16 MB each.

```text
1000 GB logical stream
  -> 62,500 x 16 MB blocks
  -> local 100^3 -> 10^3 -> 1^3 contraction
  -> local 64-D latent fixed-point core
  -> residual-preserving decode
  -> threshold + Top-K correction
  -> block latent aggregation
  -> global 64-D latent fixed-point core
```

Each `1000^3` volume contains a `10 x 10 x 10` lattice of `100^3` blocks. The GUI renders that lattice and draws processed blocks inward to the global latent core.

## Fixed-point core

The local and global latent cores use

```text
z[k+1] = (1-rho) z[k] + rho tanh(Wz z[k] + Wo Omega + b)
Omega[t+1] = beta Omega[t] + (1-beta) z*
```

with `rho=0.30`, `||Wz||_2=0.82`, and `beta=0.88`. The conservative relaxed contraction bound is

```text
q <= 1-rho + rho*0.82 = 0.946
q^64 ~= 2.864e-2
```

A measured final fold displacement is reported separately; it is not treated as the theoretical bound.

## Residual codec

Within each streamed block:

```text
100^3 --mean 10x10x10 neighborhoods--> 10^3 --mean--> 1^3 root
```

The decoder reconstructs geometrically as

```text
root_model + root_residual
  -> 10^3 + coarse_residual
  -> 100^3 + fine_residual
```

With residual quantization disabled, the tests verify an exact floating-point round trip. With quantization enabled, spatial residuals are int8-coded at the configured quantum (`0.02` by default). The 64-D latent is a summary; reconstruction information remains in the residual stream.

## Active correction

For reconstruction error `delta = S - S_hat`, the scheduler evaluates

```text
A[t+1] = TopK_0.12({p : ||delta[p]|| > tau[t]})
tau[t] = tau0 * 0.6^t
S_hat[p] <- S_hat[p] + 0.35 * delta[p], p in A[t+1]
```

`numpy.argpartition` is used rather than a full sort.

## Run

From repository root:

```bash
pip install -r apps/moagi-unified-3d/requirements.txt
python apps/moagi-unified-3d/terabyte_3d_engine.py --blocks 4
```

Interactive 3D lattice:

```bash
python apps/moagi-unified-3d/terabyte_3d_engine.py --blocks 16 --gui
```

A real raw float32 stream can be memory-mapped:

```bash
python apps/moagi-unified-3d/terabyte_3d_engine.py \
  --input /path/to/blocks.f32 \
  --blocks 100
```

The raw file layout is contiguous `[block][x][y][z][channel]`; file size must be divisible by 16 MB for the default geometry.

## Verification

```bash
python apps/moagi-unified-3d/test_terabyte_3d_engine.py
```

Tests cover 3D addressing, exact residual round-trip, bounded lossy reconstruction, the relaxed contraction bound, and global latent reduction.

## Throughput semantics

The runtime prints measured GB/s for the blocks actually executed and an extrapolated time for the logical 1000 GB dataset. That projection is explicitly host-dependent. The implementation makes no claim that C++, GPU, optical, or EM hardware automatically reaches a specific throughput without measurement.
