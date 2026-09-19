# Dr Moagi Bitwise 3D VME

**Status:** Reference laboratory  
**Parent integration:** PR #288  
**Runtime:** `src/jarvisx/dr_moagi_bitwise_vme.py`

This layer lowers the 3D geometric ANN/VME into packed integer state.

## Logical domain

The canonical logical extent is:

```text
4000 x 4000 x 4000 = 64,000,000,000 logical voxels
```

The runtime does not allocate that volume densely. Active cells are stored sparsely by 36-bit Morton/Z-order address.

## 64-bit voxel/control word

```text
[x:12 | y:12 | z:12 | value:4 | modality:2 | residual:8 | flags:14]
```

Twelve coordinate bits are sufficient for each axis because `4000 < 4096 = 2^12`.

Modalities use two bits:

```text
00 mesh
01 audio
10 video
11 text
```

Topology selectors likewise use two bits:

```text
00 torus
01 Klein
10 sphere
11 reserved
```

## 64-bit instruction word

```text
[opcode:8 | dst:8 | srcA:8 | srcB:8 | immediate:32]
```

The reference ISA includes voxel load/store, signed axis shifts, six-neighbour diffusion,
Laplacian, INT8 encoding, gating/recurrent operations, topology mapping, sonification,
feedback, Omega update, recurrence, and halt.

## Fixed-point arithmetic

Voxel values use unsigned 4-bit saturation. Latent state uses signed INT8/Q7-like arithmetic.
The six-neighbour update uses an integer approximation of alpha ~= 0.18:

```text
alpha_q8 = 46
keep_q8  = 210

v_next = sat_u4((210*v + 46*neighbor_mean) >> 8)
```

The recurrent core is physically bounded by `max_refine_steps` and terminates early when
the integer L1 residual is within tolerance.

## 1 GiB distinction

A 1 GiB store contains `2^33` bits. Dense capacity is therefore:

```text
1 bit/voxel -> 2^33 voxels
4 bit/voxel -> 2^31 voxels
8 bit/voxel -> 2^30 voxels
```

A dense 4000^3 field therefore does not fit in 1 GiB even at one bit per voxel.
The runtime uses sparse active cells and keeps logical extent separate from physical residency.

## Claim boundary

This implementation demonstrates bit packing, sparse addressing, integer diffusion,
bounded latent recurrence, and explicit memory accounting. It does not establish hardware
throughput, dense 64-billion-voxel residency, global convergence, or trained model quality.
