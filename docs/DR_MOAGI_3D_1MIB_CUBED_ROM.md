# Dr Moagi 3D 1 MiB × 1 MiB × 1 MiB Auto-Encoding ROM

## Status

Reference integration for the Jarvis-X recursive 3D runtime.

This module defines a **virtual** ROM with one mebibyte of byte-address coordinates on each spatial axis:

\[
2^{20}\times2^{20}\times2^{20}=2^{60}\ \text{bytes}=1\ \text{EiB}.
\]

The implementation does **not** materialize an exabyte. It keeps the logical address space sparse and deterministically synthesizes finite pages on demand.

## Address geometry

Coordinates are

\[
(x,y,z)\in\{0,\ldots,2^{20}-1\}^{3}.
\]

The three 20-bit coordinates are interleaved into one 60-bit Morton/Z-order address:

\[
A=\operatorname{Morton}_{3}(x,y,z).
\]

The mapping is bijective and is covered by regression tests at the origin, interior points, and the maximum coordinate.

## Inward contraction

The reference tile is

\[
10\times10\times10=1000
\]

byte cells. One geometric contraction stage therefore maps 1000 fine spatial sites to one coarse state plus residual information:

\[
B_q=D(c_q)+R_q,
\qquad
c_q=E_{\Theta}(B_q).
\]

The **1000×** number here is a spatial site-count ratio, not a measured wall-clock speedup. Real speedup depends on residual density, memory traffic, scheduling overhead, vectorization, and hardware.

## Runtime loop

The bootstrap bytecode sequence is:

```text
INGEST_3D
LOAD_TILE
ENCODE_3D
FOLD_IN_10
LATENT_STEP
FIXPOINT_CHK
DECODE_3D
RESIDUAL_ADD
CONTRAST
OMEGA_UPDATE
THETA_UPDATE
RETILE
EMIT_BYTE
HALT
```

The state evolution is

\[
V_t\rightarrow E_{\Theta_t}\rightarrow Z_t
\rightarrow F^r(Z_t)\rightarrow Z_t^*
\rightarrow D_{\Theta_t}\rightarrow \hat V_t,
\]

with reconstruction error

\[
e_t=V_t-\hat V_t,
\]

memory

\[
\Omega_{t+1}=\rho\Omega_t+(1-\rho)e_t,
\]

and adaptive parameters represented by \(\Theta_t\).

The fixed-point refinement terminates when

\[
\frac{\|Z^{(r+1)}-Z^{(r)}\|_2}{\|Z^{(r)}\|_2+\epsilon}<\tau_Z.
\]

## Sparse ROM semantics

Unmaterialized pages are generated deterministically from

```text
seed || page_index
```

using BLAKE2b. This gives reproducible random-access content without allocating the logical 1 EiB space.

An explicit page map can override generated pages. Page zero is used for the bootstrap program when `install_boot_program()` is called.

## 64-bit instruction format

Each instruction occupies exactly eight bytes:

```text
[ opcode:8 | a:8 | b:16 | c:32 ]
```

This keeps the ROM format compact and auditable while leaving room for later mapping onto the repository's broader 3D swarm ISA.

## Compact descriptor

`descriptor_bytes()` emits a small binary descriptor containing:

- magic and format version;
- axis length;
- total logical byte capacity;
- address width;
- sparse page size;
- contraction edge;
- seed digest;
- bootstrap bytecode.

The descriptor is metadata for the virtual ROM. It is not an exabyte disk image.

## Usage

After installation:

```bash
python -m jarvisx.dr_moagi_rom3d --self-test
python -m jarvisx.dr_moagi_rom3d --write-rom-descriptor moagi_3d.rom
```

The package-level CLI entry point is also exposed as:

```bash
jarvisx-dr-moagi-rom3d --self-test
```

## Verification boundary

The reference implementation verifies:

1. reversible 60-bit 3D addressing;
2. deterministic sparse-page reads;
3. fixed-width instruction packing;
4. 1000-byte volumetric tile reads;
5. bounded latent encode/refine/decode behavior;
6. compact descriptor integrity;
7. bootstrap execution through `HALT`.

It does not claim that a Python reference implementation physically executes 1 EiB of data, nor that the 1000:1 geometric contraction guarantees 1000× wall-clock acceleration.
