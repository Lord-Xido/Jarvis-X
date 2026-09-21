# Dr Moagi 1024³-Bit Exact 1 MiB ROM Profile

**Status:** bounded reference profile  
**Scope:** deterministic 3D bit-field traversal + recursive AE/AD control  
**Authority boundary:** the canonical Jarvis-X VM and its normal validation/commit rules remain authoritative.

## Geometry

The logical field is

\[
\mathcal B =
\{0,\ldots,1023\}^3,
\qquad
|\mathcal B|
=
1024^3
=
1{,}073{,}741{,}824\text{ bits}.
\]

A dense in-memory realization therefore requires

\[
\frac{1024^3}{8}
=
134{,}217{,}728\text{ bytes}
=
128\text{ MiB}.
\]

The ROM is deliberately smaller:

\[
\boxed{|R| = 1{,}048{,}576\text{ bytes} = 1\text{ MiB}}
\]

because it stores a compact executable traversal and control description, not
the entire logical bit field.

## ROM layout

The reference image is deterministic:

- total ROM size: **1 MiB**
- header: **4 KiB**
- instruction width: **64 bits**
- active program: **35 instructions**
- remaining instruction slots: deterministic `NOP` padding
- logical working field: **1024 × 1024 × 1024 bits**
- dense field storage if materialized: **128 MiB**

Instruction word:

```text
[63:56 opcode]
[55:52 dst]
[51:48 srcA]
[47:44 srcB]
[43:32 imm12]
[31:0  arg32]
```

## Iteration pipeline

One logical bit update is represented by

```text
LOAD_BIT
  -> NEIGHBOR6
  -> ENC_3D_BIT
  -> INWARD_FOLD
  -> OMEGA_ACCUM
  -> DEC_3D_BIT
  -> RESIDUAL
  -> VERIFY
  -> CORRECT
  -> STORE_BIT
```

Three bounded `JLT` loops traverse the x, y and z axes. The program therefore
describes all

\[
1024\times1024\times1024
\]

logical coordinates without unrolling one billion updates into ROM.

## Recursive mathematical form

For logical bit field \(B_t\), encoded state \(Z_t\), recurrent memory
\(\Omega_t\), and reconstruction \(\hat B_t\):

\[
Z_t = E_\theta(B_t),
\]

\[
Z'_t = \Phi_{\lambda}(Z_t),
\]

\[
\Omega_{t+1}
=
\rho\Omega_t + (1-\rho)Z'_t,
\]

\[
\hat B_t
=
D_\phi(Z'_t,\Omega_{t+1}),
\]

\[
R_t = B_t \oplus \hat B_t,
\]

\[
B_{t+1}
=
C_\alpha(\hat B_t,R_t).
\]

The ROM encodes the control skeleton for this loop. It does **not** claim that
the profile instructions alone contain trained ANN parameters.

Nominal fixed-point controls encoded in the reference program are:

- inward contraction: \(\lambda\approx0.75\)
- recurrent memory: \(\rho\approx0.85\)
- residual correction: \(\alpha\approx0.25\)

## Canonical relationship

This profile is an adapter beneath the Jarvis-X authority boundary. It does
not redefine the repository's canonical bytecode ISA.

The execution hierarchy is:

```text
Jarvis-X authority / policy / transaction layer
        |
        +-- 1024³-bit ROM profile
                |
                +-- compact 64-bit profile instructions
                +-- virtual 3D bit coordinates
                +-- inward AE/AD recurrence
                +-- verify/correct candidate state
```

Any candidate state remains provisional until the enclosing Jarvis-X runtime
accepts it through the normal verification and commit path.

## Reproducible build

After installing the package:

```bash
jarvisx-dm3d-bit-rom build moagi_3d_1024cube_1MiB.rom
jarvisx-dm3d-bit-rom inspect moagi_3d_1024cube_1MiB.rom
```

or directly:

```bash
python -m jarvisx.dm3d_bit_rom build moagi_3d_1024cube_1MiB.rom
```

The reference image must satisfy:

```text
size    = 1,048,576 bytes
sha256  = bb61432c3c952742f2c7fb81b47e1590b87e451efcf85fb87d896e39237f14e0
```

The test suite independently checks exact size, logical geometry, dense-state
size, deterministic SHA-256, code CRC, bounded loop structure, and corrupt-ROM
rejection.

## Operational boundary

This is custom profile bytecode, not ARM64 or x86 native machine code. Native
execution requires an interpreter, JIT, or lowering adapter that implements
the declared profile operations and then submits candidate state through the
canonical Jarvis-X authority/verification path.
