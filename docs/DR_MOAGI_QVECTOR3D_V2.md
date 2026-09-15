# Dr Moagi Q16.16x3 Field Engine v2

This document defines the strengthened **reference architecture** for the Dr Moagi
Q16.16 x Q16.16 x Q16.16 vector-field runtime.  The v2 target is "10/10" in the
sense of a coherent, deterministic, testable software reference machine.  It does
not claim unbuilt multi-host consensus, GPU/RTL acceleration, or physical
electromagnetic hardware.

## 1. State contract

Each spatial cell is a three-component signed fixed-point vector:

```text
Xi[i,j,k] = (Qx, Qy, Qz)
Qx,Qy,Qz in signed Q16.16
```

One logical cell therefore occupies exactly 96 raw numeric bits / 12 bytes.
Spatial location `(i,j,k)` and vector value `(Qx,Qy,Qz)` are separate concepts.

## 2. Numeric control register semantics

`qvector_v2.py` adds explicit architectural numeric policy:

- `QRoundMode.TRUNCATE`
- `QRoundMode.NEAREST_AWAY`
- `QRoundMode.NEAREST_EVEN`
- output saturation enable/disable
- accumulator saturation enable/disable

Sticky status is reported through:

```text
saturated
accumulator_saturated
inexact
divide_by_zero
```

The field coprocessor exposes numeric state with `QSETMODE`, `QCLRSTATUS`, and
`QSTATUS` rather than leaving rounding behavior implicit.

## 3. Checked wide accumulation

Convolution/MAC does not requantize every product.  Each Q16.16 product is Q32.32
and is accumulated in a checked signed-64 architectural accumulator:

```text
ACC64 += Q16.16 * Q16.16
```

Only after the complete kernel MAC is the accumulator requantized once:

```text
Qout = Pi_Q16.16( round(ACC64 / 2^16) )
```

Accumulator overflow either saturates and raises the sticky status flag or fails
closed when accumulator saturation is disabled.

## 4. Packed data plane

`PackedQVectorField3D` stores the field in one contiguous big-endian bytearray:

```text
qx[32] | qy[32] | qz[32]
```

for exactly 12 bytes per cell, avoiding one Python object per resident cell.
It preserves the v1 binary digest contract and supports bounded tile iteration:

```text
for tile in field.iter_tiles((tx,ty,tz)):
    execute(tile)
```

This separates a very large virtual field from its bounded resident working set.

## 5. Deterministic field calculus

`QVectorFieldOps3D` provides fixed-point operators with explicit boundary policy:

```text
VGRADX
VGRADY
VGRADZ
VDIV
VCURL
VLAPLACE
VCONV3D
```

Supported boundary modes are clamp, zero, and wrap.

Directional derivative:

```text
dXi/dx ~= (Xi[i+1]-Xi[i-1]) / (2*dx)
```

Divergence:

```text
div Xi = dX/dx + dY/dy + dZ/dz
```

The scalar divergence is replicated across the three-vector destination cell so
it remains representable by the QVector data type.

Curl:

```text
curl Xi = (
  dZ/dy - dY/dz,
  dX/dz - dZ/dx,
  dY/dx - dX/dy
)
```

Laplacian uses the deterministic six-neighbor 3D stencil.

## 6. 128-bit field coprocessor ISA

The field coprocessor intentionally retains the existing physical instruction
width:

```text
opcode[8] | flags[8] | x[16] | y[16] | z[16] | a[16] | b[16] | imm[32]
```

It shares vector-field handles and vector registers with the existing QVector VM.
The stable v1 opcode map is not changed; v2 field operators live in a separate
coprocessor opcode domain.

```text
QSETMODE     set architectural rounding mode
QCLRSTATUS   clear sticky numeric flags
QSTATUS      emit (sat, acc_sat, inexact) into Vd

VGRADX       field[a] -> field[b]
VGRADY       field[a] -> field[b]
VGRADZ       field[a] -> field[b]
VDIV         field[a] -> field[b]
VCURL        field[a] -> field[b]
VLAPLACE     field[a] -> field[b]
VCONV3D      field[a], kernel[imm] -> field[b]

VERIFY
SEAL
HALT
```

The field program itself is a 3D ROM with the same program-counter-to-coordinate
mapping used by the existing VM.

## 7. Bit-exact inward optimization

Candidate latent geometry selection no longer depends on floating-point ranking.
For source field `X`, candidate reconstruction `Xhat_m`, Q16.16 complexity weight
`wq`, source cell count `N`, and latent cell count `Nz_m`, define raw three-lane
squared error:

```text
SSE_m = sum[(Qx-Qxhat)^2 + (Qy-Qyhat)^2 + (Qz-Qzhat)^2]
```

The human-readable objective remains:

```text
J_m = component_mse_m + w * compression_ratio_m
```

but every candidate for the same source is ranked using the exact integer common-
denominator numerator:

```text
Jnum_m = SSE_m + 3 * 2^16 * wq * Nz_m
Jden   = 3 * N * (2^16)^2
```

Therefore:

```text
argmin J_m == argmin Jnum_m
```

using integers only.  Floating-point values are retained only as display metrics.
The result payload exposes both `objective` and exact `objective_q` numerator /
denominator fields.

## 8. Operational recurrence

The strengthened reference state is:

```text
Sigma = (
  3D ROM,
  PC,
  sparse QVector register lattice,
  packed/dense Q16.16x3 fields,
  field kernels,
  numeric policy/status,
  cloud scheduling state,
  Omega ledger,
  cryptographic execution trace
)
```

One bounded transition is:

```text
I_t = ROM[PC_xyz]
Sigma_(t+1) = Pi_Lambda[ Execute(I_t, Sigma_t) ]
PC_(t+1) = PC_t + 1 or a validated control transfer
```

The inward auto-encoding cycle is:

```text
X_Q3D
 -> candidate latent geometries Z_1:M
 -> deterministic decode Xhat_1:M
 -> exact raw SSE
 -> exact integer objective ranking
 -> select m*
 -> Pi_Lambda
 -> Omega commit
```

## 9. Electromagnetic bridge

The new differential operators are deliberately compatible with a future field
backend.  A complete electromagnetic state would require multiple staggered field
components and material/source state, for example:

```text
(E, H, epsilon, mu, sigma, J, rho)
```

The v2 software field ISA provides the deterministic gradient/divergence/curl and
Laplacian substrate needed before an FDTD/Yee backend can be implemented.  This
repository still does **not** claim a physical electromagnetic processor.

## 10. Validation

The dedicated Cloud OS workflow compiles and executes:

```text
tests/test_qvector3d.py
tests/test_qvector_v2.py
tests/test_qvector_field_bytecode.py
tests/test_qvector_optimizer_v2.py
examples/qvector3d_demo.py
examples/qvector_field_v2_demo.py
```

alongside the existing cloud/API/scalar bytecode tests and Docker build.

## 11. Remaining production backends

The software reference architecture is intentionally separated from these future
backends:

- native SIMD/GPU kernels;
- authenticated remote workers and durable queues;
- replicated/consensus-backed Omega state;
- untrusted-bytecode sandboxing;
- trainable quantized Conv3D/neural-operator weights;
- FDTD/Yee electromagnetic execution;
- FPGA/RTL or photonic/electromagnetic hardware mapping.

These are deployment/hardware implementations of the stable machine semantics,
not reasons to weaken the reference contract.

The engineering order remains:

```text
Working -> Robust -> Portable -> Elegant -> Advanced
```
