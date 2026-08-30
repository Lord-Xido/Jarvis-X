# Dr Moagi PDF-Carried DM3D Bytecode Runtime

## Status

This layer turns the PDF experiment into a bounded repository runtime:

```text
PDF transport container
-> embedded manifest
-> embedded DM3D bytecode
-> SHA-256 verification
-> structural bytecode validation
-> bounded 3D VM
-> encode/refine/decode/re-encode
-> reconstruction + cycle error
-> measured physical-work telemetry
```

Opening the PDF never executes it. Execution occurs only when the explicit Jarvis-X runtime is invoked.

## Why bytecode instead of embedded Python

The earlier proof demonstrated that a PDF can carry source, be parsed, verified, extracted, and handed to CPython. That proves packaging but leaves arbitrary Python authority outside the document model.

DM3D narrows authority to a finite instruction set:

| Opcode | Operation |
| --- | --- |
| `ENCODE_3D` | Block-average 3D contraction |
| `REFINE_3D` | Bounded six-neighbour inward latent refinement |
| `DECODE_3D` | Deterministic nearest-cell 3D expansion |
| `REENCODE_3D` | Encode reconstructed output back to latent space |
| `ERROR_3D` | Reconstruction MSE |
| `CYCLE_ERROR` | Latent cycle MSE |
| `ERROR_MEMORY` | Persistent error-memory update |
| `CHECK_FIXED` | Fixed-point tolerance check |
| `HALT` | Final instruction |

The bytecode therefore describes computation without granting general-purpose source-code execution.

## Binary format

A DM3D program is:

```text
16-byte header
N x 16-byte fixed-width instructions
32-byte SHA-256 digest
```

Header fields:

```text
magic   = DM3DVM1\0
version = 1
count   = uint16 instruction count
reserved = 0
```

The parser rejects malformed sizes, unknown opcodes, misplaced `HALT`, digest failures, excessive instruction counts, and excessive refinement passes before execution.

## Canonical recurrence

The reference program is:

\[
X_t
\xrightarrow{E}
Z_t^{(0)}
\xrightarrow{\mathcal R^K}
Z_t^*
\xrightarrow{D}
\widehat X_t
\xrightarrow{E}
\widetilde Z_t.
\]

The six-neighbour inward refinement is

\[
Z_{ijk}^{(k+1)}
=0.88Z_{ijk}^{(k)}
+0.10\,\overline{Z}_{\mathcal N(i,j,k)}^{(k)}
+0.02\,\mu_k.
\]

Reconstruction and cycle errors are

\[
e_x=\operatorname{MSE}(X_t,\widehat X_t),
\qquad
e_z=\operatorname{MSE}(Z_t^*,\widetilde Z_t).
\]

Error memory is

\[
\Omega_{t+1}
=\rho\Omega_t+(1-\rho)(\lambda_xe_x+\lambda_ze_z).
\]

The canonical program uses

\[
\rho=0.9,
\qquad\lambda_x=0.7,
\qquad\lambda_z=0.3.
\]

## Kinetic accounting

DM3D reports measured physical work rather than claiming a symbolic source-line rate:

```text
instructions executed
physical steps charged
voxel reads
voxel writes
valid neighbour reads
latent refinement updates
wall-clock runtime
physical steps / second
```

For the `16^3 -> 8^3 -> 16^3` four-pass reference, the recurrent core performs exactly

\[
4\times8^3=2048
\]

latent updates and

\[
10752
\]

valid directed six-neighbour reads.

A compressed super-instruction may represent large logical work, but represented work is not reported as physical hardware throughput.

## Resource bounds

`ProgramLimits` gates:

- maximum instruction count;
- maximum active voxels;
- maximum refinement passes;
- maximum charged physical steps.

A program that exceeds any bound aborts instead of partially continuing.

## PDF package integrity

The PDF embeds exactly two required attachments:

```text
manifest.json
engine.dm3d
```

The manifest records the bytecode SHA-256, byte length, instruction count, engine format, and the policy:

```text
explicit-runtime-only
```

The runtime verifies all manifest claims before parsing the bytecode.

## CLI

Install PDF support:

```bash
pip install -e '.[pdf]'
```

Build a package:

```bash
jarvisx-dr-moagi-pdfvm build dr-moagi.pdf --pool 2 --passes 4
```

Verify without executing:

```bash
jarvisx-dr-moagi-pdfvm inspect dr-moagi.pdf
```

Explicitly execute:

```bash
jarvisx-dr-moagi-pdfvm run dr-moagi.pdf --size 16
```

## Claim boundary

This is a deterministic executable research runtime and document-packaging format. It does not make a PDF reader itself a Python runtime, does not silently execute on document open, and does not interpret astronomical logical iteration counts as measured hardware throughput.
