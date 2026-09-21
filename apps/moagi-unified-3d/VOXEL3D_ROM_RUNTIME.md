# VOXEL3D ROM Runtime — Jarvis X Permeation Layer

Status: reference specification

This document binds the structured `VOXEL3D` ROM image to the existing Jarvis X / Dr Moagi unified 3D runtime without claiming native CPU/GPU executability for bytes whose instruction semantics are not yet specified.

## 1. Binary identity

The observed image begins with the eight-byte magic:

```text
7F 56 4F 58 45 4C 33 44
\x7FVOXEL3D
```

The image also contains these explicit ASCII anchors:

```text
GLSL_SVO_ENGINE
SVO_COMPUTE_SHADER_KERNEL_VECTOR
RAYMARCHER_INIT
AUTO_EXEC_LOOP_0
ENGINE_STATE_RUN
VOXEL_COUNT_1MB3
STACK_TRACE_ZERO
VOXEL_ENGINE_EOF
```

These anchors are treated as section markers and capability declarations, not as proof that embedded bytes are directly executable machine code.

## 2. Canonical operational pipeline

```text
VOXEL3D ROM BYTE IMAGE
        |
        v
      Decode
        |
        v
    SPATIAL VM
        |
        v
 GLSL SVO ENGINE
        |
        v
SVO COMPUTE / ENCODER
        |
        v
  LATENT CORE Z_t
        |
        v
     DECODER
        |
        v
RECONSTRUCTION X_hat_t
        |
        v
RESIDUAL R_t = X_t - X_hat_t
        |
        v
 CTR VERIFY / CORRECT
        |
        v
 SCHEDULER / POLICY Pi_t
        |
        v
 ENGINE STATE RUN
        |
        v
  RAYMARCHER INIT
        |
        v
   FRAME OUTPUT
```

The recursive invariant is:

```text
Generate -> Contrast -> Reckon -> Verify -> Correct -> Recur
```

## 3. State model

The integration uses the unified state:

```text
S_t = [X_t, Z_t, Omega_t, X_hat_t, R_t, Pi_t]
```

with

```text
Z_t      = E_theta(X_t)
X_hat_t  = D_theta(Z_t)
R_t      = X_t - X_hat_t
Omega_t  = temporal / persistent memory
Pi_t     = runtime and scheduling policy
```

A reference transition is:

```text
S_(t+1) = M_(theta,Pi_t)(S_t, verify(R_t))
```

The ROM is therefore a compact machine description. Execution expands it into dynamic state; the ROM itself is not the expanded computation.

## 4. ABI policy

The reference parser MUST:

1. Validate the `\x7FVOXEL3D` magic before interpreting the image.
2. Preserve byte offsets exactly.
3. Treat integer endianness as little-endian only where explicitly configured.
4. Discover printable ASCII anchors without mutating the underlying byte stream.
5. Expose candidate fixed-width instruction records as raw words until opcode semantics are formally assigned.
6. Never silently reinterpret unknown opcodes as host CPU instructions.
7. Preserve the original image as the source of truth for reproducible decoding.

## 5. AUTO_EXEC_LOOP_0

The bytes following `AUTO_EXEC_LOOP_0` contain repeated fixed-width-looking records. The currently observed leading words include values such as:

```text
0x3A
0x4B
0x3E
0x3F
0x80
0x84
0xFD
```

These values are exposed by the reference runtime as candidate VM opcodes. Their semantics remain `UNSPECIFIED` unless a formal opcode table is supplied.

`0xFD` may look like a terminator in the sample, but this is deliberately not encoded as a fact in the runtime until the ABI defines it.

## 6. CTR verification boundary

The verification layer is explicit:

```text
R_t = R(X_t, X_hat_t, H_t, P_t, E_t, C_t)
```

Operationally:

```text
Generate -> Contrast -> Reckon -> Verify -> Correct
```

Only a verified correction is allowed to update persistent runtime state. This keeps model-space adaptation separate from byte-level parsing and prevents malformed ROM content from being treated as trusted executable behavior.

## 7. Runtime integration

`voxel3d_rom_runtime.py` provides:

- strict magic validation;
- binary/hex loading;
- ASCII anchor discovery;
- immutable section metadata;
- raw little-endian word iteration;
- a reference phase trace for the Jarvis X pipeline;
- JSON telemetry suitable for the existing 3D visualizer or a browser UI.

It does **not** claim to execute GLSL, raymarching kernels, or undocumented VM opcodes. Those become executable only when corresponding host adapters and an opcode ABI are supplied.

## 8. Permeation rule

The VOXEL3D layer is inserted as an input/runtime-description substrate beneath the existing unified 3D engine:

```text
ROM -> parser -> spatial VM adapter -> volumetric engine -> CTR -> scheduler -> render
                         ^                              |
                         |------------------------------|
                               recursive feedback
```

This preserves the repository's existing runtime while giving the binary image a concrete, inspectable, testable place in the architecture.
