# DMKB-1 — Dr Moagi Kinetic Bytecode Binary Profile

## Status

DMKB-1 is a bounded binary lowering and transport profile for the QSOL 3D graphics codec. It is **not** a replacement for the canonical Jarvis-X authority VM and it is not a second top-level bytecode architecture.

Its role is:

```text
application / geometry / latent state
        -> typed graphics IR
        -> DMKB-1 instruction + data packets
        -> verified binary transport
        -> CPU / GPU / WebGPU / native backend adapter
```

The outer Jarvis-X rule remains candidate-first: decoding or executing a DMKB-1 packet does not make the resulting state authoritative. The enclosing runtime still applies its ordinary validation and commit/rollback gate.

## 1. Core objects

DMKB-1 defines four objects:

- `I32`: one validated 32-bit graphics operational instruction;
- `M_Q`: a self-contained quantized meshlet packet;
- `Z_Q`: a self-contained quantized latent-vector packet;
- `C`: a versioned integrity-protected DMKB container.

The lowering path is:

```text
X -> E(X) -> Z -> Q(Z) -> C -> verify -> decode/execute -> X_hat
```

## 2. Common container

Every persistent packet uses the following outer envelope:

```text
offset  bytes  field
0       4      magic = "DMKB"
4       1      version = 1
5       1      packet kind
6       2      flags, little-endian
8       4      payload length, little-endian
12      N      payload
12+N    32     SHA-256(header || payload)
```

Packet kinds are:

```text
1  Meshlet
2  LatentVector
3  InstructionStream
```

The decoder rejects:

- invalid magic;
- unsupported version;
- unknown packet kind;
- a declared payload above the implementation ceiling;
- length mismatch or trailing bytes;
- SHA-256 mismatch.

The digest establishes packet integrity. It is not an encryption mechanism and is not treated as an invertible latent representation.

## 3. Bit ordering

Variable-width fields use LSB-first packing within the byte stream. Sequential writes consume the least-significant available bits of the current byte before continuing into subsequent bytes.

The bit reader and writer are fail-closed:

```text
requested bits > remaining bits  -> reject
invalid bit width                -> reject
NaN / Infinity                   -> reject
invalid quantization range       -> reject
```

Generic integer fields support widths `1..32`. Float quantization uses `1..24` bits because `float32` has 24 bits of significand precision including the implicit leading bit.

## 4. Graphics instruction word

The 32-bit word is:

```text
bits       width  field
0..7       8      opcode
8..12      5      dst
13..17     5      src1
18..22     5      src2
23..25     3      flags
26..31     6      immediate
```

Therefore:

```text
8 + 5 + 5 + 5 + 3 + 6 = 32 bits
```

Registers must be `0..31`, flags `0..7`, and the immediate `0..63`. Invalid host values are rejected rather than silently masked.

The C# `CompactInstruction` object is a host representation. `Pack32()` is the authoritative 4-byte word representation.

Current opcode surface:

```text
0x00 NOP
0x01 LOAD_VEC3
0x02 LOAD_VEC4
0x03 STORE_VEC3
0x04 ADD_VEC3
0x05 MUL_SCALAR
0x06 ROT_AXIS
0x07 BIND_MESHLET
0x08 BIND_SHADER
0x09 DRAW_INDEXED
0x0A REFLECT_EVAL
```

An instruction stream stores a 32-bit word count followed by exactly that many little-endian `I32` words.

## 5. Quantization law

For a scalar `x` in declared range `[xmin,xmax]`, nearest quantization is:

```text
q = round((x-xmin)/(xmax-xmin) * (2^b-1))
```

and reconstruction is:

```text
x_hat = xmin + q/(2^b-1) * (xmax-xmin).
```

The quantization step is:

```text
Delta_b = (xmax-xmin)/(2^b-1)
```

and, under nearest rounding,

```text
|x-x_hat| <= Delta_b/2
```

apart from ordinary finite `float32` arithmetic tolerance.

For a constant range `xmin == xmax`, the canonical quantized code is zero and decoding returns the constant exactly. This avoids `0/0` normalization.

## 6. Meshlet packet

A meshlet payload contains:

```text
offset  bytes  field
0       1      quantization bits
1       1      canonical local-index bit width
2       2      reserved = 0
4       2      vertex count
6       2      index count
8       12     anchor xyz float32
20      12     delta minimum xyz float32
32      12     delta maximum xyz float32
44      4      exact bit-payload length
48      N      packed vertices followed by packed indices
```

The encoder derives bounds from the actual anchor-relative geometry:

```text
d_i = v_i - anchor
min_k = min_i d_i,k
max_k = max_i d_i,k
```

No out-of-band `[-10,+10]` window is required.

The local topology width is:

```text
b_i = max(1, ceil(log2(vertex_count))).
```

Every index must be less than `vertex_count`. The decoder verifies the canonical width, counts, exact bit length, zero padding bits and topology bounds.

For three independently quantized coordinates, the deterministic Euclidean error envelope is:

```text
E_vertex <= sqrt((Delta_x/2)^2 + (Delta_y/2)^2 + (Delta_z/2)^2).
```

## 7. Latent-vector packet

A latent payload contains:

```text
offset  bytes  field
0       1      quantization bits
1       1      reserved = 0
2       2      reserved = 0
4       4      vector length
8       4      minimum float32
12      4      maximum float32
16      4      exact bit-payload length
20      N      packed scalar codes
```

The current implementation permits `1..1,000,000` float elements and `1..24` quantization bits.

Rate-distortion telemetry should report at least:

```text
encoded bytes
raw float32 bytes
compression ratio
mean squared error
maximum absolute error
```

A compression ratio without distortion is not sufficient evidence for a lossy representation.

## 8. Verification invariants

The executable self-test verifies:

1. variable-width fields across byte boundaries;
2. exact `Pack32 -> Unpack32` equality for every instruction field;
3. exact instruction-stream round trip;
4. rejection of invalid register fields;
5. meshlet decode without out-of-band bounds;
6. exact meshlet topology preservation;
7. measured mesh vertex error within the deterministic quantization envelope;
8. SHA-256 corruption rejection;
9. truncation rejection;
10. latent max error within its deterministic quantization envelope;
11. exact constant-latent round trip;
12. rejection of invalid float quantization depths.

These checks run through the existing QSOL graphics codec `--self-test` path and therefore participate in its GitHub Actions build-and-verify workflow.

## 9. Authority boundary

DMKB-1 verifies binary structure and reconstruction semantics. It does not grant side-effect authority.

The system hierarchy remains:

```text
semantic state
 -> candidate graphics/latent state
 -> DMKB-1 lowering
 -> binary verification
 -> backend decode/execute
 -> measured result
 -> outer Pi_Lambda validation
 -> COMMIT or ROLLBACK
```

Accordingly:

```text
VALID DMKB PACKET != AUTHORITATIVE JARVIS-X STATE
```

## 10. Follow-on lowering targets

The profile is designed to support later adapters without redefining its semantic contracts:

- direct C# interpreter for the current graphics opcode family;
- WebGPU compute/raster lowering;
- Direct3D 12 or Vulkan command generation;
- meshlet GPU upload records;
- PBR material and texture packets;
- animation/motion packets;
- bounded shader identifiers and resource tables;
- cross-language Python/C++/C# golden vectors;
- rate-distortion selection across quantization candidates.

Any accelerator backend remains subordinate to the same packet validation, reconstruction bounds and outer transaction gate.
