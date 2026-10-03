# Dr Moagi Exynos-Inspired 3D Bit Fabric

**Status:** integration candidate  
**Scope:** exact bit/data mapping for the canonical 1000 x 1000 x 1000 virtual lattice  
**Runtime:** `src/jarvisx/exynos3d_bit_fabric.py`

## Purpose

This layer permeates the current Dr Moagi bit-data model into one explicit,
testable mapping contract for a heterogeneous virtual SoC.

It is **Exynos-inspired only**. It is not Samsung firmware, does not contain
Samsung proprietary code, does not model undocumented Exynos microarchitecture,
and makes no physical Exynos performance claim.

The virtual processing domains are generic heterogeneous-compute roles:

```text
CPU | GPU | NPU | ISP | CODEC | DSP | MODEM | DISPLAY
```

The layer adds exact spatial addressing, deterministic domain routing,
procedural word materialization, reversible structural bit transport,
representation-residual telemetry, and candidate-first branchless commit.

## Logical geometry

The canonical coordinate domain is

```text
(x, y, z) in {0, ..., 999}^3
```

with exactly

```text
1000^3 = 1,000,000,000
```

logical cells.

Linear addressing is

```text
a(x,y,z) = x + 1000 * (y + 1000 * z)
```

and the runtime proves the inverse mapping for the complete legal address
interval.

The fabric is also decomposed into 10 x 10 x 10 logical working tiles:

```text
100 tiles/axis
100^3 = 1,000,000 virtual tiles
1000 cells/tile
```

For each coordinate the adapter derives:

```text
coordinate
 -> linear address
 -> tile coordinate
 -> tile id
 -> local coordinate
 -> local id
```

without allocating the billion-cell lattice.

## Heterogeneous routing

Each processing role has a public geometric anchor. A coordinate routes to the
nearest anchor using deterministic squared Euclidean distance with a stable
domain-id tie break.

The routing result is a **logical placement policy**, not a statement that a
real Exynos chip is physically laid out this way.

```text
R(x,y,z)
 -> CPU | GPU | NPU | ISP | CODEC | DSP | MODEM | DISPLAY
```

The role mapping can later be lowered through an explicit hardware/backend
adapter while preserving this reference contract.

## Procedural bit materialization

The full logical state is never resident.

A 32-bit word can be materialized on demand as

```text
B(x,y,z) = Mix32(
    address(x,y,z)
    XOR seed
    XOR domain_tag
)
```

so identical coordinates and seeds reproduce identical words without storing
one billion words.

This is algorithmic/procedural state, not evidence of 4 GB of resident memory.

## Data representation

The reference word width is 32 bits.

A word may represent data, a typed numeric payload, a state digest fragment, or
another explicitly declared machine-level value. This adapter does not erase
the type distinction between those meanings.

The system therefore treats these as different operations:

```text
serialization != compression
quantization  != hashing
encoding      != encryption
bit equality  != numeric equality
```

## Representation residual versus numeric residual

For baseline word B and candidate C:

```text
E_bit = B XOR C
H     = popcount(E_bit)
E_num = int(C) - int(B)
```

The Hamming distance is a representation-level metric. Numeric delta is a
value-level metric. They are deliberately reported separately.

Example:

```text
0xB6 XOR 0xA6 = 0x10
Hamming distance = 1
numeric delta    = -16
```

One changed bit is not equivalent to one unit of numeric error.

## Structural transport codec

The adapter includes a reversible 32-bit transport transform:

```text
encoded = ROTL32(word XOR spatial_key, domain_shift)
decoded = ROTR32(encoded, domain_shift) XOR spatial_key
```

with the invariant

```text
decode_transport_word(p, encode_transport_word(p, w)) == w
```

for every legal 32-bit word and coordinate.

This transform is intentionally classified as a **structural bit codec**. It is
not compression and is not a learned autoencoder.

## Relationship to the 3D autoencoder

The repository's numerical 1000^3 autoencoding semantics remain owned by
`SparseBillionField`.

The intended composition is:

```text
external bits
 -> typed unpack
 -> coordinate/address mapping
 -> SparseBillionField observation
 -> Q3 encode
 -> bounded reason/couple/control
 -> decode
 -> numeric residual + Omega
 -> candidate
 -> typed pack
 -> XOR/Hamming residual
 -> Lambda decision
 -> branchless COMMIT / ROLLBACK
```

This keeps the numerical autoencoding error distinct from exact representation
error.

## Transaction law

A candidate bit-state does not become authoritative merely because it exists.

Given a Lambda/admission decision V:

```text
mask = 0xFFFFFFFF if V else 0x00000000

B_next =
    (B_candidate AND mask)
    OR
    (B_before AND NOT(mask))
```

Therefore:

```text
V = true  => B_next == B_candidate
V = false => B_next == B_before
```

Rejection preserves the exact prior 32-bit word.

The caller owns the policy producing V; the bit fabric only guarantees the
commit semantics.

## Inward-loop interpretation

The bit fabric can participate in the wider Jarvis-X inward loop as:

```text
B_t
 -> typed decode / numerical field
 -> E_theta
 -> Z_t
 -> D_theta
 -> B_candidate
 -> (numeric residual, XOR residual, Hamming residual)
 -> Omega
 -> Lambda
 -> COMMIT / ROLLBACK
 -> B_(t+1)
```

The exact binary closure metric remains

```text
popcount(F(B) XOR B)
```

when a subsystem declares bit-exact fixed-point verification.

## Capability boundary

Implemented by this integration candidate:

- exact 1000^3 linear coordinate mapping;
- exact 10^3 tile decomposition and inverse;
- one-million-tile logical geometry;
- deterministic heterogeneous-role routing;
- lazy deterministic 32-bit word materialization;
- reversible structural bit transport;
- XOR/Hamming and numeric-delta telemetry;
- branchless candidate/baseline selection;
- immutable transition receipts.

Not established by this layer:

- Samsung firmware compatibility;
- Exynos instruction compatibility;
- actual CPU/GPU/NPU/ISP/modem placement;
- physical cache, DVFS, voltage, power or thermal behaviour;
- dense billion-cell allocation;
- learned semantic compression;
- hardware throughput or silicon acceleration;
- cryptographic confidentiality.

Those require separate backend implementations and measured evidence.
