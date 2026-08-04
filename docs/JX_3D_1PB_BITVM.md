# JX-3D-1PB-BitVM

## Status

**Reference implementation / integration candidate.** This subsystem exposes a canonical
`1000 × 1000 × 1000` lattice of decimal 1 MB bricks while materializing only non-zero active
bricks. It is a bounded correctness reference, not a one-petabyte resident allocation or a
production security sandbox.

## Geometry

The default virtual extent is:

```text
1,000,000,000 bricks × 1,000,000 bytes
= 1,000,000,000,000,000 bytes
= 1 PB decimal
= 8,000,000,000,000,000 addressable bits
```

A brick coordinate is `(x, y, z)` with each component in `[0, 999]`. The linear brick index is:

```text
a = x + 1000 * (y + 1000 * z)
```

The canonical 64-bit address is:

```text
ASID[8] | CLASS[3] | Z[10] | Y[10] | X[10] | BYTE[20] | BIT[3]
```

The 20-bit byte field carries decimal offsets `0..999,999`; unused encodings are rejected.

## Sparse storage contract

- Missing bricks read as immutable zero.
- Reading does not allocate.
- The first non-zero write copy-on-write materializes one brick.
- A brick that returns to all-zero state is pruned by default.
- `max_resident_bricks` bounds committed physical payload.
- Virtual extent and physical residency are reported separately.

For `k` active bricks, payload residency is approximately:

```text
resident_payload = k × brick_bytes
```

rather than one petabyte.

## Transaction cycle

Every instruction follows:

```text
validate envelope
→ validate address range
→ capability check
→ copy-on-write stage
→ execute bounded bit operation
→ enforce accessed/resident brick budgets
→ calculate deterministic state digest
→ commit or rollback
→ append hash-chained journal receipt
```

Rejected instructions leave authoritative brick state unchanged but remain visible in the audit
journal.

## Reference ISA

| Opcode | Function |
|---|---|
| `BSET` | set a destination bit range |
| `BCLR` | clear a destination bit range |
| `BCOPY` | copy one bit range to another |
| `BNOT` | invert a source range into a destination |
| `BAND` | bitwise conjunction |
| `BOR` | bitwise disjunction |
| `BXOR` | bitwise exclusive-or |
| `BPOPCNT` | count set bits without mutation |
| `BHASH` | SHA-256 a canonically packed bit range |

The Python reference executes bit-by-bit and enforces `max_instruction_bits`. Optimized SIMD, GPU,
and distributed kernels may replace the physical execution strategy only if they preserve the same
addressing, state digest, transaction, and journal semantics.

## Example

```python
from jarvisx.bitvm_3d_1pb import (
    AddressClass,
    BitAddress,
    BitInstruction,
    BitOpcode,
    Sparse3DBitVM,
)

vm = Sparse3DBitVM()
destination = BitAddress(
    asid=0,
    access_class=int(AddressClass.WRITE),
    x=10,
    y=20,
    z=30,
    byte_offset=0,
    bit_offset=0,
)

receipt = vm.execute(
    BitInstruction(BitOpcode.BSET, destination=destination, length_bits=16)
)
assert receipt.committed
assert vm.resident_brick_count == 1
```

## Determinism and checkpointing

The implementation provides:

- canonical address packing and unpacking;
- deterministic state hashing over sorted sparse bricks;
- a deterministic hash-chained instruction journal without wall-clock fields;
- JSON-serializable checkpoints containing sparse payloads and journal receipts;
- checkpoint chain, state-digest, payload-length, and resident-budget verification.

## Explicit boundaries

This implementation does **not** establish:

- one petabyte of resident physical memory;
- fast full-volume scans;
- secure execution of hostile bytecode;
- distributed consensus or high availability;
- external API rollback;
- GPU or SIMD performance;
- intelligence, consciousness, or unrestricted self-modification.

The next promotion step is a named benchmark corpus comparing memory, range-operation latency,
serialization, rollback, and recovery against straightforward dense/sparse baselines.
