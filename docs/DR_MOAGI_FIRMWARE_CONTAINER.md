# Dr Moagi DMLAMBDA 1 GiB Firmware Container

`DMLAMBDA-1G` is the executable container boundary for the bounded Dr Moagi system. It turns the earlier conceptual byte map into an exact **1 GiB logical image** while preserving sparse residency and verified state transitions.

## Exact memory map

| Region | Start | End (inclusive) | Capacity |
|---|---:|---:|---:|
| Genesis / manifest | `0x00000000` | `0x000FFFFF` | 1 MiB |
| QSOL sparse state | `0x00100000` | `0x17FFFFFF` | 383 MiB |
| Sparse SPD metric field | `0x18000000` | `0x2BFFFFFF` | 320 MiB |
| Immutable RISC-V kernel | `0x2C000000` | `0x37FFFFFF` | 192 MiB |
| Trace / audit reserve | `0x38000000` | `0x3FFFFFFF` | 128 MiB |

The exclusive end is `0x40000000`, exactly `1,073,741,824` bytes. `FirmwareBuilder` creates the file with `truncate()` and writes only used section payloads, so supporting filesystems represent the unused reservations as sparse holes rather than physically writing a gigabyte of zeros.

## Trust and boot chain

```text
external Ed25519 public trust anchor
  -> verify 512-byte DMLAMBDA header
  -> SHA-256 verify canonical manifest
  -> Ed25519 verify manifest signature
  -> verify stored section SHA-256 values
  -> AES-256-GCM authenticate/decrypt QSOL + metric sections
  -> verify plaintext SHA-256 values
  -> DMOS2 exact sparse-state decode
  -> DMMET1 sparse SPD metric decode
  -> validate ELF64 RISC-V e_machine = 243 and executable PT_LOAD
  -> verify SHA3-256 trace anchor
  -> boot DrMoagiOSKernel
  -> wrap SelfOptimizing3DSystem
  -> wrap SelfEvolving3DArchitecture
  -> execute bounded autonomic cycles
```

A signed image requires the public key to be supplied externally at verify/boot time. The image does not treat an embedded public key as its own root of trust.

## Section semantics

### Genesis

The first 512 bytes are the DMLAMBDA header. The canonical JSON manifest begins at offset 512 and contains the exact region map, per-section used byte counts, hashes, codecs, encryption metadata, signer fingerprint, kernel metadata, and capability boundaries.

### QSOL sparse state

QSOL stores the existing exact `DMOS2` Morton-delta/float64 state packet. The logical lattice may be much larger than its resident state; only active coordinates are encoded.

### Metric field

`DMMET1` stores sparse symmetric 3x3 metric tensors as six float32 components:

```text
(gxx, gyy, gzz, gxy, gxz, gyz)
```

Every stored tensor must satisfy Sylvester positive-definiteness checks. Coordinates not explicitly present are interpreted by the runtime as the Euclidean identity metric where needed. The reference dynamic step is **SPD-preserving metric relaxation**, not a claim of Ricci-flow integration.

### Kernel

The default image contains a minimal valid ELF64 little-endian RISC-V executable monitor stub with `EM_RISCV = 243`, one executable `PT_LOAD` segment, and entry `0x80000000`. It is deliberately a reference monitor, not board-specific startup firmware. A real board supervisor can be injected with `--supervisor` and must pass the same ELF validation.

The executable region is immutable under the runtime. Self-optimization changes bounded state/model/configuration/architecture parameters, not executable instructions.

### Trace

The genesis trace anchor is SHA3-256 over the initial QSOL, metric, and kernel content digests. A boot session advances the trace head with canonical run events. A hash chain is tamper-evident only when the expected head is anchored outside an attacker-controlled image; the manifest states that boundary explicitly.

## Cryptography

- Manifest authentication: Ed25519.
- Section confidentiality/integrity: AES-256-GCM.
- Per-section key derivation: HKDF-SHA256 with a random image salt and section-specific context.
- Section and manifest digests: SHA-256.
- Runtime trace chain: SHA3-256.

No key is derived from geometry, holonomy, state values, or other public deterministic data. Private signing and AES master keys must remain outside source control and firmware images.

## CLI

Generate external keys:

```bash
jarvisx-dr-moagi-firmware keygen /secure/dm-fw
```

Build an exact-size signed/encrypted demo image:

```bash
jarvisx-dr-moagi-firmware build-demo dr-moagi-1g.img \
  --side 64 \
  --signing-private-key /secure/dm-fw.ed25519.private \
  --encryption-key /secure/dm-fw.aes256
```

Verify it:

```bash
jarvisx-dr-moagi-firmware verify dr-moagi-1g.img \
  --public-key /secure/dm-fw.ed25519.public \
  --encryption-key /secure/dm-fw.aes256 \
  --pretty
```

Verified boot and run:

```bash
jarvisx-dr-moagi-firmware run dr-moagi-1g.img \
  --cycles 8 \
  --public-key /secure/dm-fw.ed25519.public \
  --encryption-key /secure/dm-fw.aes256 \
  --pretty
```

Serve the firmware control plane:

```bash
jarvisx-dr-moagi-firmware serve dr-moagi-1g.img \
  --public-key /secure/dm-fw.ed25519.public \
  --encryption-key /secure/dm-fw.aes256 \
  --host 0.0.0.0 --port 10002
```

Service routes:

```text
GET  /healthz
GET  /v1/firmware/status
GET  /v1/firmware/manifest
POST /v1/firmware/verify
POST /v1/firmware/boot
POST /v1/firmware/run
```

## Capability boundary

Implemented:

- exact 1 GiB logical image layout;
- sparse physical file allocation where the host filesystem supports it;
- exact DMOS2 state persistence;
- sparse SPD metric storage and bounded relaxation;
- valid reference RV64 ELF payload and ELF validation;
- Ed25519 authenticated manifest;
- AES-256-GCM encrypted state/metric sections;
- verified boot into the existing state/model/configuration/architecture loops;
- hash-chain runtime trace;
- CLI and FastAPI control planes;
- fail-closed verification before execution.

Not claimed:

- a board-specific bare-metal bootloader;
- native execution of RISC-V instructions by a generic GPU;
- an integrated Ricci-flow numerical solver;
- arbitrary lossless compression of dense `1000^3` tensor data into the reserved regions;
- self-modifying executable code;
- secret-key derivation from public geometry;
- external SOTA superiority without matched benchmarks.

The system invariant remains:

```text
PROVISIONAL != AUTHORITATIVE
```

A firmware image is trusted only after the external trust anchor, manifest, encrypted sections, exact sparse-state transport, metric invariants, kernel ELF contract, and trace anchor all verify.
