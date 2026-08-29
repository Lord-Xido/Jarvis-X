# DM-vOmegaXi+ VMAD128 Kinetic World Engine

Status: software reference implementation

This layer extends the existing sparse `VirtualVolume3D` substrate with a typed 128-bit volumetric address descriptor and a separate 64-bit micro-op execution format. It does **not** replace or alter the existing `JX3DVM1` bytecode wire format.

## 1. VMAD128 layout

The canonical descriptor is:

| Bits | Field | Width |
|---|---|---:|
| 127:116 | region | 12 |
| 115:108 | modality | 8 |
| 107:99 | attributes | 9 |
| 98:66 | X | 33 |
| 65:33 | Y | 33 |
| 32:0 | Z | 33 |

Each coordinate lies in `[0, 2^33)`. The conceptual byte volume is therefore `2^99` addressable byte sites. The implementation remains sparse: untouched coordinates are not materialized as resident memory.

Periodic addressing uses modulo-`2^33` coordinate arithmetic. Sequential payload transfer advances Z first, then Y, then X.

## 2. Register banks

The reference machine exposes:

- 512 vector registers `V0..V511`, each 64 bytes (512 bits), for 32 KiB of vector-register state;
- 512 scalar registers `R0..R511`;
- 32 VMAD registers `A0..A31`.

A 1024-byte ingress quantum occupies exactly 16 vector registers.

## 3. 64-bit micro-op

Each micro-op is encoded as:

```text
63          56 55      47 46      38 37      29 28   24 23               0
+-------------+----------+----------+----------+-------+-------------------+
| opcode (8)  | dst (9)  | src0(9)  | src1(9)  | A(5)  | immediate (24)   |
+-------------+----------+----------+----------+-------+-------------------+
```

A 128-bit VMAD is never embedded inside the 64-bit instruction. `LOAD_VMAD` loads one descriptor from the program descriptor table into `A0..A31`.

Implemented reference micro-ops:

- `LOAD_VMAD`
- `TILE_IN_VEC`
- `STORE_VEC`
- `ENC_LAT_VOL`
- `FUSE_ATTN`
- `DEC_PIX_VOL`
- `CALC_DELTA`
- `PROPOSE_BIAS`
- `VALIDATE`
- `COMMIT_IF`
- `HALT`

Payload movement is explicitly bounded to 4096 bytes per micro-op. The canonical demo uses 1024-byte quanta.

## 4. Five-stage kinetic model

The logical stages are:

1. ingestion;
2. latent reduction;
3. cross-modal/fixed-point fusion;
4. reconstruction/store;
5. inward feedback and transactional adaptation.

The software reference accounts for one issued micro-op per logical issue cycle and reports an estimated five-stage pipeline-fill latency of `issued + 4` cycles. This is architectural accounting only; it is not a measured clock frequency or physical silicon latency.

## 5. Deterministic transforms

`ENC_LAT_VOL` reduces a bounded byte block into a 32-byte latent vector using deterministic grouped averaging plus the currently authoritative local bias state.

`FUSE_ATTN` computes a centered fixed-point dot product between two 64-byte vector registers and derives a bounded blend coefficient. This provides deterministic cross-stream fusion without claiming a photonic matrix implementation.

`DEC_PIX_VOL` expands the 32-byte latent state into a bounded output payload and writes it to both vector registers and the sparse VMAD-addressed volume.

`CALC_DELTA` computes source-minus-reconstruction deltas, stores a centered signed-byte representation, and records mean absolute byte error.

## 6. Inward transactional adaptation

The adaptation sequence is:

```text
CALC_DELTA -> PROPOSE_BIAS -> VALIDATE -> COMMIT_IF
```

`PROPOSE_BIAS` writes only a shadow candidate. `VALIDATE` places the gate result in a scalar register. `COMMIT_IF` is the only operation permitted to promote the candidate into authoritative bias state.

Therefore the invariant is:

```text
no authoritative adaptive state changes before validation
```

A rejected candidate restores the existing authoritative bias state unchanged.

## 7. Sparse world-state semantics

VMAD region/modality/attribute fields are typed metadata at this layer. Physical residency is still provided by `VirtualVolume3D`, whose pages are lazily materialized and spilled through the existing sparse backing store.

This implementation proves the software addressing and execution semantics; it does not claim that `2^99` bytes are physically resident.

## 8. Claim boundary

Implemented and testable:

- exact 128-bit VMAD packing/unpacking;
- 33-bit X/Y/Z domain;
- toroidal coordinate wrapping;
- 64-bit micro-op encoding/decoding;
- 512 vector + 512 scalar + 32 VMAD register architecture;
- bounded sparse ingress/output;
- deterministic reduction/fusion/reconstruction;
- candidate/validate/commit rollback discipline;
- five-stage logical pipeline telemetry.

Not established by this software reference:

- silicon-photonic routing latency;
- TSV/eSRAM realization;
- sub-nanosecond or sub-picosecond timing;
- physical energy efficiency;
- external SOTA performance;
- patentability.

Those remain separate hardware and benchmarking evidence gates.
