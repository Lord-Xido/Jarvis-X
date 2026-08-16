# DM–vΩΞ⁺ 32-bit Swarm ISA Contract

## Word format

```text
bits 31:24  opcode
bits 23:20  Rd, or Ra for store
bits 19:16  Rs1, or Rv for store
bits 15:12  Rs2
bits 11:0   immediate/sub-op
```

For `HASH_ADDR`, `imm[11:8]` carries `Rs3`; `imm[7:0]` remains available as a sub-op field.

## Registers

| Register | Role | Interpretation |
|---|---|---|
| R0 | zero | hardwired 0 |
| R1-R3 | POS_X/Y/Z | Q16.16 spatial coordinates |
| R4-R6 | VEL_X/Y/Z | Q16.16 velocities |
| R7-R9 | FRC_X/Y/Z | Q16.16 forces/intermediates |
| R10 | VOX_ADDR | raw 32-bit resident-page handle |
| R11 | LATENT_VAL | Q16.16 latent value |
| R12 | ERR_GRAD | Q16.16 local gradient |
| R13 | SYS_MODE | raw mode (`0` encode, `1` decode) |
| R14 | SYNC_CTR / scalar | typed control/scalar; v1.1 uses Q16.16 `dt` |
| R15 | SCRATCH | Q16.16 scratch/voxel value |

The phrase "32-bit Q16.16 register file" refers to the arithmetic datapath. Address and control instructions interpret their designated registers as raw 32-bit typed values.

## Opcodes

| Hex | Mnemonic | Semantics |
|---|---|---|
| 0x0F | SYNC_SWARM | global barrier |
| 0x10 | VREAD3D | `Rd <- VRAM[Rs1]` |
| 0x11 | VWRITE3D | `VRAM[Ra] <- Rv` |
| 0x20 | Q16MUL | saturating `Rd <- (Rs1 * Rs2) >> 16` |
| 0x21 | Q16ADD | saturating addition |
| 0x22 | Q16SUB | saturating subtraction |
| 0x30 | EVAL_FGRAD | local residual-gradient force |
| 0x31 | EVAL_FLATENT | latent-core attraction |
| 0x32 | EVAL_FREPEL | deterministic collision-avoidance term |
| 0x40 | ENCODE_STEP | bounded local compression |
| 0x41 | DECODE_STEP | bounded local expansion |
| 0x50 | HASH_ADDR | position tuple to sparse page-table handle |
| 0xFF | HALT | terminate the agent epoch |

## Addressing invariant

A 32-bit `VOX_ADDR` is a **handle**, not the full 3D address. The sparse page table stores the collision-resolving tuple `(page_x, page_y, page_z)` and returns a resident-slot handle. This is required because the complete 3D virtual key is wider than 32 bits.

## Q16.16 invariants

- additions and subtractions saturate to signed 32-bit limits;
- multiplication uses a signed 64-bit intermediate followed by a 16-bit fractional shift and saturation;
- the fixed-point representation of `0.01` used by v1.1 is `655 / 65536 = 0.0099945068359375`;
- the 6400-coordinate axis range is representable in signed Q16.16.

## Reality boundary

The ISA models deterministic local swarm computation and sparse virtual VRAM. It does not, by itself, grant external side-effect authority or convert a browser/native research state into authoritative system state.
