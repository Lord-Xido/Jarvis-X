# Dr Moagi 3D Bytecode Engine

`jarvisx.bytecode3d` is the execution substrate beneath the Dr Moagi Cloud OS reference runtime. It converts the geometric auto-encoding/decoding model into a bounded, deterministic 128-bit virtual instruction machine.

It is a software virtual machine, not a claim of physical electromagnetic hardware. Its purpose is to make the operational mechanics explicit and executable.

## 1. Core execution law

The engine realizes one bounded state transition per instruction:

```text
Sigma(t+1) = Pi_Lambda[ Execute( Decode( Fetch(ROM, PC_xyz) ), Sigma(t) ) ]
```

where:

- `ROM` is a dense 3D bytecode lattice;
- `PC_xyz` is the 3D coordinate of the current instruction;
- `Sigma` is the sparse 3D voxel-register state plus mounted 3D field handles;
- `Pi_Lambda` is implemented by opcode validation, fixed widths, finite Q16.16 values, bounded shapes, valid handles, valid jumps, node capacity and cycle limits;
- each step extends an in-memory cryptographic trace chain;
- `SEAL` and run completion commit checkpoints into the Cloud OS hash-chain ledger.

The inward auto-encoding cycle is exposed directly by the field opcodes:

```text
FROUND: X -> Encode -> Z -> Decode -> X_hat -> MSE -> cloud commit
FAUTO : X -> candidate latent geometries -> score -> select -> decode -> cloud commit
```

## 2. 128-bit instruction word

Every instruction occupies exactly 16 bytes:

```text
127           120 119           112 111            96
+----------------+----------------+------------------+
|   opcode 8     |    flags 8     |       x 16       |
+----------------+----------------+------------------+
95              80 79             64 63             48
+------------------+----------------+----------------+
|       y 16       |      z 16      |      a 16      |
+------------------+----------------+----------------+
47              32 31                               0
+------------------+--------------------------------+
|       b 16       |            imm 32              |
+------------------+--------------------------------+
```

Field meanings:

| Field | Mechanism |
|---|---|
| `opcode` | selects the micro-operation |
| `flags[3:0]` | destination local register `R0..R15` |
| `x,y,z` | target voxel in the sparse virtual processor lattice |
| `a,b` | source register indices or field handles, opcode dependent |
| `imm` | Q16.16 immediate, packed 3D shape, packed neighbor offset or jump target |

The binary representation is big-endian and round-trippable through `Instruction128.to_bytes()` / `Instruction128.from_bytes()`.

## 3. Coupled 3D spaces

The VM has two different 3D coordinate systems.

### 3D ROM space

A program has shape `(Rx, Ry, Rz)` and exactly `Rx * Ry * Rz` instruction cells. The scalar program counter maps to ROM coordinates by:

```text
pc_x = pc mod Rx
pc_y = floor(pc / Rx) mod Ry
pc_z = floor(pc / (Rx * Ry))
```

Thus fetch is conceptually:

```text
I_t = ROM[pc_x, pc_y, pc_z]
```

### 3D processor/data space

Each instruction separately names a virtual processor coordinate `(x,y,z)`. A voxel materializes only when used and owns 16 signed 64-bit fixed-point registers.

```text
Voxel[x,y,z] = {R0, R1, ... R15}
```

This separation permits the instruction trajectory through ROM to differ from the geometric location being transformed.

## 4. Q16.16 arithmetic

Scalar arithmetic uses Q16.16 fixed point:

```text
Q(x) = round(x * 2^16)
```

Operations are deterministic integer transforms:

```text
QADD: Rd <- Ra + Rb
QSUB: Rd <- Ra - Rb
QMUL: Rd <- trunc((Ra * Rb) / 2^16)
QDIV: Rd <- trunc((Ra * 2^16) / Rb)
```

Register writeback saturates to signed 64-bit range. Immediate literals are bounded to signed 32-bit Q16.16.

## 5. 3D movement

`VMOVE` transfers one register value from a neighboring voxel into the current voxel. Its immediate contains three signed 10-bit offsets:

```text
source_xyz = target_xyz + (dx, dy, dz)
Rd(target_xyz) <- Ra(source_xyz)
```

Offsets are bounded to `-512..511` per axis and the resulting coordinate must stay inside the unsigned 16-bit voxel lattice.

This is the literal bytecode form of point-to-point movement in virtual 3D state space.

## 6. Field/autoencoder operations

Mounted `Field3D` objects live in a 16-bit handle table.

| Opcode | Operation |
|---|---|
| `FENCODE` | `field[a] -> encoder -> field[b]` using packed latent shape |
| `FDECODE` | `field[a] -> decoder -> field[b]` using packed output shape |
| `FERR` | MSE between `field[a]` and `field[b]` -> destination Q register |
| `FROUND` | scheduled Cloud OS round trip; reconstruction -> `b`, latent -> `b+1`, MSE -> destination register |
| `FAUTO` | scheduled Cloud OS inward optimization; reconstruction -> `b`, latent -> `b+1`, objective -> destination register |

Packed field shapes use 10 bits per dimension and therefore support dimensions `1..1023` in a single instruction.

`FROUND` and `FAUTO` do not bypass the cloud runtime. They invoke `DrMoagiCloudOS`, which performs capacity-aware node selection, deterministic execution, result hashing and `job.committed` journaling.

## 7. Control and verification opcodes

| Opcode | Mechanism |
|---|---|
| `VERIFY` | writes Q16.16 `1.0` if the Cloud OS ledger verifies, otherwise `0.0` |
| `SEAL` | journals program digest, ROM PC, state digest and trace digest |
| `JMP` | absolute bounded linear ROM jump |
| `JNZ` | jump when a local Q register is nonzero |
| `HALT` | ends execution and permits final commit |

The engine has no unbounded hidden background loop. `run(max_cycles=N)` fails explicitly once the cycle budget is exhausted.

## 8. Micro-execution pipeline

One bytecode cycle is:

```text
1. muFETCH
   instruction <- ROM[PC_xyz]

2. muDECODE
   opcode, flags, target_xyz, a, b, imm <- instruction[127:0]

3. LAMBDA-GUARD
   validate opcode / register / field handle / shape / jump / neighbor bounds

4. muEXECUTE
   arithmetic, geometric move, autoencode/decode, cloud job or branch

5. muWRITEBACK
   update voxel register and/or field handle

6. muVERIFY
   calculate deterministic post-state digest

7. OMEGA-TRACE
   trace_digest(t+1) = SHA256(trace_entry(t) || trace_digest(t))

8. muNEXT
   PC <- PC + 1 or bounded branch target

9. SEAL / HALT boundary
   commit checkpoint/final digest to Cloud OS hash-chain ledger
```

This gives the operational correspondence:

```text
virtual arithmetic = local register transformation
geometric movement = VMOVE across voxel coordinates
encoding           = FENCODE / FROUND / FAUTO
error               = FERR or cloud MSE
memory              = field/register state + Omega ledger
projection          = validation and fixed-width bounds
execution           = movement through 3D ROM and 3D processor state
```

## 9. Minimal auto-executing program

```python
from jarvisx.bytecode3d import (
    BytecodeProgram3D,
    DrMoagiBytecodeEngine3D,
    Instruction128,
    Opcode,
    q_from_float,
)

q = lambda value: q_from_float(value) & 0xFFFFFFFF

program = BytecodeProgram3D.from_instructions([
    Instruction128(Opcode.QLOAD, flags=0, x=4, y=4, z=4, imm=q(1.5)),
    Instruction128(Opcode.QLOAD, flags=1, x=4, y=4, z=4, imm=q(2.25)),
    Instruction128(Opcode.QMUL, flags=2, x=4, y=4, z=4, a=0, b=1),
    Instruction128(Opcode.VERIFY, flags=3, x=4, y=4, z=4),
    Instruction128(Opcode.SEAL, x=4, y=4, z=4),
    Instruction128(Opcode.HALT),
])

engine = DrMoagiBytecodeEngine3D()
engine.load(program)
report = engine.run()

assert engine.read_q(4, 4, 4, 2) == 3.375
assert engine.read_q(4, 4, 4, 3) == 1.0
assert report.halted
```

## 10. Cloud auto-encoding program

```python
from jarvisx.bytecode3d import BytecodeProgram3D, DrMoagiBytecodeEngine3D, Instruction128, Opcode, pack_shape
from jarvisx.cloud_os import Field3D

engine = DrMoagiBytecodeEngine3D(default_node_cells=4096)
engine.mount_field(10, Field3D.from_values([3.0] * 64, (4, 4, 4)))

program = BytecodeProgram3D.from_instructions([
    Instruction128(
        Opcode.FROUND,
        flags=0,
        x=1, y=1, z=1,
        a=10,
        b=20,
        imm=pack_shape((2, 2, 2)),
    ),
    Instruction128(Opcode.SEAL, x=1, y=1, z=1),
    Instruction128(Opcode.HALT),
])

engine.load(program)
report = engine.run()

# field[20] = reconstruction
# field[21] = latent state
# R0 at voxel (1,1,1) = reconstruction MSE in Q16.16
```

## 11. Operational boundary

The VM is deliberately bounded and auditable. It currently provides a deterministic software ISA and reference execution engine. It does not itself provide process isolation, kernel-mode execution, distributed consensus, arbitrary untrusted-code sandboxing, GPU bytecode execution, JIT compilation, or electromagnetic hardware realization.

Those are separate engineering layers that can be added while retaining the same instruction/state contracts.
