"""Run arithmetic and 3D auto-encoding through the 128-bit bytecode engine."""

from jarvisx.bytecode3d import (
    BytecodeProgram3D,
    DrMoagiBytecodeEngine3D,
    Instruction128,
    Opcode,
    pack_shape,
    q_from_float,
)
from jarvisx.cloud_os import Field3D


def q(value: float) -> int:
    return q_from_float(value) & 0xFFFFFFFF


def main() -> None:
    engine = DrMoagiBytecodeEngine3D(default_node_cells=4096)
    engine.mount_field(
        10,
        Field3D.from_values(
            [float((x + y + z) % 4) for z in range(4) for y in range(4) for x in range(4)],
            (4, 4, 4),
        ),
    )

    program = BytecodeProgram3D.from_instructions(
        [
            Instruction128(Opcode.QLOAD, flags=0, x=2, y=2, z=2, imm=q(1.5)),
            Instruction128(Opcode.QLOAD, flags=1, x=2, y=2, z=2, imm=q(2.25)),
            Instruction128(Opcode.QMUL, flags=2, x=2, y=2, z=2, a=0, b=1),
            Instruction128(
                Opcode.FROUND,
                flags=3,
                x=2,
                y=2,
                z=2,
                a=10,
                b=20,
                imm=pack_shape((2, 2, 2)),
            ),
            Instruction128(Opcode.VERIFY, flags=4, x=2, y=2, z=2),
            Instruction128(Opcode.SEAL, x=2, y=2, z=2),
            Instruction128(Opcode.HALT),
        ],
        shape=(7, 1, 1),
    )

    print("ROM bytes:", len(program.to_bytes()))
    print("program digest:", program.digest)

    engine.load(program)
    report = engine.run(max_cycles=64)

    print("R2 arithmetic result:", engine.read_q(2, 2, 2, 2))
    print("R3 reconstruction MSE:", engine.read_q(2, 2, 2, 3))
    print("R4 ledger verification:", engine.read_q(2, 2, 2, 4))
    print("latent shape:", engine.fields[21].shape)
    print("cycles:", report.cycles)
    print("trace digest:", report.trace_digest)
    print("state digest:", report.final_state_digest)
    print("ledger valid:", engine.cloud.ledger.verify())


if __name__ == "__main__":
    main()
