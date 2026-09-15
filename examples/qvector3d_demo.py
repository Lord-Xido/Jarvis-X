"""Executable Q16.16 x Q16.16 x Q16.16 vector-bytecode demonstration."""

from jarvisx.qvector3d import QVectorField3D, q16_from_float
from jarvisx.qvector_bytecode3d import (
    DrMoagiQVectorBytecodeEngine3D,
    QVectorInstruction128,
    QVectorOpcode,
    QVectorProgram3D,
    pack_coordinate,
    pack_shape,
)


def main() -> None:
    field = QVectorField3D.from_vectors(
        [
            (float(x), float(y), float(z))
            for z in range(2)
            for y in range(2)
            for x in range(2)
        ],
        (2, 2, 2),
    )

    program = QVectorProgram3D.from_instructions(
        [
            QVectorInstruction128(
                QVectorOpcode.VFETCH,
                flags=0,
                x=2,
                y=2,
                z=2,
                a=10,
                imm=pack_coordinate(1, 1, 1),
            ),
            QVectorInstruction128(
                QVectorOpcode.VSCALE,
                flags=1,
                x=2,
                y=2,
                z=2,
                a=0,
                imm=q16_from_float(0.5) & 0xFFFFFFFF,
            ),
            QVectorInstruction128(
                QVectorOpcode.VFROUND,
                flags=2,
                x=2,
                y=2,
                z=2,
                a=10,
                b=20,
                imm=pack_shape((1, 1, 1)),
            ),
            QVectorInstruction128(
                QVectorOpcode.VFAUTO,
                flags=3,
                x=2,
                y=2,
                z=2,
                a=10,
                b=30,
                imm=q16_from_float(0.01) & 0xFFFFFFFF,
            ),
            QVectorInstruction128(QVectorOpcode.VERIFY, flags=4, x=2, y=2, z=2),
            QVectorInstruction128(QVectorOpcode.SEAL, x=2, y=2, z=2),
            QVectorInstruction128(QVectorOpcode.HALT),
        ],
        shape=(7, 1, 1),
    )

    engine = DrMoagiQVectorBytecodeEngine3D()
    engine.mount_vector_field(10, field)
    engine.load(program)
    report = engine.run()

    print("program:", report.program_digest)
    print("cycles:", report.cycles)
    print("fetched vector:", engine.read_vector_floats(2, 2, 2, 0))
    print("scaled vector:", engine.read_vector_floats(2, 2, 2, 1))
    print("round-trip axis MSE:", engine.read_vector_floats(2, 2, 2, 2))
    print("auto metrics [objective, component_mse, compression]:", engine.read_vector_floats(2, 2, 2, 3))
    print("ledger verification vector:", engine.read_vector_floats(2, 2, 2, 4))
    print("vector fields:", report.vector_field_handles)
    print("trace:", report.trace_digest)
    print("ledger valid:", engine.cloud.cloud.ledger.verify())


if __name__ == "__main__":
    main()
