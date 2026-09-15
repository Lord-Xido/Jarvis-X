"""Smoke demonstration for the Q16.16x3 geometric field coprocessor."""

from jarvisx.qvector3d import QVectorField3D, q16_from_float
from jarvisx.qvector_field_bytecode import (
    DrMoagiQVectorFieldBytecodeEngine3D,
    QVectorFieldInstruction128,
    QVectorFieldOpcode,
    QVectorFieldProgram3D,
)
from jarvisx.qvector_v2 import PackedQVectorField3D, QScalarKernel3D


def rotation_field() -> QVectorField3D:
    vectors = []
    for z in range(1):
        for y in range(3):
            for x in range(3):
                vectors.append((-y, x, 0))
    return QVectorField3D.from_vectors(vectors, (3, 3, 1))


def main() -> None:
    source = rotation_field()
    packed = PackedQVectorField3D.from_field(source)

    engine = DrMoagiQVectorFieldBytecodeEngine3D()
    engine.mount_vector_field(10, source)
    engine.mount_kernel(1, QScalarKernel3D.identity())
    program = QVectorFieldProgram3D.from_instructions(
        [
            QVectorFieldInstruction128(
                QVectorFieldOpcode.VCURL,
                a=10,
                b=20,
                imm=q16_from_float(1.0) & 0xFFFFFFFF,
            ),
            QVectorFieldInstruction128(
                QVectorFieldOpcode.VCONV3D,
                a=20,
                b=21,
                imm=1,
            ),
            QVectorFieldInstruction128(
                QVectorFieldOpcode.QSTATUS,
                flags=0,
                x=1,
                y=1,
                z=1,
            ),
            QVectorFieldInstruction128(QVectorFieldOpcode.VERIFY, flags=1),
            QVectorFieldInstruction128(QVectorFieldOpcode.SEAL),
            QVectorFieldInstruction128(QVectorFieldOpcode.HALT),
        ]
    )
    engine.load(program)
    report = engine.run()

    print("protocol:", report.protocol)
    print("cycles:", report.cycles)
    print("packed bytes:", packed.raw_bytes)
    print("curl center:", engine.vector_fields[20].at(1, 1, 0).to_floats())
    print("numeric status:", report.numeric_status)
    print("ledger valid:", engine.base.cloud.cloud.ledger.verify())
    print("trace:", report.trace_digest)


if __name__ == "__main__":
    main()
