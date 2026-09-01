from jarvisx.qvector3d import Q_ONE, QVectorField3D, q16_from_float
from jarvisx.qvector_bytecode3d import DrMoagiQVectorBytecodeEngine3D
from jarvisx.qvector_field_bytecode import (
    DrMoagiQVectorFieldBytecodeEngine3D,
    QVectorFieldInstruction128,
    QVectorFieldOpcode,
    QVectorFieldProgram3D,
)
from jarvisx.qvector_v2 import QRoundMode, QScalarKernel3D


def field_from_function(shape, function):
    sx, sy, sz = shape
    vectors = []
    for z in range(sz):
        for y in range(sy):
            for x in range(sx):
                vectors.append(function(x, y, z))
    return QVectorField3D.from_vectors(vectors, shape)


def test_field_instruction_codec_is_exactly_128_bits() -> None:
    instruction = QVectorFieldInstruction128(
        QVectorFieldOpcode.VCURL,
        flags=3,
        x=1,
        y=2,
        z=3,
        a=10,
        b=20,
        imm=q16_from_float(1.0) & 0xFFFFFFFF,
    )
    encoded = instruction.to_bytes()
    assert len(encoded) == 16
    assert QVectorFieldInstruction128.from_bytes(encoded) == instruction


def test_field_coprocessor_shares_state_with_base_vm() -> None:
    base = DrMoagiQVectorBytecodeEngine3D()
    engine = DrMoagiQVectorFieldBytecodeEngine3D(base)
    field = field_from_function((3, 3, 1), lambda x, y, z: (-y, x, 0))
    engine.mount_vector_field(10, field)
    program = QVectorFieldProgram3D.from_instructions(
        [
            QVectorFieldInstruction128(
                QVectorFieldOpcode.VCURL,
                a=10,
                b=20,
                imm=q16_from_float(1.0) & 0xFFFFFFFF,
            ),
            QVectorFieldInstruction128(QVectorFieldOpcode.HALT),
        ]
    )
    engine.load(program)
    report = engine.run()
    assert report.halted
    assert base.vector_fields[20].at(1, 1, 0).to_floats() == (0.0, 0.0, 2.0)


def test_field_coprocessor_convolution_and_status_register() -> None:
    base = DrMoagiQVectorBytecodeEngine3D()
    engine = DrMoagiQVectorFieldBytecodeEngine3D(base)
    field = field_from_function((2, 2, 2), lambda x, y, z: (x + y + z, x, -z))
    engine.mount_vector_field(10, field)
    engine.mount_kernel(7, QScalarKernel3D.identity())
    program = QVectorFieldProgram3D.from_instructions(
        [
            QVectorFieldInstruction128(
                QVectorFieldOpcode.VCONV3D,
                a=10,
                b=30,
                imm=7,
            ),
            QVectorFieldInstruction128(
                QVectorFieldOpcode.QSTATUS,
                flags=2,
                x=5,
                y=5,
                z=5,
            ),
            QVectorFieldInstruction128(QVectorFieldOpcode.HALT),
        ]
    )
    engine.load(program)
    engine.run()
    assert base.vector_fields[30] == field
    assert base.read_vector_floats(5, 5, 5, 2) == (0.0, 0.0, 0.0)


def test_field_coprocessor_rounding_mode_is_architectural_state() -> None:
    engine = DrMoagiQVectorFieldBytecodeEngine3D()
    program = QVectorFieldProgram3D.from_instructions(
        [
            QVectorFieldInstruction128(
                QVectorFieldOpcode.QSETMODE,
                imm=int(QRoundMode.TRUNCATE),
            ),
            QVectorFieldInstruction128(QVectorFieldOpcode.HALT),
        ]
    )
    engine.load(program)
    engine.run()
    assert engine.ops.policy.rounding == QRoundMode.TRUNCATE


def test_field_verify_and_seal_commit_to_omega() -> None:
    engine = DrMoagiQVectorFieldBytecodeEngine3D()
    program = QVectorFieldProgram3D.from_instructions(
        [
            QVectorFieldInstruction128(
                QVectorFieldOpcode.VERIFY,
                flags=0,
                x=1,
                y=1,
                z=1,
            ),
            QVectorFieldInstruction128(QVectorFieldOpcode.SEAL),
            QVectorFieldInstruction128(QVectorFieldOpcode.HALT),
        ]
    )
    engine.load(program)
    report = engine.run()
    assert report.halted
    assert engine.base.read_vector_floats(1, 1, 1, 0) == (1.0, 1.0, 1.0)
    events = [record["event"] for record in engine.base.cloud.cloud.ledger.records]
    assert "qvector.field.bytecode.sealed" in events
    assert "qvector.field.bytecode.run.committed" in events


def test_zero_spacing_defaults_to_one_voxel() -> None:
    base = DrMoagiQVectorBytecodeEngine3D()
    engine = DrMoagiQVectorFieldBytecodeEngine3D(base)
    field = field_from_function((3, 1, 1), lambda x, y, z: (x, 0, 0))
    engine.mount_vector_field(1, field)
    program = QVectorFieldProgram3D.from_instructions(
        [
            QVectorFieldInstruction128(QVectorFieldOpcode.VGRADX, a=1, b=2, imm=0),
            QVectorFieldInstruction128(QVectorFieldOpcode.HALT),
        ]
    )
    engine.load(program)
    engine.run()
    assert base.vector_fields[2].at(1, 0, 0).x == Q_ONE
