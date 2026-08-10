import pytest

from jarvisx.cloud_os import DrMoagiCloudOS
from jarvisx.qvector3d import (
    QVector3Q16,
    QVectorAutoencoder3D,
    QVectorField3D,
    q16_from_float,
)
from jarvisx.qvector_cloud import DrMoagiQVectorCloudEngine3D
from jarvisx.qvector_bytecode3d import (
    DrMoagiQVectorBytecodeEngine3D,
    QVectorInstruction128,
    QVectorOpcode,
    QVectorProgram3D,
    pack_coordinate,
    pack_offset,
    pack_shape,
)


def sample_field() -> QVectorField3D:
    return QVectorField3D.from_vectors(
        [
            (0, 0, 0),
            (1, 2, 3),
            (2, 4, 6),
            (3, 6, 9),
            (4, 8, 12),
            (5, 10, 15),
            (6, 12, 18),
            (7, 14, 21),
        ],
        (2, 2, 2),
    )


def test_qvector_cell_is_12_bytes_and_round_trips() -> None:
    vector = QVector3Q16.from_floats(1.5, -2.25, 3.125)
    assert len(vector.to_bytes()) == 12
    assert QVector3Q16.from_bytes(vector.to_bytes()) == vector
    assert vector.to_floats() == (1.5, -2.25, 3.125)


def test_qvector_field_binary_round_trip_and_digest() -> None:
    field = sample_field()
    encoded = field.to_bytes()
    assert len(encoded) == field.cells * 12
    assert QVectorField3D.from_bytes(encoded, field.shape) == field
    assert QVectorField3D.from_bytes(encoded, field.shape).digest == field.digest


def test_constant_vector_field_round_trip_is_exact() -> None:
    field = QVectorField3D.from_vectors([(1.25, -2.5, 4.0)] * 64, (4, 4, 4))
    result = QVectorAutoencoder3D().round_trip(field, (1, 1, 1))
    assert result.reconstruction == field
    assert result.axis_mse == (0.0, 0.0, 0.0)
    assert result.component_mse == 0.0
    assert result.compression_ratio == pytest.approx(1 / 64)


def test_vector_error_metrics_are_per_axis_and_vector() -> None:
    source = QVectorField3D.from_vectors([(0, 0, 0), (2, 4, 6)], (2, 1, 1))
    decoded = QVectorField3D.from_vectors([(1, 1, 1), (1, 3, 5)], (2, 1, 1))
    axis, component, vector = QVectorAutoencoder3D.error_metrics(source, decoded)
    assert axis == pytest.approx((1.0, 1.0, 1.0))
    assert component == pytest.approx(1.0)
    assert vector == pytest.approx(3.0)


def test_cloud_capacity_counts_three_scalar_lanes_per_vector_cell() -> None:
    cloud = DrMoagiCloudOS()
    cloud.register_node("small", max_cells=23)
    engine = DrMoagiQVectorCloudEngine3D(cloud=cloud)
    with pytest.raises(RuntimeError):
        engine.round_trip(sample_field(), (1, 1, 1), request_id="too-large")

    cloud.register_node("exact", max_cells=24)
    job = engine.round_trip(sample_field(), (1, 1, 1), request_id="fits")
    assert job.status == "succeeded"
    assert job.node_id == "exact"


def test_auto_optimizer_can_select_lossless_full_resolution() -> None:
    engine = DrMoagiQVectorCloudEngine3D()
    field = sample_field()
    job = engine.auto_optimize(
        field,
        request_id="lossless",
        complexity_weight=0.0,
        candidates=[(1, 1, 1), (2, 2, 2)],
    )
    assert job.result is not None
    assert job.result["selected_latent_shape"] == [2, 2, 2]
    assert job.result["component_mse"] == 0.0


def test_qvector_instruction_codec_is_exactly_128_bits() -> None:
    instruction = QVectorInstruction128(
        QVectorOpcode.VFETCH,
        flags=3,
        x=4,
        y=5,
        z=6,
        a=9,
        b=10,
        imm=pack_coordinate(1, 2, 3),
    )
    encoded = instruction.to_bytes()
    assert len(encoded) == 16
    assert QVectorInstruction128.from_bytes(encoded) == instruction


def test_qvector_bytecode_fetch_scale_and_neighbor_move() -> None:
    field = sample_field()
    program = QVectorProgram3D.from_instructions(
        [
            QVectorInstruction128(
                QVectorOpcode.VFETCH,
                flags=0,
                x=1,
                y=1,
                z=1,
                a=10,
                imm=pack_coordinate(1, 1, 1),
            ),
            QVectorInstruction128(
                QVectorOpcode.VSCALE,
                flags=1,
                x=1,
                y=1,
                z=1,
                a=0,
                imm=q16_from_float(0.5) & 0xFFFFFFFF,
            ),
            QVectorInstruction128(
                QVectorOpcode.VFETCH,
                flags=0,
                x=2,
                y=1,
                z=1,
                a=10,
                imm=pack_coordinate(0, 0, 0),
            ),
            QVectorInstruction128(
                QVectorOpcode.VMOVE,
                flags=2,
                x=2,
                y=1,
                z=1,
                a=1,
                imm=pack_offset(-1, 0, 0),
            ),
            QVectorInstruction128(QVectorOpcode.HALT),
        ]
    )
    engine = DrMoagiQVectorBytecodeEngine3D()
    engine.mount_vector_field(10, field)
    engine.load(program)
    report = engine.run()
    assert report.halted
    assert engine.read_vector_floats(1, 1, 1, 1) == (3.5, 7.0, 10.5)
    assert engine.read_vector_floats(2, 1, 1, 2) == (3.5, 7.0, 10.5)


def test_vfround_writes_reconstruction_latent_and_axis_error() -> None:
    field = sample_field()
    program = QVectorProgram3D.from_instructions(
        [
            QVectorInstruction128(
                QVectorOpcode.VFROUND,
                flags=0,
                x=3,
                y=3,
                z=3,
                a=10,
                b=20,
                imm=pack_shape((1, 1, 1)),
            ),
            QVectorInstruction128(QVectorOpcode.HALT),
        ]
    )
    engine = DrMoagiQVectorBytecodeEngine3D()
    engine.mount_vector_field(10, field)
    engine.load(program)
    engine.run()
    assert 20 in engine.vector_fields
    assert 21 in engine.vector_fields
    assert engine.vector_fields[20].shape == field.shape
    assert engine.vector_fields[21].shape == (1, 1, 1)
    assert engine.read_vector_floats(3, 3, 3, 0) == pytest.approx((5.25, 21.0, 47.25))


def test_vfauto_verify_and_seal_commit_to_omega_ledger() -> None:
    field = sample_field()
    program = QVectorProgram3D.from_instructions(
        [
            QVectorInstruction128(
                QVectorOpcode.VFAUTO,
                flags=0,
                x=4,
                y=4,
                z=4,
                a=10,
                b=30,
                imm=q16_from_float(0.01) & 0xFFFFFFFF,
            ),
            QVectorInstruction128(QVectorOpcode.VERIFY, flags=1, x=4, y=4, z=4),
            QVectorInstruction128(QVectorOpcode.SEAL, x=4, y=4, z=4),
            QVectorInstruction128(QVectorOpcode.HALT),
        ]
    )
    engine = DrMoagiQVectorBytecodeEngine3D()
    engine.mount_vector_field(10, field)
    engine.load(program)
    report = engine.run()
    assert report.halted
    assert engine.read_vector_floats(4, 4, 4, 1) == (1.0, 1.0, 1.0)
    assert engine.cloud.cloud.ledger.verify()
    events = [record["event"] for record in engine.cloud.cloud.ledger.records]
    assert "qvector.bytecode.sealed" in events
    assert "qvector.bytecode.run.committed" in events


def test_qvector_cycle_limit_fails_closed() -> None:
    program = QVectorProgram3D.from_instructions(
        [QVectorInstruction128(QVectorOpcode.JMP, imm=0)]
    )
    engine = DrMoagiQVectorBytecodeEngine3D()
    engine.load(program)
    with pytest.raises(RuntimeError, match="cycle limit"):
        engine.run(max_cycles=4)
    assert engine.cloud.cloud.ledger.verify()


def test_qvector_trace_is_deterministic_for_same_program_and_input() -> None:
    field = sample_field()
    program = QVectorProgram3D.from_instructions(
        [
            QVectorInstruction128(
                QVectorOpcode.VFETCH,
                flags=0,
                x=1,
                y=2,
                z=3,
                a=10,
                imm=pack_coordinate(1, 0, 1),
            ),
            QVectorInstruction128(QVectorOpcode.HALT),
        ]
    )
    digests = []
    for _ in range(2):
        engine = DrMoagiQVectorBytecodeEngine3D()
        engine.mount_vector_field(10, field)
        engine.load(program)
        digests.append(engine.run().trace_digest)
    assert digests[0] == digests[1]
