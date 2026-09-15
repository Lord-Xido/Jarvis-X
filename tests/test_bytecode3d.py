from pathlib import Path

import pytest

from jarvisx.bytecode3d import (
    BytecodeProgram3D,
    DrMoagiBytecodeEngine3D,
    Instruction128,
    Opcode,
    pack_offset,
    pack_shape,
    q_from_float,
    q_to_float,
    unpack_offset,
    unpack_shape,
)
from jarvisx.cloud_os import DrMoagiCloudOS, Field3D


def _q(value: float) -> int:
    return q_from_float(value) & 0xFFFFFFFF


def test_instruction_is_exact_128_bit_round_trip() -> None:
    instruction = Instruction128(
        Opcode.QLOAD,
        flags=0xA2,
        x=3,
        y=4,
        z=5,
        a=6,
        b=7,
        imm=0xFEDCBA98,
    )
    encoded = instruction.to_bytes()
    assert len(encoded) == 16
    assert Instruction128.from_bytes(encoded) == instruction
    assert instruction.destination_register == 2


def test_shape_and_neighbor_immediates_round_trip() -> None:
    assert unpack_shape(pack_shape((7, 9, 11))) == (7, 9, 11)
    assert unpack_offset(pack_offset(-7, 0, 511)) == (-7, 0, 511)
    with pytest.raises(ValueError):
        pack_shape((0, 1, 1))
    with pytest.raises(ValueError):
        pack_offset(-513, 0, 0)


def test_q16_helpers_are_deterministic() -> None:
    assert q_to_float(q_from_float(3.375)) == 3.375
    with pytest.raises(ValueError):
        q_from_float(float("nan"))
    with pytest.raises(OverflowError):
        q_from_float(100_000.0)


def test_program_binary_and_3d_pc_mapping() -> None:
    instructions = [Instruction128(Opcode.NOP) for _ in range(7)] + [Instruction128(Opcode.HALT)]
    program = BytecodeProgram3D.from_instructions(instructions, shape=(2, 2, 2))
    assert program.pc_xyz(0) == (0, 0, 0)
    assert program.pc_xyz(1) == (1, 0, 0)
    assert program.pc_xyz(2) == (0, 1, 0)
    assert program.pc_xyz(4) == (0, 0, 1)
    assert BytecodeProgram3D.from_bytes(program.to_bytes(), program.shape) == program
    assert len(program.digest) == 64


def test_local_q16_arithmetic_and_trace_chain() -> None:
    coordinate = (9, 8, 7)
    program = BytecodeProgram3D.from_instructions(
        [
            Instruction128(Opcode.QLOAD, flags=0, x=9, y=8, z=7, imm=_q(1.5)),
            Instruction128(Opcode.QLOAD, flags=1, x=9, y=8, z=7, imm=_q(2.25)),
            Instruction128(Opcode.QADD, flags=2, x=9, y=8, z=7, a=0, b=1),
            Instruction128(Opcode.QSUB, flags=3, x=9, y=8, z=7, a=1, b=0),
            Instruction128(Opcode.QMUL, flags=4, x=9, y=8, z=7, a=0, b=1),
            Instruction128(Opcode.QDIV, flags=5, x=9, y=8, z=7, a=1, b=0),
            Instruction128(Opcode.QMOV, flags=6, x=9, y=8, z=7, a=4),
            Instruction128(Opcode.VERIFY, flags=7, x=9, y=8, z=7),
            Instruction128(Opcode.SEAL, x=9, y=8, z=7),
            Instruction128(Opcode.HALT),
        ]
    )
    engine = DrMoagiBytecodeEngine3D()
    engine.load(program)
    report = engine.run()

    assert engine.read_q(*coordinate, 2) == 3.75
    assert engine.read_q(*coordinate, 3) == 0.75
    assert engine.read_q(*coordinate, 4) == 3.375
    assert engine.read_q(*coordinate, 5) == 1.5
    assert engine.read_q(*coordinate, 6) == 3.375
    assert engine.read_q(*coordinate, 7) == 1.0
    assert report.halted
    assert report.cycles == 10
    assert len(engine.trace) == 10
    assert len(report.trace_digest) == 64
    assert engine.cloud.ledger.verify()
    assert any(record["event"] == "bytecode.sealed" for record in engine.cloud.ledger.records)


def test_neighbor_voxel_move() -> None:
    program = BytecodeProgram3D.from_instructions(
        [
            Instruction128(Opcode.QLOAD, flags=0, x=1, y=1, z=1, imm=_q(7.0)),
            Instruction128(
                Opcode.VMOVE,
                flags=3,
                x=2,
                y=1,
                z=1,
                a=0,
                imm=pack_offset(-1, 0, 0),
            ),
            Instruction128(Opcode.HALT),
        ]
    )
    engine = DrMoagiBytecodeEngine3D()
    engine.load(program)
    engine.run()
    assert engine.read_q(2, 1, 1, 3) == 7.0


def test_field_encode_decode_and_error() -> None:
    field = Field3D.from_values([2.0] * 8, (2, 2, 2))
    program = BytecodeProgram3D.from_instructions(
        [
            Instruction128(Opcode.FENCODE, a=10, b=11, imm=pack_shape((1, 1, 1))),
            Instruction128(Opcode.FDECODE, a=11, b=12, imm=pack_shape((2, 2, 2))),
            Instruction128(Opcode.FERR, flags=2, a=10, b=12),
            Instruction128(Opcode.HALT),
        ]
    )
    engine = DrMoagiBytecodeEngine3D()
    engine.mount_field(10, field)
    engine.load(program)
    engine.run()

    assert engine.fields[11].shape == (1, 1, 1)
    assert engine.fields[11].values == (2.0,)
    assert engine.fields[12] == field
    assert engine.read_q(0, 0, 0, 2) == 0.0


def test_cloud_round_trip_opcode_obeys_scheduler_and_commits() -> None:
    cloud = DrMoagiCloudOS()
    cloud.register_node("cloud-a", max_cells=64, max_concurrency=1)
    engine = DrMoagiBytecodeEngine3D(cloud=cloud)
    field = Field3D.from_values([3.0] * 8, (2, 2, 2))
    engine.mount_field(100, field)
    program = BytecodeProgram3D.from_instructions(
        [
            Instruction128(
                Opcode.FROUND,
                flags=4,
                x=4,
                y=3,
                z=2,
                a=100,
                b=200,
                imm=pack_shape((1, 1, 1)),
            ),
            Instruction128(Opcode.HALT),
        ]
    )
    engine.load(program)
    engine.run()

    assert engine.fields[200] == field
    assert engine.fields[201].shape == (1, 1, 1)
    assert engine.read_q(4, 3, 2, 4) == 0.0
    assert any(record["event"] == "job.committed" for record in cloud.ledger.records)


def test_cloud_auto_optimize_opcode_returns_selected_fields() -> None:
    field = Field3D.from_values([float(index % 3) for index in range(64)], (4, 4, 4))
    engine = DrMoagiBytecodeEngine3D(default_node_cells=128)
    engine.mount_field(1, field)
    program = BytecodeProgram3D.from_instructions(
        [
            Instruction128(
                Opcode.FAUTO,
                flags=5,
                x=1,
                y=2,
                z=3,
                a=1,
                b=20,
                imm=_q(0.05),
            ),
            Instruction128(Opcode.HALT),
        ]
    )
    engine.load(program)
    report = engine.run()

    assert 20 in report.field_handles
    assert 21 in report.field_handles
    assert engine.fields[20].shape == field.shape
    assert engine.read_q(1, 2, 3, 5) >= 0.0


def test_jnz_and_jmp_change_3d_rom_flow() -> None:
    program = BytecodeProgram3D.from_instructions(
        [
            Instruction128(Opcode.QLOAD, flags=0, x=1, y=1, z=1, imm=_q(1.0)),
            Instruction128(Opcode.JNZ, x=1, y=1, z=1, a=0, imm=3),
            Instruction128(Opcode.QLOAD, flags=1, x=1, y=1, z=1, imm=_q(99.0)),
            Instruction128(Opcode.JMP, imm=4),
            Instruction128(Opcode.HALT),
        ]
    )
    engine = DrMoagiBytecodeEngine3D()
    engine.load(program)
    engine.run()

    assert engine.read_q(1, 1, 1, 1) == 0.0
    assert engine.trace[1]["branch_taken"] is True
    assert engine.trace[2]["opcode"] == "JMP"


def test_cycle_limit_and_division_by_zero_are_explicit_failures() -> None:
    looping = BytecodeProgram3D.from_instructions([Instruction128(Opcode.JMP, imm=0)])
    engine = DrMoagiBytecodeEngine3D()
    engine.load(looping)
    with pytest.raises(RuntimeError, match="cycle limit"):
        engine.run(max_cycles=3)
    assert engine.cloud.ledger.records[-1]["event"] == "bytecode.run.failed"

    division = BytecodeProgram3D.from_instructions(
        [
            Instruction128(Opcode.QLOAD, flags=0, imm=_q(1.0)),
            Instruction128(Opcode.QDIV, flags=1, a=0, b=2),
            Instruction128(Opcode.HALT),
        ]
    )
    engine.load(division, reset_state=True)
    with pytest.raises(ZeroDivisionError):
        engine.run()


def test_invalid_binary_program_and_field_handle_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        Instruction128.from_bytes(b"short")
    with pytest.raises(ValueError, match="unknown opcode"):
        Instruction128.from_bytes(bytes.fromhex("aa" + "00" * 15))
    with pytest.raises(ValueError):
        BytecodeProgram3D.from_bytes(b"\x00" * 16, (2, 1, 1))

    engine = DrMoagiBytecodeEngine3D(ledger_path=tmp_path / "ledger.jsonl")
    program = BytecodeProgram3D.from_instructions(
        [Instruction128(Opcode.FENCODE, a=55, b=56, imm=pack_shape((1, 1, 1)))]
    )
    engine.load(program)
    with pytest.raises(KeyError, match="not mounted"):
        engine.run(max_cycles=1)
