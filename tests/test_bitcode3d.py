import math

import pytest

from jarvisx import bitcode3d as b3


def test_instruction32_round_trip_and_bounds() -> None:
    instruction = b3.Instruction32(
        opcode=b3.Opcode.ENCODE3D, dst=1, src1=2, src2=3, imm=-7
    )
    encoded = instruction.encode()

    assert 0 <= encoded <= 0xFFFFFFFF
    assert b3.Instruction32.decode(encoded) == instruction

    with pytest.raises(ValueError, match="dst"):
        b3.Instruction32(opcode=b3.Opcode.NOP, dst=16).encode()
    with pytest.raises(ValueError, match="signed 12 bits"):
        b3.Instruction32(opcode=b3.Opcode.NOP, imm=2048).encode()
    with pytest.raises(ValueError, match="unsigned 32 bits"):
        b3.Instruction32.decode(1 << 32)
    with pytest.raises(ValueError, match="unknown opcode"):
        b3.Instruction32.decode(0xFE000000)


def test_q16_quantization_saturates_and_rejects_non_finite_values() -> None:
    value, clipped = b3.quantize_q16_16(1.5)
    assert value == int(1.5 * b3.Q_SCALE)
    assert clipped is False
    assert b3.dequantize_q16_16(value) == 1.5

    high, high_clipped = b3.quantize_q16_16(1e20)
    low, low_clipped = b3.quantize_q16_16(-1e20)
    assert high == b3.Q_MAX and high_clipped
    assert low == b3.Q_MIN and low_clipped

    with pytest.raises(ValueError, match="finite"):
        b3.quantize_q16_16(math.inf)


def test_end_to_end_3d_contract_expand_verify_is_deterministic() -> None:
    runtime = b3.BitCode3DRuntime()
    first = runtime.execute(list(range(8)), (2, 2, 2), pool=2, tolerance=4.0)
    second = runtime.execute(list(range(8)), (2, 2, 2), pool=2, tolerance=4.0)

    assert first.input_shape == (2, 2, 2)
    assert first.latent_shape == (1, 1, 1)
    assert first.latent == (3.5,)
    assert first.reconstructed == (3.5,) * 8
    assert first.verification.mse == 5.25
    assert first.verification.max_abs_error == 3.5
    assert first.verification.passed is True
    assert first.verification.checksum_sha256 == second.verification.checksum_sha256
    assert first.bytecode == second.bytecode
    assert first.telemetry.cycles == len(first.bytecode) == 11
    assert first.telemetry.active_cells == 8
    assert first.telemetry.latent_cells == 1
    assert [step.instruction.opcode for step in first.spatial_program] == [
        b3.Opcode.NET_RX,
        b3.Opcode.HOST_STAGE,
        b3.Opcode.Q16_CONVERT,
        b3.Opcode.PACK3D,
        b3.Opcode.ENCODE3D,
        b3.Opcode.LATENT_WRITE,
        b3.Opcode.DECODE3D,
        b3.Opcode.VERIFY,
        b3.Opcode.TELEMETRY,
        b3.Opcode.EMIT,
        b3.Opcode.HALT,
    ]


def test_pool_one_is_exact_q16_roundtrip() -> None:
    result = b3.BitCode3DRuntime().execute(
        [0.125, -0.5, 3.25], (3, 1, 1), pool=1, tolerance=0.0
    )

    assert result.latent_shape == (3, 1, 1)
    assert result.reconstructed == (0.125, -0.5, 3.25)
    assert result.verification.mse == 0.0
    assert result.verification.passed is True
    payload = result.as_payload()
    assert payload["bytecode"][0] == "0x10000000"
    assert payload["spatial_program"][4]["opcode"] == "ENCODE3D"


def test_runtime_bounds_and_validation_fail_closed() -> None:
    with pytest.raises(ValueError, match="positive"):
        b3.BitCode3DRuntime(max_voxels=0)

    runtime = b3.BitCode3DRuntime(max_voxels=4)
    with pytest.raises(ValueError, match="three positive dimensions"):
        runtime.execute([1.0], (1, 0, 1))
    with pytest.raises(ValueError, match="exceeds runtime limit"):
        runtime.execute([0.0] * 8, (2, 2, 2))
    with pytest.raises(ValueError, match="does not match"):
        b3.BitCode3DRuntime().execute([1.0], (2, 1, 1))
    with pytest.raises(ValueError, match="tolerance"):
        b3.BitCode3DRuntime().execute([1.0], (1, 1, 1), tolerance=-1.0)
    with pytest.raises(ValueError, match="pool"):
        b3.compile_spatial_program(0)
