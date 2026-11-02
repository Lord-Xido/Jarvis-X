import pytest

from jarvisx.dm3d_rom import (
    AXIS_BYTES,
    HEADER_SIZE,
    INSTR_SIZE,
    LATENT_BYTES,
    OP_CORRECT,
    OP_DECODE,
    OP_ENCODE,
    OP_FILL_TILE,
    OP_HALT,
    OP_JUMP,
    OP_REFINE,
    OP_RESIDUAL,
    OP_VERIFY,
    TILE_BYTES,
    TILES_PER_AXIS,
    DM3DVM,
    build_rom,
    pack_instruction,
    parse_header,
)


def test_rom_header_declares_virtual_1gb_cubed_geometry() -> None:
    rom = build_rom([pack_instruction(OP_HALT)])
    header = parse_header(rom)

    assert header["axis_bytes"] == 1_000_000_000
    assert AXIS_BYTES**3 == 10**27
    assert header["code_offset"] == HEADER_SIZE
    assert len(rom) == HEADER_SIZE + INSTR_SIZE
    assert TILE_BYTES == 64**3
    assert LATENT_BYTES == 8**3


def test_sparse_roundtrip_verifies_exact_after_residual_correction() -> None:
    key = (TILES_PER_AXIS - 1, 2, 3)
    x, y, z = key
    program = [
        pack_instruction(OP_FILL_TILE, x, y, z, p0=1),
        pack_instruction(OP_ENCODE, x, y, z, p0=8),
        pack_instruction(OP_REFINE, x, y, z, p0=4, p1=1, p2=1),
        pack_instruction(OP_DECODE, x, y, z),
        pack_instruction(OP_RESIDUAL, x, y, z),
        pack_instruction(OP_CORRECT, x, y, z),
        pack_instruction(OP_VERIFY, x, y, z, flags=0x01),
        pack_instruction(OP_HALT),
    ]

    vm = DM3DVM(trace=False)
    stats = vm.run(build_rom(program))

    assert stats.verifies == 1
    assert stats.verified_exact == 1
    assert list(vm.volume.tiles) == [key]
    assert vm.corrected[key] == vm.volume.get(key)


def test_invalid_magic_is_rejected() -> None:
    rom = bytearray(build_rom([pack_instruction(OP_HALT)]))
    rom[0] ^= 0xFF

    try:
        parse_header(bytes(rom))
    except ValueError as exc:
        assert "magic" in str(exc)
    else:
        raise AssertionError("corrupt ROM magic was accepted")


def test_self_jump_exhausts_fuel_without_committing_any_state() -> None:
    vm = DM3DVM(trace=False, max_steps=5)
    program = build_rom(
        [pack_instruction(OP_FILL_TILE, 0, 0, 0), pack_instruction(OP_JUMP, p0=1)]
    )

    with pytest.raises(RuntimeError, match="step budget exhausted"):
        vm.run(program)

    assert vm.volume.tiles == {}
    assert vm.latent == {}
    assert vm.stats.instructions == 0
    assert vm.pc == 0


def test_resident_tile_limit_rejects_candidate_without_losing_incumbent() -> None:
    vm = DM3DVM(trace=False, max_active_tiles=1)
    vm.run(build_rom([pack_instruction(OP_FILL_TILE, 0, 0, 0, p0=1), pack_instruction(OP_HALT)]))
    previous = bytes(vm.volume.get((0, 0, 0)))
    old_stats = vm.stats.instructions

    with pytest.raises(RuntimeError, match="active-tile budget exhausted"):
        vm.run(build_rom([pack_instruction(OP_FILL_TILE, 1, 0, 0), pack_instruction(OP_HALT)]))

    assert list(vm.volume.tiles) == [(0, 0, 0)]
    assert vm.volume.get((0, 0, 0)) == previous
    assert vm.stats.instructions == old_stats


def test_tile_sweep_budget_prevents_excessive_refinement() -> None:
    vm = DM3DVM(trace=False, max_tile_sweeps=2)
    program = build_rom(
        [
            pack_instruction(OP_FILL_TILE, p0=1),
            pack_instruction(OP_ENCODE, p0=8),
            pack_instruction(OP_REFINE, p0=4),
            pack_instruction(OP_HALT),
        ]
    )
    with pytest.raises(RuntimeError, match="tile-work budget exhausted"):
        vm.run(program)
    assert not vm.volume.tiles
    assert not vm.latent


def test_invalid_header_fields_and_trailing_data_fail_closed() -> None:
    rom = bytearray(build_rom([pack_instruction(OP_HALT)]))
    rom[8:12] = (999).to_bytes(4, "little")
    with pytest.raises(ValueError, match="version"):
        parse_header(bytes(rom))

    valid = build_rom([pack_instruction(OP_HALT)])
    with pytest.raises(ValueError, match="instruction section length"):
        parse_header(valid + b"trailing")


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_execution_limits_require_positive_integers(limit: object) -> None:
    with pytest.raises(ValueError, match="max_steps"):
        DM3DVM(trace=False, max_steps=limit)  # type: ignore[arg-type]
