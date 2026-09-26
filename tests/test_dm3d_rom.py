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
    OP_INWARD_LOOP_K,
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


def test_fused_inward_loop_fast_forwards_only_after_exact_fixed_point() -> None:
    key = (7, 0, 0)
    x, y, z = key
    logical_iterations = 1_000_000
    logical_lanes = 1_000

    program = [
        pack_instruction(OP_FILL_TILE, x, y, z, p0=1),
        pack_instruction(OP_ENCODE, x, y, z, p0=8),
        pack_instruction(
            OP_INWARD_LOOP_K,
            x,
            y,
            z,
            p0=logical_iterations,
            p1=8,
            p2=logical_lanes,
            flags=0x01,
        ),
        pack_instruction(OP_HALT),
    ]

    stats = DM3DVM(trace=False).run(build_rom(program))

    assert stats.logical_refine_iterations == logical_iterations
    assert stats.logical_voxel_updates == logical_iterations * logical_lanes
    assert 1 <= stats.physical_refine_steps <= 8
    assert stats.elided_fixed_point_steps == logical_iterations - stats.physical_refine_steps
    assert stats.elided_logical_updates == (
        logical_iterations - stats.physical_refine_steps
    ) * logical_lanes
