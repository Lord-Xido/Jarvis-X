from voxel3d_rom_runtime import (
    JarvisXVoxelRuntime,
    RomFormatError,
    Voxel3DRomImage,
)


def sample_rom() -> bytes:
    return (
        b"\x7fVOXEL3D"
        + b"\x01\x00\x00\x00"
        + b"GLSL_SVO_ENGINE\x00"
        + b"AUTO_EXEC_LOOP_0\x00\x00\x00\x00"
        + (0x3E).to_bytes(4, "little")
        + (0x3F).to_bytes(4, "little")
        + (0xFD).to_bytes(4, "little")
        + b"ENGINE_STATE_RUN\x00"
        + b"VOXEL_ENGINE_EOF"
    )


def test_magic_validation():
    rom = Voxel3DRomImage(sample_rom())
    assert rom.data.startswith(b"\x7fVOXEL3D")


def test_invalid_magic_is_rejected():
    try:
        Voxel3DRomImage(b"NOTVOXEL")
    except RomFormatError:
        pass
    else:
        raise AssertionError("invalid magic should raise RomFormatError")


def test_anchor_discovery_preserves_offsets():
    rom = Voxel3DRomImage(sample_rom())
    names = [anchor.name for anchor in rom.anchors()]
    assert "GLSL_SVO_ENGINE" in names
    assert "AUTO_EXEC_LOOP_0" in names
    assert "ENGINE_STATE_RUN" in names
    assert "VOXEL_ENGINE_EOF" in names


def test_auto_exec_words_are_raw_not_interpreted():
    rom = Voxel3DRomImage(sample_rom())
    words = rom.words_after_anchor("AUTO_EXEC_LOOP_0", count=3)
    assert [word.value for word in words] == [0x3E, 0x3F, 0xFD]
    assert rom.telemetry()["opcode_semantics"] == "UNSPECIFIED"
    assert rom.telemetry()["native_executable"] is False


def test_ctr_gate_controls_correction_commit():
    runtime = JarvisXVoxelRuntime(Voxel3DRomImage(sample_rom()))
    rejected = runtime.step(residual_verified=False)
    accepted = runtime.step(residual_verified=True)

    assert rejected.verified is False
    assert rejected.correction_committed is False
    assert accepted.verified is True
    assert accepted.correction_committed is True
    assert accepted.cycle == 2
