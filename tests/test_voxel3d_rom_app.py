from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "apps" / "moagi-unified-3d" / "voxel3d_rom_runtime.py"

spec = importlib.util.spec_from_file_location("voxel3d_rom_runtime_app", MODULE_PATH)
assert spec is not None and spec.loader is not None
voxel3d = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = voxel3d
spec.loader.exec_module(voxel3d)


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


def test_voxel3d_rom_reference_layer_smoke() -> None:
    rom = voxel3d.Voxel3DRomImage(sample_rom())
    runtime = voxel3d.JarvisXVoxelRuntime(rom)

    anchor_names = [anchor.name for anchor in rom.anchors()]
    assert "GLSL_SVO_ENGINE" in anchor_names
    assert "AUTO_EXEC_LOOP_0" in anchor_names
    assert "ENGINE_STATE_RUN" in anchor_names
    assert "VOXEL_ENGINE_EOF" in anchor_names

    words = rom.words_after_anchor("AUTO_EXEC_LOOP_0", count=3)
    assert [word.value for word in words] == [0x3E, 0x3F, 0xFD]
    assert rom.telemetry()["opcode_semantics"] == "UNSPECIFIED"
    assert rom.telemetry()["native_executable"] is False

    rejected = runtime.step(residual_verified=False)
    accepted = runtime.step(residual_verified=True)
    assert rejected.correction_committed is False
    assert accepted.correction_committed is True
