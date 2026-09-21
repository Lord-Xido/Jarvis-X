import json

import pytest

from jarvisx.virtual3d import OperationalMode, Virtual3DComputer, VolumeGeometry


def test_6400_cubed_gigabyte_cells_equal_262_144_exabytes_decimal():
    geometry = VolumeGeometry(
        extent=(6400, 6400, 6400),
        block_shape=(1024, 1024, 1024),
        cell_bytes=1_000_000_000,
    )

    assert geometry.logical_cells == 262_144_000_000
    assert geometry.logical_capacity_bytes == 262_144_000_000_000_000_000
    assert geometry.block_grid_shape == (7, 7, 7)
    assert geometry.maximum_block_count == 343


def test_byte_addressable_interpretation_is_262_144_gigabytes_decimal():
    geometry = VolumeGeometry(
        extent=(6400, 6400, 6400),
        block_shape=(1024, 1024, 1024),
        cell_bytes=1,
    )

    assert geometry.logical_capacity_bytes == 262_144_000_000


def test_mapping_handles_regular_and_boundary_blocks():
    geometry = VolumeGeometry(
        extent=(6400, 6400, 6400),
        block_shape=(1024, 1024, 1024),
    )

    address = geometry.map_coordinate((6399, 1024, 2047))

    assert address.block == (6, 1, 1)
    assert address.offset == (255, 0, 1023)
    assert geometry.effective_block_shape((6, 6, 6)) == (256, 256, 256)


def test_reads_do_not_allocate_and_writes_allocate_lazily():
    computer = Virtual3DComputer(
        VolumeGeometry(
            extent=(16, 16, 16),
            block_shape=(4, 4, 4),
            cell_bytes=8,
        )
    )

    assert computer.read((5, 5, 5)) == b""
    assert computer.statistics().allocated_blocks == 0

    computer.write((5, 5, 5), b"Jarvis")

    assert computer.read((5, 5, 5)) == b"Jarvis"
    assert computer.statistics().allocated_blocks == 1
    assert computer.statistics().allocated_cells == 1


def test_empty_write_releases_the_last_cell_and_block():
    computer = Virtual3DComputer(
        VolumeGeometry(
            extent=(8, 8, 8),
            block_shape=(4, 4, 4),
            cell_bytes=4,
        )
    )
    computer.write((1, 1, 1), b"data")
    computer.write((1, 1, 1), b"")

    assert computer.statistics().allocated_blocks == 0


def test_payload_cannot_exceed_cell_capacity():
    computer = Virtual3DComputer(
        VolumeGeometry(
            extent=(2, 2, 2),
            block_shape=(1, 1, 1),
            cell_bytes=2,
        )
    )

    with pytest.raises(ValueError):
        computer.write((0, 0, 0), b"abc")


def test_rom_round_trip_restores_geometry_values_and_fingerprint(tmp_path):
    computer = Virtual3DComputer(
        VolumeGeometry(
            extent=(32, 32, 32),
            block_shape=(8, 8, 8),
            cell_bytes=16,
        ),
        mode=OperationalMode.OFFLINE,
    )
    computer.write((1, 2, 3), b"alpha")
    computer.write((31, 31, 31), b"omega")

    path = tmp_path / "state.jxrom"
    fingerprint = computer.save_rom(str(path))
    restored = Virtual3DComputer.load_rom(str(path))

    envelope = json.loads(path.read_text())
    assert envelope["fingerprint"] == fingerprint
    assert restored.geometry == computer.geometry
    assert restored.read((1, 2, 3)) == b"alpha"
    assert restored.read((31, 31, 31)) == b"omega"


def test_rom_tampering_is_detected():
    computer = Virtual3DComputer(
        VolumeGeometry(
            extent=(4, 4, 4),
            block_shape=(2, 2, 2),
            cell_bytes=8,
        )
    )
    computer.write((0, 0, 0), b"safe")
    envelope = json.loads(computer.to_rom_bytes().decode("utf-8"))
    envelope["state"]["writes"] = 999

    with pytest.raises(ValueError, match="fingerprint"):
        Virtual3DComputer.from_rom_bytes(
            json.dumps(envelope).encode("utf-8")
        )


def test_layout_optimizer_can_reblock_without_changing_values():
    computer = Virtual3DComputer(
        VolumeGeometry(
            extent=(16, 16, 16),
            block_shape=(8, 8, 8),
            cell_bytes=8,
        ),
        mode=OperationalMode.ADAPTIVE,
    )
    computer.write((0, 0, 0), b"a")
    computer.write((15, 15, 15), b"b")

    report = computer.optimize_layout([(4, 4, 4), (16, 16, 16)])

    assert report.selected.block_shape in {
        (4, 4, 4),
        (8, 8, 8),
        (16, 16, 16),
    }
    assert computer.read((0, 0, 0)) == b"a"
    assert computer.read((15, 15, 15)) == b"b"
