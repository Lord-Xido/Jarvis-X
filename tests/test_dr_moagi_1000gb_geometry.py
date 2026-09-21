import math

import pytest

from jarvisx.dr_moagi_1000gb_geometry import GIB_BYTES, ThousandGBGeometry


def test_exact_decimal_capacity_and_binary_equivalent() -> None:
    g = ThousandGBGeometry()
    assert g.logical_bytes == 1_000_000_000_000
    assert g.logical_bits == 8_000_000_000_000
    assert g.decimal_gb == 1000
    assert g.decimal_tb == 1
    assert math.isclose(g.gibibytes, 931.3225746154785)
    assert g.thousand_gib_bytes() == 1_073_741_824_000
    assert g.thousand_gib_bits() == 8_589_934_592_000
    assert GIB_BYTES == 1_073_741_824


def test_hardware_breakdown_is_exact() -> None:
    h = ThousandGBGeometry().hardware_breakdown()
    assert h.minimum_address_bits == 40
    assert h.cache_line_bytes == 64
    assert h.cache_lines == 15_625_000_000
    assert h.page_bytes == 4096
    assert h.pages == 244_140_625


def test_minimum_address_width_boundary() -> None:
    assert ThousandGBGeometry.minimum_address_bits(1) == 0
    assert ThousandGBGeometry.minimum_address_bits(2) == 1
    assert ThousandGBGeometry.minimum_address_bits(2**39) == 39
    assert ThousandGBGeometry.minimum_address_bits(1_000_000_000_000) == 40
    with pytest.raises(ValueError):
        ThousandGBGeometry.minimum_address_bits(0)


def test_exact_3d_coordinate_mapping_round_trip() -> None:
    g = ThousandGBGeometry()
    probes = [
        (0, 0, 0),
        (1, 2, 3),
        (9999, 9999, 9999),
        (5000, 1234, 9876),
    ]
    for xyz in probes:
        offset = g.offset(*xyz)
        assert 0 <= offset < g.logical_bytes
        assert g.coordinate(offset) == xyz
    assert g.offset(9999, 9999, 9999) == g.logical_bytes - 1


def test_coordinate_bounds_fail_closed() -> None:
    g = ThousandGBGeometry()
    with pytest.raises(IndexError):
        g.offset(10_000, 0, 0)
    with pytest.raises(IndexError):
        g.offset(0, -1, 0)
    with pytest.raises(IndexError):
        g.coordinate(g.logical_bytes)


def test_tensor_capacity() -> None:
    t = ThousandGBGeometry().tensor_breakdown()
    assert t.byte_voxel_edge == 10_000
    assert t.byte_voxels == 1_000_000_000_000
    assert t.bit_voxel_edge == 20_000
    assert t.bit_voxels == 8_000_000_000_000
    assert t.fp32_values == 250_000_000_000
    assert t.fp16_values == 500_000_000_000
    assert t.int8_values == 1_000_000_000_000


def test_decimal_one_megabyte_tiles_cover_exactly() -> None:
    g = ThousandGBGeometry()
    tile = g.tile_cover(100)
    assert tile.tile_bytes == 1_000_000
    assert tile.tiles_per_axis == 100
    assert tile.covering_tiles == 1_000_000
    assert tile.full_tile_equivalent == 1_000_000
    assert not tile.has_partial_boundary_tiles


def test_64_cubed_tiles_report_partial_boundaries() -> None:
    tile = ThousandGBGeometry().tile_cover(64)
    assert tile.tile_bytes == 262_144
    assert tile.tiles_per_axis == 157
    assert tile.covering_tiles == 157**3
    assert math.isclose(tile.full_tile_equivalent, 1_000_000_000_000 / 262_144)
    assert tile.has_partial_boundary_tiles


def test_multimedia_equivalents_use_raw_assumptions() -> None:
    m = ThousandGBGeometry().multimedia_breakdown()
    assert m.raw_4k_rgb_frame_bytes == 24_883_200
    assert m.raw_4k_rgb_frames == 40_187
    assert math.isclose(m.raw_4k_rgb_seconds_at_24fps, 40_187 / 24)
    assert m.cd_stereo_bytes_per_second == 176_400
    assert math.isclose(m.cd_stereo_days, 65.61266481901399)


def test_report_keeps_virtual_extent_distinct_from_claims() -> None:
    report = ThousandGBGeometry().report()
    assert report["geometry"]["voxel_shape"] == [10_000, 10_000, 10_000]
    assert report["claims"]["dense_allocation_required"] is False
    assert report["claims"]["geometry_implies_compression"] is False
    assert report["claims"]["geometry_implies_throughput"] is False
