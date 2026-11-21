"""Lossless 3D image ISA regressions without ANN/GPU dependencies."""
import io
import random
import zipfile

import pytest

from jarvisx.image_isa3d import (
    CUBE_BYTES, CUBES_PER_TILE, decode_archive, decode_cube, encode_cube,
    encode_ppm, image_address, inverse_xor_shear, make_ppm, parse_ppm, sha256,
    xor_shear,
)


def test_8k_image_space_boundaries():
    assert image_address(0, 0, 0) == 0
    assert image_address(59, 33, 999) == 2_039_999
    with pytest.raises(ValueError):
        image_address(60, 0, 0)
    with pytest.raises(ValueError):
        image_address(0, 0, 1000)


@pytest.mark.parametrize("seed", list(range(12)))
def test_gf2_inverse_for_random_binary_cubes(seed):
    raw = random.Random(seed).randbytes(CUBE_BYTES)
    assert inverse_xor_shear(xor_shear(raw)) == raw
    assert decode_cube(encode_cube(raw)) == raw


def test_all_one_all_zero_cubes():
    for raw in (bytes(CUBE_BYTES), bytes([255]) * CUBE_BYTES):
        assert decode_cube(encode_cube(raw)) == raw


@pytest.mark.parametrize("width,height", [(1, 1), (7, 9), (128, 128), (129, 131)])
def test_rgb8_ppm_exact_roundtrip_and_edge_tiles(width, height):
    rng = random.Random(width * 1000 + height)
    source = make_ppm(width, height, rng.randbytes(width * height * 3))
    frame = encode_ppm(source, slot=42)
    restored, meta = decode_archive(frame)
    assert restored == source
    assert meta["pixels_sha256"] == sha256(parse_ppm(source)[2])
    assert meta["source_sha256"] == sha256(source)
    assert meta["cubes_per_tile"] == CUBES_PER_TILE


def _mutate_archive(archive, name, edit):
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(archive)) as source, zipfile.ZipFile(out, "w") as destination:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename == name:
                data = edit(data)
            destination.writestr(item.filename, data)
    return out.getvalue()


def test_corrupt_residual_rejected():
    archive = encode_ppm(make_ppm(1, 1, bytes([34, 45, 56])))
    corrupt = _mutate_archive(archive, "tiles/000_000.bin", lambda d: d[:44] + bytes([d[44] ^ 1]) + d[45:])
    with pytest.raises(ValueError, match="digest mismatch"):
        decode_archive(corrupt)


def test_source_digest_corruption_rejected():
    archive = encode_ppm(make_ppm(2, 2, bytes(12)))
    corrupt = _mutate_archive(archive, "source.ppm", lambda d: d[:-1] + bytes([d[-1] ^ 1]))
    with pytest.raises(ValueError, match="source digest mismatch"):
        decode_archive(corrupt)


def test_archive_is_deterministic():
    source = make_ppm(9, 7, bytes(range(9 * 7 * 3)))
    assert encode_ppm(source) == encode_ppm(source)


def test_reject_non_ppm_and_oversize_dimensions():
    with pytest.raises(ValueError):
        encode_ppm(b"not an image")
    with pytest.raises(ValueError):
        parse_ppm(b"P6\n7681 4320\n255\n")
