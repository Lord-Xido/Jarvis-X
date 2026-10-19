"""Verification of the sparse, PNG-containerized virtual 80K pixel ROM."""
import struct

import pytest
from jarvisx.virtual_80k_image_rom import (
    WIDTH, VIRTUAL_PIXELS, PNG_MAGIC, build_png, decode_rom, encode_3d,
    decode_3d, fetch_tile, fold_3d, read_png, voxel,
)


def test_80k_virtual_extent_does_not_allocate_dense_image():
    assert VIRTUAL_PIXELS == 6_400_000_000
    png = build_png(tile=8, tx=9999, ty=9999)
    assert png.startswith(PNG_MAGIC)
    assert len(png) < 50_000
    data, record = decode_rom(png)
    assert data == fetch_tile(tile=8, tx=9999, ty=9999)
    assert record["virtual_width"] == WIDTH


def test_three_dimensional_pooling_and_inverse_shape():
    raw = fetch_tile(tile=8, tx=1, ty=2)
    latent = encode_3d(raw, 8)
    assert len(latent) == len(raw)//8
    changed = fold_3d(latent, 8)
    assert len(changed) == len(latent)
    assert len(decode_3d(changed, 8)) == len(raw)


@pytest.mark.parametrize("tile", [4, 8, 16])
def test_lossless_png_round_trip(tile):
    png = build_png(tile=tile, tx=1, ty=2, seed=11)
    raw, record = decode_rom(png)
    assert raw == fetch_tile(tile=tile, tx=1, ty=2, seed=11)
    assert record["lossless"] is True


def test_lossy_mode_does_not_claim_exactness():
    raw, record = decode_rom(build_png(lossless=False))
    assert len(raw) == 8*8*4*4
    assert record["lossless"] is False


def test_invalid_address_and_duplicate_chunk():
    with pytest.raises(ValueError, match="invalid virtual"):
        voxel(80_000, 0, 0, 0)
    with pytest.raises(ValueError, match="bounds"):
        build_png(tile=8, tx=10_000)
    png = build_png()
    at = png.index(b"vpRm") - 4
    size = struct.unpack_from(">I", png, at)[0]
    dup = png[at:at+12+size]
    with pytest.raises(ValueError, match="duplicate"):
        read_png(png[:at] + dup + png[at:])


def test_corruption_rejected():
    image = bytearray(build_png())
    pos = image.index(b"vpRm") + 4
    image[pos] ^= 1
    with pytest.raises(ValueError, match="CRC"):
        decode_rom(bytes(image))


def test_png_preview_is_standard_rgba():
    png = build_png()
    assert png.startswith(PNG_MAGIC)
    ihdr = png[16:29]
    assert struct.unpack(">IIBBBBB", ihdr)[:4] == (48,48,8,6)


def test_deterministic_serialization():
    assert build_png(seed=17,tx=4,ty=8) == build_png(seed=17,tx=4,ty=8)
    assert build_png(seed=17,tx=4,ty=8) != build_png(seed=18,tx=4,ty=8)
