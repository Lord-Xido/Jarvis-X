"""Dr Moagi Image ISA v2: bounded, exact 3D bitwise image archive.

This reference is intentionally independent of model weights and CUDA. A
32-byte *non-neural* occupancy latent + XOR residual demonstrates the codec
contract; it is NOT a pretrained Conv3D ANN. Only bounded active tiles exist.

Usage: python -m jarvisx.image_isa3d encode image.ppm archive.dmia2
       python -m jarvisx.image_isa3d decode archive.dmia2 recovered.ppm
       python -m jarvisx.image_isa3d inspect archive.dmia2
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import struct
import zipfile
from pathlib import Path

TILE = 128
CUBE = 16
CUBE_BITS = CUBE ** 3
CUBE_BYTES = CUBE_BITS // 8
RGB_TILE_BYTES = TILE * TILE * 3
CUBES_PER_TILE = RGB_TILE_BYTES // CUBE_BYTES
MAGIC = "dr-moagi-image-isa3d"
VERSION = 2
MAX_SOURCE_BYTES = 128 * 1024 * 1024
MAX_PIXELS = 7680 * 4320
MAX_MEMBER_BYTES = 12 + CUBES_PER_TILE * (512 + 32) + 32
LATENT_VALUES = 32
FRAME = struct.Struct(">4sHHHH")  # magic, tile_x, tile_y, valid width, valid height


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def image_address(x: int, y: int, slot: int, nx: int = 60, ny: int = 34,
                  slots: int = 1000) -> int:
    """Unique canonical 8K tile address in the 3D (column,row,image) lattice."""
    if not all(isinstance(v, int) and not isinstance(v, bool)
               for v in (x, y, slot, nx, ny, slots)):
        raise ValueError("coordinates and dimensions must be integers")
    if nx < 1 or ny < 1 or slots < 1 or not (0 <= x < nx and 0 <= y < ny and 0 <= slot < slots):
        raise ValueError("invalid 3D tile address")
    return ((slot * ny) + y) * nx + x


def _bits(raw: bytes) -> bytearray:
    if len(raw) != CUBE_BYTES:
        raise ValueError("expected 512 input bytes")
    return bytearray((raw[i >> 3] >> (i & 7)) & 1 for i in range(CUBE_BITS))


def _bytes(bits: bytearray) -> bytes:
    if len(bits) != CUBE_BITS:
        raise ValueError("expected 4096 bits")
    out = bytearray(CUBE_BYTES)
    for i, bit in enumerate(bits):
        out[i >> 3] |= (bit & 1) << (i & 7)
    return bytes(out)


def xor_shear(raw: bytes) -> bytes:
    """Reversible finite-boundary GF(2) delta along X, Y, then Z."""
    bits = _bits(raw)
    for stride in (256, 16, 1):
        old = bits[:]
        for i in range(CUBE_BITS):
            if (i // stride) % CUBE:
                bits[i] = old[i] ^ old[i - stride]
    return _bytes(bits)


def inverse_xor_shear(conditioned: bytes) -> bytes:
    bits = _bits(conditioned)
    for stride in (1, 16, 256):
        for i in range(CUBE_BITS):
            if (i // stride) % CUBE:
                bits[i] ^= bits[i - stride]
    return _bytes(bits)


def occupancy_encode(conditioned: bytes) -> tuple[bytes, bytes]:
    """Reference 32-byte occupancy latent, plus exact XOR residual.

    The latent is NOT an ANN. Each int8 is the number of 1-bits in one
    consecutive 128-bit interval minus 64. Prediction is group majority.
    """
    bits = _bits(conditioned)
    codes = bytearray(LATENT_VALUES)
    predicted = bytearray(CUBE_BITS)
    for block in range(LATENT_VALUES):
        start = block * (CUBE_BITS // LATENT_VALUES)
        end = start + (CUBE_BITS // LATENT_VALUES)
        count = sum(bits[start:end])
        codes[block] = count - 64 + 128  # stored unsigned, interpreted as signed offset
        predicted[start:end] = bytes([int(count >= 64)]) * (end - start)
    residual = bytearray(a ^ b for a, b in zip(bits, predicted))
    return bytes(codes), _bytes(residual)


def occupancy_decode(codes: bytes, residual: bytes) -> bytes:
    if len(codes) != LATENT_VALUES or len(residual) != CUBE_BYTES:
        raise ValueError("invalid latent or residual length")
    r = _bits(residual)
    out = bytearray(CUBE_BITS)
    width = CUBE_BITS // LATENT_VALUES
    for block, code in enumerate(codes):
        if code < 64 or code > 192:  # count must be in [0,128]; code = count + 64
            raise ValueError("invalid latent occupancy")
        p = int(code >= 128)  # count >= 64
        for i in range(block * width, (block + 1) * width):
            out[i] = p ^ r[i]
    return _bytes(out)


def encode_cube(raw: bytes) -> bytes:
    conditioned = xor_shear(raw)
    latent, residual = occupancy_encode(conditioned)
    return latent + residual


def decode_cube(frame: bytes) -> bytes:
    if len(frame) != LATENT_VALUES + CUBE_BYTES:
        raise ValueError("incorrect cube frame size")
    return inverse_xor_shear(occupancy_decode(frame[:LATENT_VALUES], frame[LATENT_VALUES:]))


def parse_ppm(source: bytes) -> tuple[int, int, bytes]:
    """Read bounded binary P6 RGB8 PPM; comments and arbitrary header spaces accepted."""
    if len(source) > MAX_SOURCE_BYTES:
        raise ValueError("source exceeds size bound")
    mv = memoryview(source)
    offset = 0

    def token() -> bytes:
        nonlocal offset
        while offset < len(mv):
            if mv[offset] == 35:  # # comment
                while offset < len(mv) and mv[offset] not in (10, 13):
                    offset += 1
            elif mv[offset] in (9, 10, 13, 32):
                offset += 1
            else:
                break
        begin = offset
        while offset < len(mv) and mv[offset] not in (9, 10, 13, 32, 35):
            offset += 1
        if begin == offset or offset - begin > 32:
            raise ValueError("invalid PPM token")
        return bytes(mv[begin:offset])

    if token() != b"P6":
        raise ValueError("only RGB8 P6 PPM is supported")
    try:
        width, height, maxval = int(token()), int(token()), int(token())
    except (ValueError, OverflowError) as exc:
        raise ValueError("malformed PPM dimensions") from exc
    if width < 1 or height < 1 or width > 7680 or height > 4320 or width * height > MAX_PIXELS or maxval != 255:
        raise ValueError("unsupported PPM geometry or maxval")
    if offset >= len(mv) or mv[offset] not in (9, 10, 13, 32):
        raise ValueError("PPM missing header delimiter")
    # Exactly one header delimiter precedes binary pixel bytes (may be CRLF).
    offset += 1
    if offset < len(mv) and mv[offset - 1] == 13 and mv[offset] == 10:
        offset += 1
    rgb = bytes(mv[offset:])
    if len(rgb) != width * height * 3:
        raise ValueError("PPM data length mismatch")
    return width, height, rgb


def make_ppm(width: int, height: int, rgb: bytes) -> bytes:
    return f"P6\n{width} {height}\n255\n".encode("ascii") + rgb


def tile_raw(rgb: bytes, width: int, height: int, tx: int, ty: int) -> tuple[bytes, int, int]:
    actual_w = min(TILE, width - tx * TILE)
    actual_h = min(TILE, height - ty * TILE)
    buf = bytearray(RGB_TILE_BYTES)
    for y in range(actual_h):
        start = ((ty * TILE + y) * width + tx * TILE) * 3
        buf[y * TILE * 3:y * TILE * 3 + actual_w * 3] = rgb[start:start + actual_w * 3]
    return bytes(buf), actual_w, actual_h


def _safe_zip_names(z: zipfile.ZipFile, expected: set[str]) -> None:
    names = z.namelist()
    if len(names) != len(set(names)) or set(names) != expected:
        raise ValueError("missing, extra, or duplicate archive members")
    for info in z.infolist():
        bound = MAX_SOURCE_BYTES if info.filename == "source.ppm" else (65536 if info.filename == "manifest.json" else MAX_MEMBER_BYTES)
        if info.file_size > bound or info.compress_size > MAX_SOURCE_BYTES or info.flag_bits & 1:
            raise ValueError("oversized or encrypted archive member")


def encode_ppm(source: bytes, slot: int = 42) -> bytes:
    width, height, rgb = parse_ppm(source)
    if not isinstance(slot, int) or isinstance(slot, bool) or not 0 <= slot < 1000:
        raise ValueError("image slot out of range")
    nx, ny = math.ceil(width / TILE), math.ceil(height / TILE)
    metadata = {
        "format": MAGIC, "version": VERSION, "image_slot": slot,
        "width": width, "height": height, "tile_side": TILE,
        "tile_grid": [nx, ny], "cube_side": CUBE,
        "cubes_per_tile": CUBES_PER_TILE, "latent_kind": "non-neural-occupancy-v1",
        "latent_values_per_cube": LATENT_VALUES, "source_sha256": sha256(source),
        "pixels_sha256": sha256(rgb), "source_size": len(source),
        "source_type": "P6 RGB8", "exact": True,
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("source.ppm", source)
        for ty in range(ny):
            for tx in range(nx):
                raw, ww, hh = tile_raw(rgb, width, height, tx, ty)
                encoded = b"".join(encode_cube(raw[i:i + CUBE_BYTES]) for i in range(0, len(raw), CUBE_BYTES))
                header = FRAME.pack(b"DMT2", tx, ty, ww, hh)
                z.writestr(f"tiles/{ty:03d}_{tx:03d}.bin", header + encoded + hashlib.sha256(raw).digest())
        z.writestr("manifest.json", json.dumps(metadata, sort_keys=True, separators=(",", ":")))
    return buf.getvalue()


def decode_archive(archive: bytes) -> tuple[bytes, dict]:
    """Verify every source tile and reconstruct original source bytes exactly."""
    if len(archive) > 2 * MAX_SOURCE_BYTES:
        raise ValueError("archive exceeds bound")
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        if z.getinfo("manifest.json").file_size > 65536:
            raise ValueError("oversized manifest")
        meta = json.loads(z.read("manifest.json"))
        if meta.get("format") != MAGIC or meta.get("version") != VERSION:
            raise ValueError("unsupported archive format")
        width, height, slot = meta["width"], meta["height"], meta["image_slot"]
        if not all(type(v) is int for v in (width, height, slot)):
            raise ValueError("invalid geometry")
        if not (0 < width <= 7680 and 0 < height <= 4320 and width * height <= MAX_PIXELS and 0 <= slot < 1000):
            raise ValueError("image geometry outside bounds")
        nx, ny = math.ceil(width / TILE), math.ceil(height / TILE)
        if meta.get("tile_grid") != [nx, ny] or meta.get("tile_side") != TILE or meta.get("cubes_per_tile") != CUBES_PER_TILE:
            raise ValueError("invalid tile geometry")
        expected = {"manifest.json", "source.ppm"} | {
            f"tiles/{ty:03d}_{tx:03d}.bin" for ty in range(ny) for tx in range(nx)
        }
        _safe_zip_names(z, expected)
        source = z.read("source.ppm")
        if len(source) != meta.get("source_size") or sha256(source) != meta["source_sha256"]:
            raise ValueError("source digest mismatch")
        sw, sh, raw_pixels = parse_ppm(source)
        if (sw, sh) != (width, height):
            raise ValueError("source dimensions mismatch")
        restored = bytearray(width * height * 3)
        for ty in range(ny):
            for tx in range(nx):
                frame = z.read(f"tiles/{ty:03d}_{tx:03d}.bin")
                if len(frame) != FRAME.size + CUBES_PER_TILE * (LATENT_VALUES + CUBE_BYTES) + 32:
                    raise ValueError("tile frame length mismatch")
                magic, fx, fy, ww, hh = FRAME.unpack_from(frame)
                if (magic, fx, fy, ww, hh) != (b"DMT2", tx, ty, min(TILE, width - tx * TILE), min(TILE, height - ty * TILE)):
                    raise ValueError("tile address mismatch")
                parts = frame[FRAME.size:-32]
                frame_size = LATENT_VALUES + CUBE_BYTES
                tile = b"".join(decode_cube(parts[i:i + frame_size]) for i in range(0, len(parts), frame_size))
                if hashlib.sha256(tile).digest() != frame[-32:]:
                    raise ValueError("tile digest mismatch")
                for y in range(hh):
                    start = ((ty * TILE + y) * width + tx * TILE) * 3
                    restored[start:start + ww * 3] = tile[y * TILE * 3:y * TILE * 3 + ww * 3]
        if sha256(restored) != meta["pixels_sha256"] or restored != raw_pixels:
            raise ValueError("pixel digest or source equality mismatch")
        return source, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    enc = sub.add_parser("encode")
    enc.add_argument("source", type=Path)
    enc.add_argument("archive", type=Path)
    enc.add_argument("--slot", type=int, default=42)
    dec = sub.add_parser("decode")
    dec.add_argument("archive", type=Path)
    dec.add_argument("restored", type=Path)
    ins = sub.add_parser("inspect")
    ins.add_argument("archive", type=Path)
    args = parser.parse_args()
    if args.action == "encode":
        args.archive.write_bytes(encode_ppm(args.source.read_bytes(), slot=args.slot))
        print(f"encoded: {args.archive}")
    elif args.action == "decode":
        source, meta = decode_archive(args.archive.read_bytes())
        args.restored.write_bytes(source)
        print(f"verified: {sha256(source)}  slot={meta['image_slot']}")
    else:
        _, meta = decode_archive(args.archive.read_bytes())
        print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
