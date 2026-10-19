"""Virtual 80K x 80K image ROM: a bounded 3D codec embedded in a valid PNG.

The 80,000 x 80,000 address space is logical; only one active 3D tile is
materialized. PNG includes a non-executable, versioned instruction manifest,
3D latent bytes, and optional XOR recovery bytes. No external image library.

Run: python -m jarvisx.virtual_80k_image_rom build --out rom.png
     python -m jarvisx.virtual_80k_image_rom verify --input rom.png
     python -m jarvisx.virtual_80k_image_rom extract --input rom.png --out tile.rgba
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import struct
import zlib
from pathlib import Path

WIDTH = HEIGHT = 80_000
VIRTUAL_PIXELS = WIDTH * HEIGHT
DEPTH = 4
CHANNELS = 4
BLOCK = 2
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
CHUNK = b"vpRm"  # valid private, ancillary, safe-to-copy PNG chunk
INSTRUCTIONS = ("FETCH_TILE", "ENCODE_3D", "COUPLE_6N", "FOLD_IN",
                "DECODE_3D", "COMPARE", "VERIFY", "COMMIT")
MAX_PAYLOAD = 2_000_000


def _check(ok: bool, msg: str) -> None:
    if not ok:
        raise ValueError(msg)


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate(tile: int, tx: int, ty: int) -> None:
    _check(isinstance(tile, int) and 4 <= tile <= 128 and tile % 2 == 0
           and WIDTH % tile == 0 and HEIGHT % tile == 0, "invalid tile size")
    _check(isinstance(tx, int) and isinstance(ty, int)
           and 0 <= tx < WIDTH // tile and 0 <= ty < HEIGHT // tile,
           "tile coordinates out of virtual bounds")


def voxel(x: int, y: int, z: int, channel: int, seed: int = 7) -> int:
    """Deterministic addressable 3D feature byte; not a trained media model."""
    _check(0 <= x < WIDTH and 0 <= y < HEIGHT and 0 <= z < DEPTH
           and 0 <= channel < CHANNELS, "invalid virtual 3D address")
    address = f"DM-ROM1:{seed}:{x}:{y}:{z}:{channel}".encode("ascii")
    return hashlib.blake2s(address, digest_size=1).digest()[0]


def fetch_tile(tile: int = 8, tx: int = 0, ty: int = 0, seed: int = 7) -> bytes:
    _validate(tile, tx, ty)
    return bytes(voxel(tx * tile + x, ty * tile + y, z, c, seed)
                 for z in range(DEPTH) for y in range(tile)
                 for x in range(tile) for c in range(CHANNELS))


def _index(x: int, y: int, z: int, c: int, n: int) -> int:
    return (((z * n + y) * n + x) * CHANNELS + c)


def encode_3d(raw: bytes, n: int) -> bytes:
    """2x2x2 voxel blocks -> channel mean, giving an 8:1 lossy latent."""
    _check(len(raw) == n*n*DEPTH*CHANNELS and n % 2 == 0, "bad tile bytes")
    result = bytearray()
    for z in range(0, DEPTH, BLOCK):
        for y in range(0, n, BLOCK):
            for x in range(0, n, BLOCK):
                for c in range(CHANNELS):
                    v = sum(raw[_index(x+dx, y+dy, z+dz, c, n)]
                            for dz in (0, 1) for dy in (0, 1) for dx in (0, 1))
                    result.append((v+4)//8)
    return bytes(result)


def decode_3d(latent: bytes, n: int) -> bytes:
    expected = (n//2)*(n//2)*(DEPTH//2)*CHANNELS
    _check(n > 0 and n % 2 == 0 and len(latent) == expected, "bad latent")
    raw = bytearray(n*n*DEPTH*CHANNELS)
    k = 0
    for z in range(0, DEPTH, BLOCK):
        for y in range(0, n, BLOCK):
            for x in range(0, n, BLOCK):
                for c in range(CHANNELS):
                    v = latent[k]
                    k += 1
                    for dz in (0, 1):
                        for dy in (0, 1):
                            for dx in (0, 1):
                                raw[_index(x+dx, y+dy, z+dz, c, n)] = v
    return bytes(raw)


def fold_3d(latent: bytes, n: int, numerator: int = 1,
            denominator: int = 4) -> bytes:
    """Finite 6-neighbor periodic latent diffusion, one bounded step."""
    _check(0 <= numerator <= denominator and denominator > 0, "invalid gain")
    side = n//2
    layers = DEPTH//2
    def idx(x, y, z, c):
        return (((z % layers)*side + (y % side))*side + (x % side))*CHANNELS+c
    out = bytearray(len(latent))
    for z in range(layers):
        for y in range(side):
            for x in range(side):
                for c in range(CHANNELS):
                    k = idx(x, y, z, c)
                    neighbors = (latent[idx(x+dx, y+dy, z+dz, c)]
                                 for dx,dy,dz in ((1,0,0),(-1,0,0),(0,1,0),
                                                   (0,-1,0),(0,0,1),(0,0,-1)))
                    total = 6*(denominator-numerator)*latent[k] + numerator*sum(neighbors)
                    out[k] = min(255, max(0, (total+3*denominator)//(6*denominator)))
    return bytes(out)


def _png_chunk(name: bytes, payload: bytes) -> bytes:
    return (struct.pack(">I", len(payload)) + name + payload +
            struct.pack(">I", zlib.crc32(name+payload)&0xffffffff))


def _preview(tx: int, ty: int, tile: int, seed: int, preview: int = 48) -> bytes:
    rows = bytearray()
    for y in range(preview):
        rows.append(0)
        for x in range(preview):
            px = tx*tile + min(tile-1, x*tile//preview)
            py = ty*tile + min(tile-1, y*tile//preview)
            rows.extend(voxel(px, py, 0, c, seed) for c in range(CHANNELS))
    ihdr = struct.pack(">IIBBBBB", preview, preview, 8, 6, 0, 0, 0)
    return _png_chunk(b"IHDR", ihdr), _png_chunk(b"IDAT", zlib.compress(bytes(rows), 9))


def build_png(*, tile: int = 8, tx: int = 0, ty: int = 0,
              seed: int = 7, lossless: bool = True) -> bytes:
    """Build a valid, viewable PNG with a bounded 3D executable *data* recipe."""
    source = fetch_tile(tile, tx, ty, seed)
    latent = fold_3d(encode_3d(source, tile), tile)
    decoded = decode_3d(latent, tile)
    residual = bytes(a ^ b for a, b in zip(source, decoded)) if lossless else b""
    mse = sum((a-b)**2 for a,b in zip(source, decoded))/len(source)
    doc = {
        "format": "dr-moagi-virtual-80k-image-rom", "version": 1,
        "virtual_width": WIDTH, "virtual_height": HEIGHT,
        "virtual_pixels": VIRTUAL_PIXELS,
        "tile": tile, "tile_x": tx, "tile_y": ty, "depth": DEPTH,
        "channels": CHANNELS, "seed": seed, "lossless": lossless,
        "instructions": list(INSTRUCTIONS),
        "latent_b64": base64.b64encode(latent).decode("ascii"),
        "residual_b64": base64.b64encode(zlib.compress(residual)).decode("ascii") if lossless else "",
        "source_sha256": _hash(source) if lossless else None,
        "lossy_mse": mse,
        "note": "Finite 3D numeric proxy; no trained text/audio/video generator. Opening PNG never executes instructions."
    }
    packed = zlib.compress(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode(), 9)
    _check(len(packed) <= MAX_PAYLOAD, "ROM payload too large")
    ihdr, image_data = _preview(tx, ty, tile, seed)
    return PNG_MAGIC + ihdr + _png_chunk(CHUNK, packed) + image_data + _png_chunk(b"IEND", b"")


def _bounded_inflate(compressed: bytes, maximum: int) -> bytes:
    _check(len(compressed) <= MAX_PAYLOAD, "oversized payload")
    obj = zlib.decompressobj()
    out = obj.decompress(compressed, maximum+1)
    _check(len(out) <= maximum and obj.eof and not obj.unused_data
           and not obj.unconsumed_tail, "invalid or oversized compressed data")
    return out


def read_png(data: bytes) -> dict:
    """Validate PNG framing and ROM metadata without executing embedded content."""
    _check(data.startswith(PNG_MAGIC) and len(data) <= 4_000_000, "invalid PNG")
    off = len(PNG_MAGIC)
    record = None
    seen_end = False
    seen_idat = False
    while off < len(data):
        _check(off+12 <= len(data), "truncated PNG chunk")
        size = struct.unpack_from(">I", data, off)[0]
        name = data[off+4:off+8]
        _check(size <= MAX_PAYLOAD and off+12+size <= len(data), "oversized PNG chunk")
        payload = data[off+8:off+8+size]
        crc = struct.unpack_from(">I", data, off+8+size)[0]
        _check(zlib.crc32(name+payload)&0xffffffff == crc, "PNG CRC mismatch")
        off += 12+size
        if name == CHUNK:
            _check(record is None, "duplicate ROM chunk")
            record = json.loads(_bounded_inflate(payload, MAX_PAYLOAD))
        elif name == b"IDAT":
            seen_idat = True
        elif name == b"IEND":
            seen_end = True
            break
    _check(seen_end and seen_idat and off == len(data) and isinstance(record, dict), "missing PNG data or ROM")
    _check(record.get("format") == "dr-moagi-virtual-80k-image-rom"
           and type(record.get("version")) is int and record["version"] == 1,
           "bad ROM version")
    _check(record.get("virtual_width") == WIDTH
           and record.get("virtual_height") == HEIGHT
           and record.get("virtual_pixels") == VIRTUAL_PIXELS
           and record.get("depth") == DEPTH and record.get("channels") == CHANNELS,
           "bad virtual geometry")
    _check(record.get("instructions") == list(INSTRUCTIONS), "unrecognized program")
    _validate(record["tile"], record["tile_x"], record["tile_y"])
    _check(type(record.get("seed")) is int and -2**63 <= record["seed"] < 2**63,
           "bad seed")
    _check(type(record.get("lossless")) is bool, "bad codec mode")
    return record


def decode_rom(data: bytes) -> tuple[bytes, dict]:
    doc = read_png(data)
    n = doc["tile"]
    latent = base64.b64decode(doc["latent_b64"], validate=True)
    base = decode_3d(latent, n)
    if doc["lossless"]:
        packed = base64.b64decode(doc["residual_b64"], validate=True)
        residual = _bounded_inflate(packed, len(base))
        _check(len(residual) == len(base), "bad residual length")
        raw = bytes(a^b for a,b in zip(base,residual))
        _check(_hash(raw) == doc["source_sha256"], "ROM source hash mismatch")
    else:
        raw = base
    return raw, doc


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--out", required=True)
    build.add_argument("--tile", type=int, default=8)
    build.add_argument("--tile-x", type=int, default=0)
    build.add_argument("--tile-y", type=int, default=0)
    build.add_argument("--seed", type=int, default=7)
    build.add_argument("--lossy", action="store_true")
    for name in ("inspect", "verify", "extract"):
        p = sub.add_parser(name)
        p.add_argument("--input", required=True)
        if name == "extract":
            p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.command == "build":
        data = build_png(tile=a.tile, tx=a.tile_x, ty=a.tile_y,
                         seed=a.seed, lossless=not a.lossy)
        Path(a.out).write_bytes(data)
        result = {"path": a.out, "png_bytes": len(data), "preview": "48x48",
                  "virtual_pixels": VIRTUAL_PIXELS}
    else:
        data = Path(a.input).read_bytes()
        if a.command == "inspect":
            doc = read_png(data)
            result = {k:v for k,v in doc.items() if not k.endswith("_b64")}
        else:
            raw, doc = decode_rom(data)
            if a.command == "extract":
                Path(a.out).write_bytes(raw)
            result = {"verified": bool(doc["lossless"]),
                      "decoded_bytes": len(raw), "sha256": _hash(raw),
                      "tile": doc["tile"], "depth": DEPTH,
                      "lossless": doc["lossless"]}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
