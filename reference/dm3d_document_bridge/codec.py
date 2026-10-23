#!/usr/bin/env python3
"""Isolated public DM3D demonstration: byte payload -> reversible 3D -> PNG in DOCX.

No private DM-PCS source, weights, credentials, or IP documents are included.
Standard library only. This is *not* a neural codec or cryptographic encryption.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import struct
import zipfile
import zlib
from pathlib import Path

MAGIC = b"D3PB0001"
ROUNDS = 3
SIDES = (16, 32)
MAX_INPUT = 32768
MEDIA = "word/media/dm3d-atlas.png"
PNG_SIG = b"\x89PNG\r\n\x1a\n"


def _morton(x: int, y: int, z: int, side: int) -> int:
    out = 0
    for bit in range(side.bit_length() - 1):
        out |= ((x >> bit) & 1) << (3 * bit)
        out |= ((y >> bit) & 1) << (3 * bit + 1)
        out |= ((z >> bit) & 1) << (3 * bit + 2)
    return out


def _order(side: int) -> list[int]:
    triples = ((x, y, z) for x in range(side) for y in range(side) for z in range(side))
    return [((x * side + y) * side + z) for x, y, z in sorted(
        triples,
        key=lambda p: (
            min(p[0], p[1], p[2], side - 1 - p[0], side - 1 - p[1], side - 1 - p[2]),
            _morton(*p, side),
        ),
    )]


def _xor_axis(values: bytearray, side: int, axis: int, reverse: bool) -> None:
    # Linear address i=x*s*s+y*s+z. Prefix XOR is inverse of delta XOR.
    stride = (side * side, side, 1)[axis]
    if reverse:
        for x in range(side):
            for y in range(side):
                for z in range(side):
                    coords = (x, y, z)
                    if coords[axis]:
                        i = (x * side + y) * side + z
                        values[i] ^= values[i - stride]
    else:
        old = bytes(values)
        for x in range(side):
            for y in range(side):
                for z in range(side):
                    coords = (x, y, z)
                    if coords[axis]:
                        i = (x * side + y) * side + z
                        values[i] = old[i] ^ old[i - stride]


def _encode_3d(plain: bytes, side: int) -> bytes:
    values = bytearray(plain.ljust(side**3, b"\0"))
    for _ in range(ROUNDS):
        for axis in (0, 1, 2):
            _xor_axis(values, side, axis, False)
    return bytes(values[i] for i in _order(side))


def _decode_3d(encoded: bytes, side: int) -> bytes:
    if len(encoded) != side**3:
        raise ValueError("Unexpected volume length")
    values = bytearray(side**3)
    for i, j in enumerate(_order(side)):
        values[j] = encoded[i]
    for _ in range(ROUNDS):
        for axis in (2, 1, 0):
            _xor_axis(values, side, axis, True)
    return bytes(values)


def _png_chunk(name: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + name + payload + struct.pack(">I", zlib.crc32(name + payload) & 0xffffffff)


def _to_png(volume: bytes, side: int) -> bytes:
    tiles = math.isqrt(side - 1) + 1
    width = height = side * tiles
    pixels = bytearray(width * height)
    for z in range(side):
        ty, tx = divmod(z, tiles)
        for x in range(side):
            for y in range(side):
                row, col = ty * side + y, tx * side + x
                pixels[row * width + col] = volume[(x * side + y) * side + z]
    rows = b"".join(b"\0" + bytes(pixels[r * width:(r + 1) * width]) for r in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return PNG_SIG + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", zlib.compress(rows, 9)) + _png_chunk(b"IEND", b"")


def _from_png(png: bytes) -> tuple[bytes, int]:
    if not png.startswith(PNG_SIG) or len(png) > 500_000:
        raise ValueError("Invalid or oversized PNG")
    cursor = len(PNG_SIG)
    header = None
    compressed = bytearray()
    ended = False
    while cursor < len(png):
        if cursor + 12 > len(png):
            raise ValueError("Truncated PNG")
        size = struct.unpack(">I", png[cursor:cursor+4])[0]
        tag = png[cursor+4:cursor+8]
        end = cursor + 12 + size
        if end > len(png):
            raise ValueError("Truncated PNG chunk")
        payload = png[cursor+8:cursor+8+size]
        crc = struct.unpack(">I", png[cursor+8+size:end])[0]
        if zlib.crc32(tag + payload) & 0xffffffff != crc:
            raise ValueError("PNG CRC mismatch")
        if tag == b"IHDR":
            if header is not None or size != 13:
                raise ValueError("Invalid PNG header")
            header = struct.unpack(">IIBBBBB", payload)
        elif tag == b"IDAT":
            compressed.extend(payload)
        elif tag == b"IEND":
            ended = True
            if size or end != len(png):
                raise ValueError("Invalid PNG end")
            break
        elif tag not in (b"tEXt", b"pHYs"):
            raise ValueError("Unexpected PNG chunk")
        cursor = end
    if not ended or header is None:
        raise ValueError("Missing PNG header or terminator")
    width, height, depth, color, comp, filt, interlace = header
    side = next((s for s in SIDES if (s * (math.isqrt(s - 1) + 1)) == width == height), None)
    if side is None or (depth, color, comp, filt, interlace) != (8, 0, 0, 0, 0):
        raise ValueError("Unsupported PNG format")
    count = height * (width + 1)
    obj = zlib.decompressobj()
    scanlines = obj.decompress(bytes(compressed), count + 1)
    if not obj.eof or obj.unused_data or obj.unconsumed_tail or len(scanlines) != count:
        raise ValueError("Invalid or oversized PNG pixels")
    pixels = bytearray(width * height)
    for r in range(height):
        start = r * (width + 1)
        if scanlines[start] != 0:
            raise ValueError("Unsupported PNG row filter")
        pixels[r * width:(r + 1) * width] = scanlines[start+1:start+1+width]
    tiles = math.isqrt(side - 1) + 1
    volume = bytearray(side**3)
    occupied = bytearray(width * height)
    for z in range(side):
        ty, tx = divmod(z, tiles)
        for x in range(side):
            for y in range(side):
                loc = (ty * side + y) * width + tx * side + x
                occupied[loc] = 1
                volume[(x * side + y) * side + z] = pixels[loc]
    if any(value for i, value in enumerate(pixels) if not occupied[i]):
        raise ValueError("Nonzero atlas padding")
    return bytes(volume), side


def _docx(png: bytes) -> bytes:
    # A minimal, standards-compatible OOXML document with an actual raster image relationship.
    types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>'''
    rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rDoc" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'''
    docrels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rImage" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/dm3d-atlas.png"/></Relationships>'''
    document = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><w:body>
<w:p><w:r><w:t>DM3D Public Reference: Reversible 3D Data Atlas</w:t></w:r></w:p>
<w:p><w:r><w:t>Payload image is an exact, finite byte atlas. It does not encode an astronomical number's expanded digits, execute code, or provide encryption.</w:t></w:r></w:p>
<w:p><w:r><w:drawing><wp:inline><wp:extent cx="1828800" cy="1828800"/><wp:docPr id="1" name="DM3D data atlas" descr="Lossless 3D byte atlas"/><a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic><pic:nvPicPr><pic:cNvPr id="1" name="dm3d-atlas.png"/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip r:embed="rImage"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="1828800" cy="1828800"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p><w:sectPr/></w:body></w:document>'''
    dst = io.BytesIO()
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for name, content in [
            ("[Content_Types].xml", types), ("_rels/.rels", rels),
            ("word/document.xml", document), ("word/_rels/document.xml.rels", docrels),
        ]:
            z.writestr(name, content.encode("utf-8"))
        z.writestr(MEDIA, png)
    return dst.getvalue()


def encode(payload: bytes) -> bytes:
    if len(payload) > MAX_INPUT:
        raise ValueError("Payload exceeds bounded public reference limit")
    packed = zlib.compress(payload, 9)
    # Compression only affects storage when smaller. Bytes preserve semantics exactly.
    mode, stored = min(((0, payload), (1, packed)), key=lambda p: len(p[1]))
    header = MAGIC + struct.pack(">IB", len(payload), mode) + hashlib.sha256(payload).digest()
    buf = header + stored
    side = next((s for s in SIDES if len(buf) <= s**3), None)
    if side is None:
        raise ValueError("Encoded payload does not fit bounded lattice")
    return _docx(_to_png(_encode_3d(buf, side), side))


def decode(docx: bytes) -> bytes:
    if len(docx) > 1_000_000:
        raise ValueError("Oversized DOCX")
    try:
        with zipfile.ZipFile(io.BytesIO(docx)) as z:
            if MEDIA not in z.namelist():
                raise ValueError("Missing embedded atlas")
            member = z.getinfo(MEDIA)
            if member.file_size > 500_000:
                raise ValueError("Oversized embedded atlas")
            png = z.read(MEDIA)
    except (zipfile.BadZipFile, KeyError) as exc:
        raise ValueError("Invalid DOCX package") from exc
    encoded, side = _from_png(png)
    volume = _decode_3d(encoded, side)
    if volume[:8] != MAGIC:
        raise ValueError("Not a DM3D payload")
    size, mode = struct.unpack(">IB", volume[8:13])
    digest = volume[13:45]
    if size > MAX_INPUT or mode not in (0, 1):
        raise ValueError("Invalid source manifest")
    stored = volume[45:]
    if mode == 1:
        obj = zlib.decompressobj()
        plain = obj.decompress(stored, size + 1)
        if not obj.eof or obj.unused_data.strip(b"\0") or obj.unconsumed_tail:
            raise ValueError("Invalid compressed source")
    else:
        plain = stored[:size]
        if any(stored[size:]):
            raise ValueError("Nonzero volume padding")
    if len(plain) != size or hashlib.sha256(plain).digest() != digest:
        raise ValueError("Payload integrity check failed")
    return plain


def selftest() -> dict:
    source = Path(__file__).read_bytes()
    samples = {"empty": b"", "symbolic_tower": b"S=10^24;N=S^(S^S)",
               "binary": bytes(range(256))*12, "source": source}
    result = {}
    for key, value in samples.items():
        artifact = encode(value)
        restored = decode(artifact)
        assert restored == value, key
        result[key] = {"source_bytes": len(value), "docx_bytes": len(artifact),
                       "sha256": hashlib.sha256(value).hexdigest(), "exact": True}
    artifact = bytearray(encode(samples["binary"]))
    with zipfile.ZipFile(io.BytesIO(artifact)) as z:
        files = {n:z.read(n) for n in z.namelist()}
    modified = bytearray(files[MEDIA]); modified[-12] ^= 1
    files[MEDIA] = bytes(modified)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():z.writestr(name,data)
    try:
        decode(stream.getvalue())
    except ValueError:
        result["pixel_tamper_rejected"] = True
    else:
        raise AssertionError("Undetected PNG tampering")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("encode", "decode", "selftest"))
    ap.add_argument("input", nargs="?", type=Path)
    ap.add_argument("output", nargs="?", type=Path)
    args = ap.parse_args()
    if args.action == "selftest":
        print(json.dumps(selftest(), indent=2))
        return
    if not args.input or not args.output:
        ap.error("encode/decode require input and output paths")
    data = args.input.read_bytes()
    args.output.write_bytes(encode(data) if args.action == "encode" else decode(data))
    print(f"{args.action}: {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
