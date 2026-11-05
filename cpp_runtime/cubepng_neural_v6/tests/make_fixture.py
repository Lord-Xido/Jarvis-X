#!/usr/bin/env python3
"""Deterministic PNG fixture, pure Python stdlib: RGBA 257x259 with scanline filters."""
import binascii, struct, sys, zlib
from pathlib import Path
w, h = 257, 259
raw = bytearray()
for y in range(h):
    raw.append(0)
    for x in range(w):
        raw.extend(((x * 7 + y * 11) % 256,
                    (x ^ y) & 255,
                    (x * 19 + y * 3) & 255,
                    0 if (x * y) % 97 == 0 else 255))
def chunk(tag, data):
    return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', binascii.crc32(tag + data) & 0xffffffff)
output = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>2I5B', w, h, 8, 6, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(bytes(raw), 6)) + chunk(b'IEND', b'')
Path(sys.argv[1]).write_bytes(output)
