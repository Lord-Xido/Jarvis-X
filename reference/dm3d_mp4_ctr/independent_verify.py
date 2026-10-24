#!/usr/bin/env python3
"""Independent, standard-library JXMP4T1 decoder and SHA-256 receipt.

This deliberately does not import or call the C++ codec. It verifies a wire
artifact constructed by a different implementation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import struct
import zlib
from pathlib import Path

HEADER = struct.Struct('<8sHHQII')
RECORD = struct.Struct('<IIHHB3s')
MAGIC = b'JXMP4T1\0'
TILE = 1024
AXIS = 10 * 1024 * 1024


def decode_archive(path: Path, expected: Path | None = None) -> dict:
    with path.open('rb') as file:
        header = file.read(HEADER.size)
        if len(header) != HEADER.size:
            raise ValueError('truncated archive header')
        magic, version, flags, byte_count, crc, tiles = HEADER.unpack(header)
        if magic != MAGIC or version != 1 or flags != 0:
            raise ValueError('unsupported archive')
        if tiles != ((byte_count - 1) // TILE + 1 if byte_count else 0):
            raise ValueError('invalid declared tile count')
        sha = hashlib.sha256()
        checksum = 0
        total = 0
        source_file = expected.open('rb') if expected is not None else None
        try:
            for index in range(tiles):
                record = file.read(RECORD.size)
                if len(record) != RECORD.size:
                    raise ValueError('truncated record')
                idx, local_crc, raw_len, enc_len, mode, reserved = RECORD.unpack(record)
                expected_len = min(TILE, byte_count - total)
                if idx != index or raw_len != expected_len or not (1 <= enc_len <= 2048) or reserved != b'\0'*3:
                    raise ValueError('record layout invalid')
                payload = file.read(enc_len)
                if len(payload) != enc_len:
                    raise ValueError('truncated encoded tile')
                if mode in (0, 2):
                    if enc_len != raw_len:
                        raise ValueError('raw/predictive length mismatch')
                    decoded = payload
                elif mode in (1, 3):
                    if len(payload) % 2:
                        raise ValueError('invalid RLE stream')
                    buf = bytearray()
                    for count, value in zip(payload[::2], payload[1::2]):
                        if count == 0 or count > raw_len - len(buf):
                            raise ValueError('invalid RLE count')
                        buf.extend(bytes([value]) * count)
                    if len(buf) != raw_len:
                        raise ValueError('RLE output length mismatch')
                    decoded = bytes(buf)
                else:
                    raise ValueError('unknown codec mode')
                if mode in (2, 3):
                    restored = bytearray()
                    last = 0
                    for b in decoded:
                        last ^= b
                        restored.append(last)
                    decoded = bytes(restored)
                if zlib.crc32(decoded) != local_crc:
                    raise ValueError('tile CRC mismatch')
                if source_file is not None and source_file.read(raw_len) != decoded:
                    raise ValueError('source mismatch')
                sha.update(decoded)
                checksum = zlib.crc32(decoded, checksum)
                total += len(decoded)
            if checksum != crc or total != byte_count or file.read(1):
                raise ValueError('global CRC, size or end-of-file mismatch')
            if source_file is not None and source_file.read(1):
                raise ValueError('expected source longer than archive')
        finally:
            if source_file is not None:
                source_file.close()
    return {
        'verified': True, 'format': 'JXMP4T1', 'version': 1,
        'logical_axes': [AXIS, AXIS, AXIS], 'bytes': byte_count,
        'tiles': tiles, 'sha256': sha.hexdigest(),
        'crc32': f'{checksum:08x}',
        'expected_sha256_matches': expected is not None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    parser.add_argument('--expected', type=Path)
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    result = decode_archive(args.archive, args.expected)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.receipt:
        args.receipt.write_text(text + '\n', encoding='utf-8')
    print(text)

if __name__ == '__main__':
    main()
