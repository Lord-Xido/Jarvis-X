#!/usr/bin/env python3
"""JX-PROM-100 v1: embed a REAL 1,024,000-byte control frame into 100 RGB pixel banks.
The carrier PNG is lossless. All offsets are absolute and deterministic.
"""
import argparse
import hashlib
import json
import struct
import zlib
from pathlib import Path
from PIL import Image, ImageDraw

BANKS = 100
PAYLOAD = 10240
PATCH_WIDTH, PATCH_HEIGHT = 128, 32
HEADER = 128
RAW_BANK = PATCH_WIDTH * PATCH_HEIGHT * 3
FRAME_SIZE = BANKS * PAYLOAD
CANVAS_SIZE = (7856, 4496)
TILE = (768, 432)
GAP = 16
MAGIC = b'JXPRB001'
VERSION = 1


def patch_xy(idx):
    row, col = divmod(idx, 10)
    return GAP + col * (TILE[0] + GAP) + TILE[0] - PATCH_WIDTH - 8, \
           GAP + row * (TILE[1] + GAP) + TILE[1] - PATCH_HEIGHT - 8


def bank_bytes(idx: int, frame: bytes):
    chunk = frame[idx * PAYLOAD:(idx + 1) * PAYLOAD]
    assert len(chunk) == PAYLOAD
    buf = bytearray(RAW_BANK)
    buf[:8] = MAGIC
    struct.pack_into('<8I', buf, 8,
                     idx, BANKS, idx * PAYLOAD, PAYLOAD, FRAME_SIZE,
                     VERSION, zlib.crc32(chunk), 0)
    buf[40:72] = hashlib.sha256(frame).digest()
    buf[72:104] = hashlib.sha256(chunk).digest()
    # header CRC zeroes 104..107 during calculation
    struct.pack_into('<I', buf, 104, zlib.crc32(buf[:HEADER]))
    buf[HEADER:HEADER + PAYLOAD] = chunk
    return bytes(buf)


def encode(source: Path, framepath: Path, output: Path, manifest: Path):
    frame = framepath.read_bytes()
    if len(frame) != FRAME_SIZE:
        raise ValueError('control frame must be 1,024,000 bytes')
    if frame[:8] != b'JXSF1T12':
        raise ValueError('wrong original superframe format')
    with Image.open(source) as src:
        if src.size != CANVAS_SIZE:
            raise ValueError(f'expected original 10x10 layout {CANVAS_SIZE}, got {src.size}')
        canvas = src.convert('RGB')
    pix = canvas.load()
    for bank in range(BANKS):
        x0, y0 = patch_xy(bank)
        data = bank_bytes(bank, frame)
        for i in range(PATCH_WIDTH * PATCH_HEIGHT):
            y, x = divmod(i, PATCH_WIDTH)
            p = i*3
            pix[x0+x,y0+y] = (data[p],data[p+1],data[p+2])
    output.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(output, format='PNG', optimize=False, compress_level=5)
    doc = {
        'format':'JX-PROM-100', 'version':VERSION,
        'png':output.name, 'dimensions':list(CANVAS_SIZE),
        'frame_bytes':FRAME_SIZE,'frame_sha256':hashlib.sha256(frame).hexdigest(),
        'source_frame':framepath.name,
        'banks':BANKS,'payload_bytes_per_bank':PAYLOAD,
        'patch_pixels':[PATCH_WIDTH,PATCH_HEIGHT],
        'patch_raw_rgb_bytes':RAW_BANK,'header_bytes_per_bank':HEADER,
        'unused_zero_padding_per_bank':RAW_BANK-HEADER-PAYLOAD,
        'format_note':'Lossless pixel data region; JPEG/PDF re-encoding not guaranteed to preserve bytes.',
        'bank_xy_formula':'(16 + (i % 10)*784 + 632, 16 + (i // 10)*448 + 392)',
        'embedded_vm_program':'JXEX at control offset 4096; JXOP reserved at offset 5120'
    }
    manifest.write_text(json.dumps(doc,indent=2)+'\n', encoding='utf-8')
    print('ENCODE PASS',output, 'sha256=',doc['frame_sha256'], 'bytes=',output.stat().st_size)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--background',required=True,type=Path)
    p.add_argument('--frame',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--manifest',required=True,type=Path)
    args=p.parse_args()
    encode(args.background,args.frame,args.out,args.manifest)
