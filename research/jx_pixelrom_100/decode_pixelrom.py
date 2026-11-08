#!/usr/bin/env python3
"""Independent Python JX-PROM-100 pixel decoder and integrity verifier."""
import argparse
import hashlib
import struct
import zlib
from pathlib import Path
from PIL import Image

BANKS=100
N=10240
IMAGE_SIZE=(7856,4496)
PATCH=(128,32)
HEADER=128
MAGIC=b'JXPRB001'
FRAME_BYTES=N*BANKS

def patch_xy(idx):
    row,col=divmod(idx,10)
    return 16+col*784+632,16+row*448+392

def decode(path:Path)->bytes:
    with Image.open(path) as source:
        if source.format!='PNG' or source.size!=IMAGE_SIZE:
            raise ValueError('Not the expected lossless Pixel ROM PNG')
        image=source.convert('RGB')
    frame=bytearray(FRAME_BYTES)
    reference_digest=None
    for idx in range(BANKS):
        x,y=patch_xy(idx)
        data=image.crop((x,y,x+PATCH[0],y+PATCH[1])).tobytes()
        if len(data)!=12288 or data[:8]!=MAGIC:
            raise ValueError(f'bank {idx}: magic or pixel byte count invalid')
        fields=struct.unpack_from('<8I',data,8)
        num,total,offset,length,fullsize,version,crc,flags=fields
        if (num,total,offset,length,fullsize,version,flags)!=(idx,BANKS,idx*N,N,FRAME_BYTES,1,0):
            raise ValueError(f'bank {idx}: address/header mismatch')
        header=bytearray(data[:HEADER]);saved_crc=struct.unpack_from('<I',header,104)[0]
        header[104:108]=b'\0\0\0\0'
        if zlib.crc32(header)!=saved_crc:
            raise ValueError(f'bank {idx}: header CRC mismatch')
        digest=data[40:72]
        if reference_digest is not None and reference_digest!=digest:
            raise ValueError(f'bank {idx}: wrong full-frame digest')
        reference_digest=digest
        payload=data[HEADER:HEADER+N]
        if zlib.crc32(payload)!=crc or hashlib.sha256(payload).digest()!=data[72:104]:
            raise ValueError(f'bank {idx}: payload digest mismatch')
        if any(data[HEADER+N:]):
            raise ValueError(f'bank {idx}: padding corruption')
        frame[idx*N:(idx+1)*N]=payload
    if hashlib.sha256(frame).digest()!=reference_digest:
        raise ValueError('full control digest mismatch')
    if bytes(frame[:8])!=b'JXSF1T12':
        raise ValueError('underlying VM frame signature invalid')
    if zlib.crc32(frame[:-4])!=struct.unpack_from('<I',frame,FRAME_BYTES-4)[0]:
        raise ValueError('underlying VM frame CRC invalid')
    return bytes(frame)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('png',type=Path)
    p.add_argument('--out',type=Path)
    args=p.parse_args()
    frame=decode(args.png)
    if args.out:
        args.out.write_bytes(frame)
    print('PIXEL ROM PYTHON VERIFY PASS banks=',BANKS,'bytes=',len(frame),'sha256=',hashlib.sha256(frame).hexdigest())
