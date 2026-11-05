#!/usr/bin/env python3
"""Encode a proposed G3M computational PLAN, not an ANN execution binary.
Spec: 32-byte LE header + 16-byte LE instructions, CRC32 integrity.
"""
import struct, zlib
from pathlib import Path
H = struct.Struct('<8sHHIIIII')  # 32 B
I = struct.Struct('<BBHHHII')    # 16 B
OPS = {
    'LOAD_XYZ':0x01,'NORMALIZE':0x02,'SUPPORT':0x03,'KNN':0x04,
    'EDGECONV':0x05,'GEOATTN':0x06,'VAE512':0x07,'REFINE':0x08,
    'FOLD':0x09,'DENORM':0x0A,'VERIFY':0x0B,'WRITE_XYZ':0x0C,
    'EVIDENCE':0x0D,'TRAIN_ADAM':0x20,'HALT':0xFF
}
NONE = 65535
# opcode, flags, dst, src0, src1, immediate, auxiliary
PROGRAM = [
('LOAD_XYZ',0,0,NONE,NONE,1000000,0),
('NORMALIZE',0,1,0,NONE,0,0),
('SUPPORT',0,2,1,NONE,128,0),
('TRAIN_ADAM',0,10,2,NONE,6,2000), # lr = aux / 1e6 = .002
('KNN',0,3,2,NONE,12,0),
('EDGECONV',0,4,2,3,64,0),
('GEOATTN',0,5,4,2,64,500), # alpha = aux / 1000 = 0.5
('VAE512',0,6,5,10,512,0),
('REFINE',0,7,6,2,2,10), # aux=10 means max geo degradation 10 / 1000 = .01
('FOLD',0,8,7,NONE,1000000,1024),
('DENORM',0,9,8,1,0,0),
('VERIFY',0,11,9,NONE,1000000,0),
('WRITE_XYZ',0,12,9,11,0,0),
('EVIDENCE',0,13,12,11,0,0),
('HALT',0,NONE,NONE,NONE,0,0)
]
body=b''.join(I.pack(OPS[name],flags,dst,src0,src1,imm,aux) for name,flags,dst,src0,src1,imm,aux in PROGRAM)
crc = zlib.crc32(body)&0xffffffff
# magic, version, instruction width, count, flags, max_points, program crc32, reserved
header=H.pack(b'G3MBC001', 1, 16, len(PROGRAM), 1, 1_000_000, crc, 0)
output=Path(__file__).with_name('g3m_pipeline.g3mb')
output.write_bytes(header+body)
print(f'{output}: {len(header+body)} bytes, instructions={len(PROGRAM)}, crc32={crc:08x}')
for pc, op in enumerate(PROGRAM):
    print(f'{pc:02d} 0x{H.size+pc*I.size:04x} {op[0]:12s} dst={op[2]:2} src=({op[3]:5},{op[4]:5}) imm={op[5]} aux={op[6]}')
