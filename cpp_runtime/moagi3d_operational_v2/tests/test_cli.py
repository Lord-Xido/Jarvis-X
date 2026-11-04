#!/usr/bin/env python3
"""Black-box CLI tests for Moagi3D Operational v2.0."""
from __future__ import annotations
import random
from pathlib import Path
import subprocess
import sys
import tempfile
import struct

EXE = Path(sys.argv[1] if len(sys.argv) > 1 else 'build/moagi3d').resolve()
assert EXE.is_file(), f'No executable: {EXE}'

def call(*args: object, expect_success=True) -> subprocess.CompletedProcess:
    result = subprocess.run([str(EXE), *map(str, args)], capture_output=True, text=True)
    if expect_success and result.returncode:
        raise AssertionError(f'FAIL {args}: {result.stdout!r}, {result.stderr!r}')
    if not expect_success and result.returncode == 0:
        raise AssertionError(f'Expected rejection but accepted: {args}')
    return result

with tempfile.TemporaryDirectory() as temp:
    d = Path(temp)
    tests = [
        (1,1,1,1,0), (1,2,5,1,3), (6,5,4,1,4),
        (7,9,11,3,1), (32,24,13,2,2), (16,16,16,4,3),
        (64,64,64,1,4), (3,4,3,16,0)
    ]
    rng = random.Random(20261104)
    tests_run=0
    for i,(w,h,z,c,m) in enumerate(tests):
        n=w*h*z*c
        for p in range(3):
            data=bytes(n) if p==0 else bytes((k*17+k//31)%256 for k in range(n)) if p==1 else rng.randbytes(n)
            raw = d/'in.bin'; archive=d/'out.m3d'; decoded=d/'out.bin'
            raw.write_bytes(data)
            call('encode',raw,archive,'--dims',f'{w}x{h}x{z}','--channels',c,'--modality',m,'--steps',8)
            call('decode',archive,decoded)
            assert decoded.read_bytes()==data
            blob=archive.read_bytes()
            assert blob[:8]==b'MOAGI3D2'
            assert struct.unpack_from('<IIII',blob,8)==(w,h,z,c)
            call('inspect',archive)
            if n>1:
                invalid=bytearray(blob)
                invalid[-1]^=0x80
                (d/'bad.m3d').write_bytes(invalid)
                call('decode',d/'bad.m3d',decoded,expect_success=False)
            tests_run+=1
    (d/'truncated.m3d').write_bytes(b'MOAGI3D2')
    call('decode',d/'truncated.m3d',d/'badout.bin',expect_success=False)
    wrong=bytearray(archive.read_bytes());wrong[8]^=1
    (d/'headerbad.m3d').write_bytes(wrong)
    call('decode',d/'headerbad.m3d',d/'badout.bin',expect_success=False)
    call('encode',raw,d/'wrong.m3d','--dims','1x1x1',expect_success=False)
    call('encode',raw,d/'wrong.m3d','--dims','3x4x3','--channels','0',expect_success=False)
    call('field',archive,d/'field.csv')
    lines=(d/'field.csv').read_text().splitlines()
    assert len(lines)==(3*4*3)+1
    assert lines[0]=='x,y,z,intensity_q16,field_q16'

print(f'BLACK-BOX PASS: {tests_run} CLI round-trips, malformed-file rejection, 3D field CSV export')
