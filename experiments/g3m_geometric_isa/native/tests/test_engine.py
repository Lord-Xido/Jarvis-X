#!/usr/bin/env python3
"""End-to-end smoke tests for the compiled bounded G3M LibTorch executable.
Uses only Python standard library; creates no production financial data.
"""
import array
import hashlib
import math
import pathlib
import subprocess
import sys
import tempfile


def call(exe, cwd, *args, success=True):
    r = subprocess.run([str(exe), *map(str, args)], cwd=cwd, capture_output=True, text=True)
    if success and r.returncode != 0:
        raise AssertionError(f"command failed: {r.returncode}\n{r.stdout}\n{r.stderr[:1000]}")
    if not success and r.returncode == 0:
        raise AssertionError("invalid input unexpectedly accepted")
    return r


def main():
    exe = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else 'build/g3m').resolve()
    assert exe.is_file(), f"executable not found: {exe}"
    with tempfile.TemporaryDirectory(prefix="g3m_verified_") as tmp:
        root = pathlib.Path(tmp)
        vals = array.array('f')
        for i in range(256):
            t = 2*math.pi*i/256
            vals.extend((2+math.cos(t), -1+math.sin(t), 4+0.3*math.cos(7*t)))
        (root/'in.f32').write_bytes(vals.tobytes())
        a=call(exe,root,'--input','in.f32','--support',32,'--epochs',5,'--decode',64,
               '--refine',3,'--save-weights','weights.pt','--output','first.f32')
        b=call(exe,root,'--input','in.f32','--support',32,'--weights','weights.pt',
               '--decode',64,'--refine',3,'--output','second.f32')
        assert (root/'first.f32').read_bytes()==(root/'second.f32').read_bytes()
        assert (root/'first.f32').stat().st_size == 64*3*4
        print('PASS: train → decode → save/load deterministic equality', hashlib.sha256((root/'first.f32').read_bytes()).hexdigest())
        (root/'bad.f32').write_bytes(b'xyz')
        call(exe,root,'--input','bad.f32','--support',32,'--epochs',2,success=False)
        print('PASS: truncated data rejected')
        bad=array.array('f',[float('nan'),0,0]*64)
        (root/'nan.f32').write_bytes(bad.tobytes())
        call(exe,root,'--input','nan.f32','--support',32,'--epochs',2,success=False)
        print('PASS: NaN data rejected')
        m=call(exe,root,'--points',1000000,'--support',64,'--epochs',2,'--decode',1000000,
               '--refine',1,'--output','million.f32')
        assert (root/'million.f32').stat().st_size==12000000
        assert 'INPUT 1000000x3' in m.stdout and 'RECONSTRUCTION 1000000x3' in m.stdout
        print('PASS: bounded-support 1M input/output execution; output=12,000,000 bytes')
    print('PASS: all bounded end-to-end checks')

if __name__=='__main__':
    main()
