#!/usr/bin/env python3
"""Independent Pixel ROM -> C++ VM integration and fault injection."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import tempfile
from PIL import Image
from decode_pixelrom import decode, patch_xy

def run(cmd, want=0):
    r=subprocess.run([str(x) for x in cmd], text=True,capture_output=True)
    if want==0 and r.returncode!=0:raise RuntimeError(f'{cmd}\n{r.stdout}\n{r.stderr}')
    if want!=0 and r.returncode==0:raise RuntimeError(f'Unwanted success: {cmd}')
    return r.stdout+r.stderr

def test(base:Path, cpp:Path, vm:Path):
    png=base/'JARVIS_X_PIXEL_ROM_100.png'
    reference=(base/'reference_control.jxsf').read_bytes()
    py=decode(png)
    assert py==reference, 'Python extracted wrong frame'
    with tempfile.TemporaryDirectory(prefix='jxprom_') as d:
        wd=Path(d)
        recovered=wd/'recovered.jxsf'
        result=run([cpp,'--decode',png,recovered])
        assert 'JX-PROM-100 PASS' in result
        assert recovered.read_bytes()==reference,'C++ decoded wrong frame'
        a8=wd/'8.jxar'; a1=wd/'1.jxar'
        for workers,ar in [(8,a8),(1,a1)]:
            run([vm,'--run-control','--active',32,'--workers',workers,'--iterations',2,
                 '--seed',42,'--control',recovered,'--archive',ar,
                 '--ply',wd/f'{workers}.ply'])
            run([vm,'--verify','--control',recovered,'--archive',ar])
        assert a1.read_bytes()==a8.read_bytes(),'parallel execution produced divergent archives'
        # Single RGB byte corruption: PNG remains valid; bank CRC/SHA must reject.
        with Image.open(png) as original:
            image=original.convert('RGB')
            x,y=patch_xy(17)
            r,g,b=image.getpixel((x+22,y+11))
            image.putpixel((x+22,y+11),((r^1),g,b))
            corrupt=wd/'corrupt.png'
            image.save(corrupt,format='PNG',compress_level=1)
        try:
            decode(corrupt)
            raise AssertionError('Python accepted corrupt payload')
        except ValueError:
            pass
        run([cpp,'--decode',corrupt,wd/'bad.jxsf'],want=1)
        # Wrong-dimension RGB PNG: rejected before attempting bank reads.
        with Image.open(png) as original:
            thumb=original.crop((0,0,256,256))
            wrong=wd/'wrong_dimensions.png';thumb.save(wrong)
        run([cpp,'--decode',wrong,wd/'bad2.jxsf'],want=1)
    print('INTEGRATION PASS: independent Python/C++ recovery; 100 banks; SHA-256',hashlib.sha256(reference).hexdigest(),
          '; 1/8 worker byte-identical VM execution; tamper rejection; wrong-dimensions rejection')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('base',type=Path)
    p.add_argument('cpp',type=Path)
    p.add_argument('vm',type=Path)
    a=p.parse_args()
    test(a.base,a.cpp,a.vm)
