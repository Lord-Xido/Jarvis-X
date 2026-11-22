#!/usr/bin/env python3
"""Build a deterministic demonstration JX-PROM-100 carrier from compiled JX-VM.
The generated PNG can be replaced with any approved 7856x4496 lossless background.
"""
from pathlib import Path
import argparse
import subprocess
from PIL import Image, ImageDraw
from encode_pixelrom import encode

SIZE=(7856,4496)

def background(path:Path):
    im=Image.new('RGB',SIZE,(8,15,30))
    d=ImageDraw.Draw(im)
    for bank in range(100):
        row,col=divmod(bank,10)
        x=16+col*784; y=16+row*448
        d.rectangle((x,y,x+767,y+431),outline=(36,97,155),width=3)
        d.text((x+20,y+20),f'JARVIS-X PIXEL ROM | BANK {bank:02d}',fill=(118,224,243))
        d.rectangle((x+20,y+70,x+740,y+365),outline=(36,97,155),width=1)
        for k in range(1,7):
            xx=x+20+k*102
            d.line((xx,y+70,xx,y+365),fill=(15,46,82))
        for k in range(1,5):
            yy=y+70+k*59
            d.line((x+20,yy,x+740,yy),fill=(15,46,82))
        d.text((x+25,y+385),'10^12 TASK ADDRESS SPACE | JXEX -> DCT3 -> VERIFY',fill=(62,139,194))
    im.save(path,'PNG',compress_level=5)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--vm',type=Path,required=True)
    p.add_argument('--outdir',type=Path,required=True)
    p.add_argument('--background',type=Path)
    args=p.parse_args()
    out=args.outdir.resolve();out.mkdir(parents=True,exist_ok=True)
    control=out/'reference_control.jxsf'
    subprocess.run([str(args.vm.resolve()),'--run','--active','1','--workers','1','--iterations','1',
                    '--control',str(control),'--archive',str(out/'bootstrap.jxar'),
                    '--ply',str(out/'bootstrap.ply')],check=True)
    bg=args.background or (out/'pixelrom_background.png')
    if args.background is None:background(bg)
    encode(bg,control,out/'JARVIS_X_PIXEL_ROM_100.png',out/'pixelrom_manifest.json')
    from decode_pixelrom import decode
    assert decode(out/'JARVIS_X_PIXEL_ROM_100.png')==control.read_bytes()
    print('BOOTSTRAP PASS: recovered exact frame; bank count=100; frame bytes=1024000')
if __name__=='__main__': main()
