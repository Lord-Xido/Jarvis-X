"""Generate all five bounded media types and encode/verify them in S3H8."""
import argparse,json
from pathlib import Path
import numpy as np
from neural import generate
from engine import ingest,export
from hybrid import pack,read8
p=argparse.ArgumentParser();p.add_argument('out');p.add_argument('--model',default=str(Path(__file__).with_name('generator_refined.pt')));p.add_argument('--frames',type=int,default=240);p.add_argument('--phase',type=float,default=.31);a=p.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
if not 1<=a.frames<=1000:raise ValueError('bounded frame count 1–1000')
r={'generated':generate(a.model,o,a.frames,a.phase),'media':{}}
for kind,name in [('image','generated.png'),('video','generated.mp4'),('audio','generated.wav'),('text','generated.txt'),('geometry','generated_geometry.npy')]:
 v,m=ingest(kind,o/name);q=pack(v,m,o/(kind+'.s3h'));b,md=read8(o/(kind+'.s3h'));assert np.array_equal(v,b);export(b,md,o/('recovered_'+name));z,_=ingest(kind,o/('recovered_'+name));assert np.array_equal(v,z);r['media'][kind]=q
(o/'generation_report.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
