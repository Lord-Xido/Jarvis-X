"""S3H8 verified format-selection container and typed-media CLI."""
import argparse,hashlib,json,os,struct,tempfile,subprocess
from pathlib import Path
import numpy as np
from engine import ingest,export,encode
from independent import decode_file
from verify8 import decode as decode8
P=Path(__file__).resolve().parent

def digest(b):return hashlib.sha256(b).hexdigest()
def tiles(a,shape):
 z,y,x,c=a.shape;d,h,w=shape;zp=(z+d-1)//d;yp=(y+h-1)//h;xp=(x+w-1)//w
 p=np.pad(a,((0,zp*d-z),(0,yp*h-y),(0,xp*w-x),(0,0)));t=p.reshape(zp,d,yp,h,xp,w,c).transpose(2,4,0,1,3,5,6).reshape(-1,d,h,w,c);return t,(zp,yp,xp)
def untile(t,shape,counts):
 z,y,x,c=shape;zp,yp,xp=counts;_,d,h,w,_=t.shape
 if (len(t)!=zp*yp*xp or t.shape[-1]!=c or not(0<z<=zp*d and 0<y<=yp*h and 0<x<=xp*w)):raise ValueError('tile manifest')
 return t.reshape(yp,xp,zp,d,h,w,c).transpose(2,3,0,4,1,5,6).reshape(zp*d,yp*h,xp*w,c)[:z,:y,:x,:]
def envelope(payload,header):
 h=json.dumps(dict(header,payload_sha256=digest(payload)),separators=(',',':')).encode();return struct.pack('<4sI',b'S3H8',len(h))+h+payload

def read8(p):
 with open(p,'rb') as f:
  h=f.read(8)
  if len(h)!=8:raise ValueError('header')
  magic,n=struct.unpack('<4sI',h)
  if magic!=b'S3H8' or not 0<n<=1_000_000:raise ValueError('header')
  h=json.loads(f.read(n));payload=f.read()
 if digest(payload)!=h['payload_sha256']:raise ValueError('payload SHA256')
 with tempfile.TemporaryDirectory() as d:
  q=Path(d)/'payload';q.write_bytes(payload)
  if h['format']=='v7':a,m=decode_file(q)
  elif h['format']=='v8':
   m=h['metadata'];a=untile(decode8(q),m['shape'],h['tile_counts'])
  else:raise ValueError('format')
 if digest(np.ascontiguousarray(a))!=h['tensor_sha256']:raise ValueError('tensor SHA256')
 return a,m

def choose(a,m,old,new,counts,out):
 # Complete envelope bytes, including metadata, are compared.
 sha=digest(np.ascontiguousarray(a));candidates=[envelope(Path(old).read_bytes(),{'format':'v7','tensor_sha256':sha}),envelope(Path(new).read_bytes(),{'format':'v8','tensor_sha256':sha,'metadata':m,'tile_counts':counts})];i=min(range(2),key=lambda i:len(candidates[i]));out=Path(out);tmp=out.with_suffix(out.suffix+'.tmp');tmp.write_bytes(candidates[i]);b,md=read8(tmp)
 if not np.array_equal(a,b):tmp.unlink();raise ValueError('verify before commit')
 os.replace(tmp,out);return dict(format=['v7','v8'][i],v7_envelope_bytes=len(candidates[0]),v8_envelope_bytes=len(candidates[1]),selected_bytes=out.stat().st_size,sha256=sha)

def pack(a,m,out):
 a=np.ascontiguousarray(a)
 with tempfile.TemporaryDirectory() as d:
  d=Path(d);old=d/'old';new=d/'new';raw=d/'raw';m=dict(m,shape=list(a.shape),sha256=digest(np.ascontiguousarray(a)))
  encode(a,m,json.loads((P/'predictor.json').read_text()),old);t,counts=tiles(a,(min(4,len(a)),32,32));t.tofile(raw);subprocess.run([str(P/'core'),str(raw),str(new),*map(str,t.shape[1:]),str(len(t)),str(len(t)),'1','1','710'],check=True,stdout=subprocess.DEVNULL)
  return choose(a,m,old,new,counts,out)

def main():
 p=argparse.ArgumentParser();s=p.add_subparsers(dest='cmd',required=True);e=s.add_parser('encode');e.add_argument('kind',choices=['image','video','audio','text','geometry','file']);e.add_argument('input');e.add_argument('output');e.add_argument('--scale',nargs=2,type=int);d=s.add_parser('decode');d.add_argument('input');d.add_argument('output');r=s.add_parser('scan');r.add_argument('input');r.add_argument('x',type=int);r.add_argument('y',type=int);r.add_argument('z',type=int);args=p.parse_args()
 if args.cmd=='encode':a,m=ingest(args.kind,args.input,args.scale);print(json.dumps(pack(a,m,args.output),indent=2))
 elif args.cmd=='decode':a,m=read8(args.input);export(a,m,args.output)
 else:
  a,m=read8(args.input);z,y,x,c=a.shape;X,Y,Z=args.x,args.y,args.z
  if not (0<=X<x and 0<=Y<y and 0<=Z<z):raise ValueError('coordinate')
  v=a[Z,Y,X];q=c*(Z*y*x+Y*x+X);print(json.dumps({'xyz':[X,Y,Z],'values':v.tolist(),'bits':[f'{b:08b}' for b in v],'address':q,'load8':[f'{(7<<60)|((q+i)<<2):016X}' for i in range(c)]},indent=2))
if __name__=='__main__':main()
