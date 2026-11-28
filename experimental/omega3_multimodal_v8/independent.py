"""Standalone S3M7 decoder. No engine import, independent neural arithmetic."""
import json,struct,zlib,hashlib
import numpy as np
def decode_file(path):
 def read(f,n):
  b=f.read(n)
  if len(b)!=n:raise ValueError('truncated')
  return b
 with open(path,'rb') as f:
  magic,n=struct.unpack('<4sI',read(f,8))
  if magic!=b'S3M7' or n>1000000:raise ValueError('header')
  m=json.loads(read(f,n));z,y,x,c=m['shape'];total=z*y*x*c
  if not all(isinstance(v,int) and 0<v<=8192 for v in (z,y,x,c)) or c>4 or total>2_000_000_000:raise ValueError('dimensions')
  model=m['model'];w=np.array(model['w1'],np.int64);b=np.array(model['b1'],np.int64);v=np.array(model['w2'],np.int64);q=int(model['b2'])
  if w.shape!=(5,8) or b.shape!=(8,) or v.shape!=(8,) or max(np.max(abs(w)),np.max(abs(b)),np.max(abs(v)),abs(q))>1000000:raise ValueError('model')
  a=np.empty((z,y,x,c),np.uint8);p=np.zeros((y,x,c),np.uint8);sz=p.size
  for i in range(z):
   mode,k,h=struct.unpack('<BI32s',read(f,37))
   if mode>3 or k>sz+1024:raise ValueError('record')
   data=read(f,k)
   if mode:
    dc=zlib.decompressobj();data=dc.decompress(data,sz+1)
    if not dc.eof or dc.unused_data or dc.unconsumed_tail:raise ValueError('compression')
   if len(data)!=sz:raise ValueError('size')
   r=np.frombuffer(data,np.uint8).reshape(y,x,c).copy()
   if mode==2:r=np.bitwise_xor(r,p)
   if mode==3 and np.any(p):
    # Explicit sum over input features, independent of engine's matmul.
    feat=[p,np.concatenate((p[-1:],p[:-1]),0),np.concatenate((p[1:],p[:1]),0),np.concatenate((p[:,-1:],p[:,:-1]),1),np.concatenate((p[:,1:],p[:,:1]),1)]
    hidden=np.broadcast_to(b,(*p.shape,8)).copy()
    for j in range(5):hidden+=feat[j].astype(np.int64)[...,None]*w[j]
    hidden=np.maximum(hidden//256,0);pred=np.full(p.shape,q,np.int64)
    for j in range(8):pred+=hidden[...,j]*v[j]
    r^=np.clip(pred//256,0,255).astype(np.uint8)
   if hashlib.sha256(r).digest()!=h:raise ValueError('plane hash')
   a[i]=r;p=r
  if f.read(1) or hashlib.sha256(a).hexdigest()!=m['sha256']:raise ValueError('stream hash/trailing data')
 return a,m
