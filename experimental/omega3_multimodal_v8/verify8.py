import struct,zlib,numpy as np
def decode(path):
 with open(path,'rb') as f:
  def take(n):
   b=f.read(n)
   if len(b)!=n:raise ValueError('truncated')
   return b
  magic,z,y,x,c,n=struct.unpack('<4s5I',take(24));shape=(z,y,x,c);sz=z*y*x*c
  if magic!=b'S3B8' or not 0<sz<=1_000_000 or not 0<n<=2_000_000 or n*sz>2_000_000_000:raise ValueError('header')
  out=np.empty((n,*shape),np.uint8);p=np.zeros(shape,np.uint8)
  for i in range(n):
   mode,k,h=struct.unpack('<BII',take(9))
   if mode>8 or k>sz+max(1024,sz//64):raise ValueError('record')
   w=tuple(take(3)) if mode in (4,8) else None
   if w is not None and sum(w)!=16:raise ValueError('weights')
   b=take(k)
   if 1<=mode<=4:
    a=bytearray();j=0
    while j<len(b):
     v=b[j];j+=1;l=(v&127)+1
     if len(a)+l>sz:raise ValueError('rle length')
     if v&128:a.extend(bytes(l))
     else:
      if j+l>len(b):raise ValueError('rle literal')
      a.extend(b[j:j+l]);j+=l
    b=a
   elif mode>=5:
    d=zlib.decompressobj();b=d.decompress(b,sz+1)
    if not d.eof or d.unused_data or d.unconsumed_tail:raise ValueError('zlib')
   if len(b)!=sz:raise ValueError('length')
   a=np.frombuffer(b,np.uint8).reshape(shape).copy()
   if mode in (2,6):a^=p
   if mode in (3,7):a^=np.roll(p,1,axis=2)
   if mode in (4,8):
    q=p.astype(np.uint16);m=np.roll(q,1,axis=2);s=sum(np.roll(q,d,axis=ax) for ax in range(3) for d in (-1,1));pred=(w[0]*q+w[1]*m+w[2]*((s+3)//6)+8)//16;a^=pred.astype(np.uint8)
   if zlib.crc32(a)!=h:raise ValueError('crc')
   out[i]=a;p=a
  if f.read(1):raise ValueError('trailing')
 return out
