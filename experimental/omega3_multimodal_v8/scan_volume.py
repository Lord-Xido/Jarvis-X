"""Scan an independently verified S3M7 volume. LOAD8 extension: full byte address Q, shard 0."""
import argparse,json,hashlib
import numpy as np
from independent import decode_file

def scan(path,x,y,z):
 a,m=decode_file(path);d,h,w,c=a.shape
 if not (0<=x<w and 0<=y<h and 0<=z<d):raise ValueError('coordinate outside volume')
 base=((z*h+y)*w+x)*c;flat=a.ravel();words=[(7<<60)|((base+j)<<2) for j in range(c)];values=[]
 for inst in words:
  op=inst>>60;q=(inst>>2)&((1<<58)-1);s=inst&3
  if op!=7 or s!=0 or q>=flat.size:raise ValueError('LOAD8')
  values.append(int(flat[q]))
 prev=a[z-1,y,x] if z else np.zeros(c,np.uint8);v=np.array(values,np.uint8);delta=v^prev
 return dict(kind=m['kind'],shape=list(a.shape),xyz=[x,y,z],address=base,values=values,bits=[f'{b:08b}' for b in values],previous=prev.tolist(),xor=delta.tolist(),load8_words=[f'{i:016X}' for i in words],changed_voxels_from_previous=int(np.count_nonzero(np.any(a[z]^(a[z-1] if z else np.zeros_like(a[z])),axis=-1))),sha256=hashlib.sha256(a).hexdigest())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('packet');p.add_argument('x',type=int);p.add_argument('y',type=int);p.add_argument('z',type=int);a=p.parse_args();print(json.dumps(scan(a.packet,a.x,a.y,a.z),indent=2))
