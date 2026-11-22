#!/usr/bin/env python3
"""Independent NumPy mathematical reference; compared with separately compiled C++."""
import hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np
CASES = [('singleton',1,1,1,0,3), ('thin_z',1,9,5,0,3), ('edge_odd',5,7,3,0,3),
         ('tile_impulse',8,8,4,1,3), ('cross_tile_impulse',9,10,5,1,3),
         ('mixed_sign',17,15,9,0,3),('full_random',64,64,32,0,3),
         ('full_constant',64,64,32,2,3),('full_checker',64,64,32,3,3)]

def initial(nx,ny,nz,mode):
    a=np.empty((nz,ny,nx,3),dtype=np.int32)
    seed=0x0A17C0DE
    for z in range(nz):
      for y in range(ny):
       for x in range(nx):
        for c in range(3):
         seed=(seed^(seed<<13))&0xffffffff
         seed=(seed^(seed>>17))&0xffffffff
         seed=(seed^(seed<<5))&0xffffffff
         v=int(seed%8193)-4096
         if mode==1: v = (-4096 if c==1 else 4096) if (x,y,z)==(nx//2,ny//2,nz//2) else 0
         elif mode==2:v=-4096
         elif mode==3:v=4096 if (x+y+z+c)%2 else -4096
         a[z,y,x,c]=v
    return a

def rdiv(a,den):
    return np.where(a>=0,(a+den//2)//den,-((-a+den//2)//den))

def step(a):
    shape=a.shape
    sm=np.zeros(shape,dtype=np.int32)
    nn=np.zeros(shape[:3],dtype=np.int32)
    for axis in range(3):
      left=[slice(None)]*4; right=[slice(None)]*4
      left[axis]=slice(1,None);right[axis]=slice(None,-1)
      L=tuple(left);R=tuple(right)
      sm[L]+=a[R];nn[L[:-1]]+=1
      sm[R]+=a[L];nn[R[:-1]]+=1
    # No-neighbor singleton uses the center value (identity under 17:3 mixing).
    d=np.maximum(nn[...,None],1)
    avg=np.where(nn[...,None]>0,rdiv(sm,d),a)
    return np.clip(rdiv(17*a+3*avg,20),-4096,4096).astype(np.int32)

def run(binary_dir):
    results=[]
    for name,nx,ny,nz,mode,its in CASES:
      a=initial(nx,ny,nz,mode)
      for t in range(1,its+1):
        a=step(a)
        payload=a.astype('<i2').tobytes(order='C')
        cpp=(binary_dir/f'{name}_t{t}.bin').read_bytes()
        if cpp!=payload:
          n=next(i for i,(x,y) in enumerate(zip(cpp,payload)) if x!=y)
          raise AssertionError(f'{name} t={t} C++ vs NumPy first mismatching BYTE {n}: {cpp[n]} != {payload[n]}')
        results.append(dict(case=name,dimensions=[nx,ny,nz],iteration=t,bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest(),cpu_cpp_numpy_exact=True))
    return results
if __name__=='__main__':
  root=Path(__file__).resolve().parents[1]
  out=root/'reference'/'golden_bytes';out.mkdir(parents=True,exist_ok=True)
  subprocess.run(['g++','-std=c++20','-O3','-Wall','-Wextra','-Werror',str(root/'reference'/'oracle.cpp'),'-o',str(root/'reference'/'oracle')],check=True)
  subprocess.run([str(root/'reference'/'oracle'),str(out)],check=True,stdout=subprocess.DEVNULL)
  entries=run(out)
  result={'source':'Independent C++20 and NumPy numerical implementations','cases':len(CASES),'iterations_verified':len(entries),'comparison':'all bytes identical','golden_vectors':entries}
  (root/'reference'/'GOLDEN.json').write_text(json.dumps(result,indent=2)+'\n')
  # Build Swift literal of golden SHA lists.
  for name,*_ in CASES:
    print('"'+name+'": ['+', '.join('"'+next(e['sha256'] for e in entries if e['case']==name and e['iteration']==t)+'"' for t in range(1,4))+'],')
  print('PASS:',len(entries),'C++/NumPy exact fields')
