#!/usr/bin/env python3
"""JARVIS-X bounded reference runtime. Python 3.9+, standard library.
Never executes uploaded code. Container is lossless, not encrypted/authenticated.
"""
import argparse, hashlib, json, math, os, struct, tempfile, time, zlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import deque

N = 512
MASK = (1 << 30) - 1
MAGIC = b'JX3D\x01'
HEADER = struct.Struct('>BHI32s')  # mode, original length, payload length, SHA256
OPS = {1:'MAP', 2:'ENCODE', 3:'DECODE', 4:'VERIFY', 5:'CTR', 6:'TRACE'}
PROGRAM = tuple(op << 56 for op in OPS)

def sha(b): return hashlib.sha256(b).digest()
def clamp(x, lo, hi): return max(lo, min(hi, x))
def morton(x,y,z):
    if not all(0 <= a <= MASK for a in (x,y,z)): raise ValueError('90-bit address overflow')
    return sum(((a >> b) & 1) << (3*b+k) for k,a in enumerate((x,y,z)) for b in range(30))
def unmorton(m):
    if not 0 <= m < 1<<90: raise ValueError('90-bit address overflow')
    return tuple(sum(((m >> (3*b+k))&1)<<b for b in range(30)) for k in range(3))
def coord(i): return i&7, (i>>3)&7, (i>>6)&7

def trunc(a,b): return (abs(a)//b)*(-1 if a<0 else 1)
class ANN:
    """Causal 6->8->1 integer network; fixed hidden layer, online output learning."""
    def __init__(self): self.w=[0]*8; self.bias=128*256
    def features(self, h):
        a,b,c,d,e,f=[v-128 for v in h]
        return [a,b,c,d,e,f,trunc(a+b+c,3),clamp(a-b+c,-128,127)]
    def predict(self,h):
        x=self.features(h)
        return clamp((self.bias+sum(a*b for a,b in zip(self.w,x)))//256,0,255),x
    def learn(self,actual,p,x):
        err=actual-p
        self.bias=clamp(self.bias+16*err,0,255*256)
        self.w=[clamp(w+trunc(err*h,256),-512,512) for w,h in zip(self.w,x)]

def context(buf,i):
    x,y,z=coord(i)
    return [buf[i-d] if ok else 128 for d,ok in
            ((1,x>0),(8,y>0),(64,z>0),(2,x>1),(16,y>1),(128,z>1))]

def transform(data,decode=False):
    model=ANN(); seen=bytearray(); result=bytearray(); errors=0
    for i,v in enumerate(data):
        p,h=model.predict(context(seen,i))
        actual=v^p if decode else v
        result.append(actual if decode else actual^p)
        errors+=(actual-p)**2
        seen.append(actual); model.learn(actual,p,h)
    return bytes(result),errors,model

def inflate(payload,n):
    d=zlib.decompressobj(); out=d.decompress(payload,n+1)
    if len(out)!=n or not d.eof or d.unused_data or d.unconsumed_tail:
        raise ValueError('invalid bounded compressed payload')
    return out

def execute(task):
    idx,src,fault=task
    # Address each tile origin in a 2^30-per-axis logical grid.
    if idx >= 1<<81: raise ValueError('tile address overflow')
    tile=(idx&((1<<27)-1),(idx>>27)&((1<<27)-1),idx>>54)
    origin=tuple(a*8 for a in tile); addr=morton(*origin)
    baseline=zlib.compress(src,9); mode=0; payload=baseline
    for word in PROGRAM:
        op=word>>56
        if op==1: assert unmorton(addr)==origin
        elif op==2:
            residual,sse,model=transform(src); candidate=zlib.compress(residual,9)
            if fault: candidate=bytes([candidate[0]^1])+candidate[1:]
        elif op==3:
            try: decoded,_,_=transform(inflate(candidate,len(src)),True)
            except (ValueError,zlib.error): decoded=None
        elif op==4: exact=decoded==src
        elif op==5:
            accepted=exact and len(candidate)<=len(baseline)
            if accepted: mode,payload=1,candidate
        elif op==6:
            reason='commit' if accepted else ('integrity' if not exact else 'size')
            trace={'tile':idx,'morton90':f'{addr:023x}','origin':origin,
                   'input_sha256':sha(src).hex(),'mode':mode,'reason':reason,
                   'raw_bytes':len(src),'baseline_bytes':len(baseline),
                   'candidate_bytes':len(candidate),'payload_bytes':len(payload),
                   'candidate_exact':exact,'prediction_mse':sse/len(src),
                   'learned_weights':model.w,'learned_bias_q8':model.bias}
    return mode,payload,trace

def tasks(f,fault_every):
    i=0
    while True:
        b=f.read(N)
        if not b: break
        yield i,b,bool(fault_every and (i+1)%fault_every==0); i+=1

def ordered(pool,items,window):
    q=deque()
    for item in items:
        q.append(pool.submit(execute,item))
        if len(q)>=window: yield q.popleft().result()
    while q: yield q.popleft().result()

def atomic_path(path):
    fd,tmp=tempfile.mkstemp(prefix='.jx-',dir=str(Path(path).resolve().parent)); os.close(fd)
    return tmp

def check_paths(*paths):
    a=[Path(p).resolve() for p in paths if p is not None]
    if len(a)!=len(set(a)): raise ValueError('input, output and trace must differ')

def encode(src,dst,trace=None,workers=1,fault_every=0):
    check_paths(src,dst,trace)
    if not 1<=workers<=16 or fault_every<0: raise ValueError('bad worker/fault limit')
    tmp=atomic_path(dst); trtmp=atomic_path(trace) if trace else None
    start=time.perf_counter(); count=raw=accepted=0; full=hashlib.sha256()
    try:
        with open(src,'rb') as f,open(tmp,'wb') as o,ThreadPoolExecutor(max_workers=workers) as pool:
            t=open(trtmp,'w',encoding='utf-8') if trtmp else None
            try:
                o.write(MAGIC)
                for mode,payload,r in ordered(pool,tasks(f,fault_every),2*workers):
                    # Hash chain over ordered block hashes, includes boundaries via lengths.
                    digest=bytes.fromhex(r['input_sha256']); n=r['raw_bytes']
                    full.update(struct.pack('>H',n)+digest)
                    o.write(HEADER.pack(mode,n,len(payload),digest)); o.write(payload)
                    count+=1;raw+=n;accepted+=mode
                    if t: t.write(json.dumps(r,sort_keys=True)+'\n')
                o.write(HEADER.pack(255,0,0,full.digest()))
            finally:
                if t: t.close()
        os.replace(tmp,dst)
        if trtmp: os.replace(trtmp,trace)
    finally:
        for p in (tmp,trtmp):
            if p and os.path.exists(p): os.unlink(p)
    return {'blocks':count,'raw_bytes':raw,'container_bytes':os.path.getsize(dst),
            'neural_commits':accepted,'rollbacks':count-accepted,
            'elapsed_seconds':time.perf_counter()-start,'chain_sha256':full.hexdigest()}

def readexact(f,n):
    b=f.read(n)
    if len(b)!=n: raise ValueError('truncated container')
    return b

def decode(src,dst):
    check_paths(src,dst); tmp=atomic_path(dst); full=hashlib.sha256(); count=raw=0
    try:
        with open(src,'rb') as f,open(tmp,'wb') as o:
            if readexact(f,len(MAGIC))!=MAGIC: raise ValueError('bad magic')
            while True:
                mode,n,k,digest=HEADER.unpack(readexact(f,HEADER.size))
                if mode==255:
                    if n or k or digest!=full.digest() or f.read(1): raise ValueError('bad footer')
                    break
                if mode not in (0,1) or not 1<=n<=N or not 1<=k<=N+64:
                    raise ValueError('invalid frame')
                b=inflate(readexact(f,k),n)
                if mode: b,_,_=transform(b,True)
                if sha(b)!=digest: raise ValueError('block digest mismatch')
                full.update(struct.pack('>H',n)+digest); o.write(b); count+=1;raw+=n
        os.replace(tmp,dst)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    return {'blocks':count,'raw_bytes':raw,'chain_sha256':full.hexdigest()}

def generate(seed,n,dst,kind):
    """Bounded autoregressive byte synthesis; not semantic generation."""
    if not 1<=n<=N: raise ValueError('generation limit 512 samples')
    seed=seed[:N]; model=ANN(); seen=bytearray()
    for i,b in enumerate(seed):
        p,h=model.predict(context(seen,i));model.learn(b,p,h);seen.append(b)
    out=bytearray()
    for i in range(n):
        p,h=model.predict(context(out,i));out.append(p)
    if kind=='pgm':
        width=8; height=(n+7)//8; data=bytes(out)+bytes(width*height-n)
        Path(dst).write_bytes(f'P5\n{width} {height}\n255\n'.encode()+data)
    elif kind=='wav':
        import wave
        with wave.open(str(dst),'wb') as f:
            f.setnchannels(1);f.setsampwidth(1);f.setframerate(8000);f.writeframes(out)
    else: Path(dst).write_bytes(out)
    return {'generated_samples':n,'kind':kind,'semantic_generation':False}

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('encode');a.add_argument('src');a.add_argument('dst');a.add_argument('--trace');a.add_argument('--workers',type=int,default=1);a.add_argument('--fault-every',type=int,default=0)
    a=sub.add_parser('decode');a.add_argument('src');a.add_argument('dst')
    a=sub.add_parser('generate');a.add_argument('src');a.add_argument('dst');a.add_argument('--count',type=int,default=512);a.add_argument('--kind',choices=['raw','pgm','wav'],default='raw')
    a=sub.add_parser('bytecode');a.add_argument('dst')
    q=p.parse_args()
    if q.cmd=='encode': r=encode(q.src,q.dst,q.trace,q.workers,q.fault_every)
    elif q.cmd=='decode': r=decode(q.src,q.dst)
    elif q.cmd=='generate':
        check_paths(q.src,q.dst)
        with open(q.src,'rb') as f: r=generate(f.read(N),q.count,q.dst,q.kind)
    else:
        Path(q.dst).write_bytes(b''.join(struct.pack('>Q',w) for w in PROGRAM));r={'words':len(PROGRAM),'hex':[f'{w:016x}' for w in PROGRAM]}
    print(json.dumps(r,sort_keys=True))
if __name__=='__main__': main()
