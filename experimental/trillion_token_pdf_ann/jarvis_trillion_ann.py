#!/usr/bin/env python3
"""Deterministic 3D integer ANN token VM. Requires NumPy; PyMuPDF only for extract."""
import argparse, hashlib, json, math, time
from pathlib import Path
import numpy as np
PROGRAM=bytes.fromhex('01 10 20 50 52 53 60 61 70 FF')
NAMES=['LOAD','MAP','ENCODE','ANN','DECODE','TOKEN','VERIFY','COMMIT','RECURSE','HALT']
def sha(b): return hashlib.sha256(b).hexdigest()
def morton(x,y,z,side):
    return sum(((v>>k)&1)<<(3*k+a) for a,v in enumerate((x,y,z)) for k in range(side.bit_length()-1))
def model():
    def weights(a,b,seed):
        return np.array([[(i*17+j*31+i*j*7+seed)%7-3 for j in range(b)] for i in range(a)],dtype=np.int32)
    return weights(16,8,3),weights(8,8,1),weights(8,16,5),weights(16,32,2)
def scale():
    r=10**12; p=10**9; d=16; h=4; w=64; q=2; c=2048
    kv=2*d*h*w*q; stream=kv*c
    return dict(target_tokens_s=r,logical_streams=10**9,per_stream_tokens_s=1000,parameters=p,flops_per_token=2*p,flops_s=2*p*r,kv_bytes_per_token=kv,kv_bytes_per_stream=stream,kv_capacity_bytes=stream*10**9,kv_read_bytes_s=stream*r,kv_write_bytes_s=kv*r,ideal_compute_devices=math.ceil(2*p*r/10**15),ideal_kv_bandwidth_devices=math.ceil(stream*r/(3*10**12)))
def run(side,rounds,seed):
    if side not in (2,4,8,16,32) or not 1<=rounds<=10000: raise ValueError('side or rounds outside supported range')
    n=side**3; xyz=sorted(((x,y,z) for x in range(side) for y in range(side) for z in range(side)),key=lambda a:morton(*a,side)); neighbors=np.array([[morton((x+dx)%side,(y+dy)%side,(z+dz)%side,side) for dx,dy,dz in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1))] for x,y,z in xyz]); E,W,D,O=model(); source=hashlib.sha256(seed).digest(); s=np.frombuffer((source*((n*16+31)//32))[:n*16],dtype=np.uint8).astype(np.int32).reshape(n,16)%31-15; tokens=np.arange(n,dtype=np.int32)%32; t=pc=ops=checks=0; trace=[]; stream=hashlib.sha256(); start=time.perf_counter()
    def scalar(i):
        a=[max(-127,min(127,int(s[i,j])+sum(int(s[q,j]) for q in neighbors[i])//6+(int(tokens[i])+j)%7-3)) for j in range(16)]
        def mat(v,m): return [sum(int(v[k])*int(m[k,j]) for k in range(len(v))) for j in range(m.shape[1])]
        def act(v): return [max(-127,min(127,x//16)) for x in v]
        e=act(mat(a,E)); h=act(mat(e,W)); out=act(mat(h,D)); logits=mat(out,O); return out, max(range(32),key=lambda k:logits[k])
    while True:
        op=PROGRAM[pc];pc+=1;ops+=1
        if op==1: pass
        elif op==0x10: a=np.clip(s+s[neighbors].sum(axis=1)//6+(tokens[:,None]+np.arange(16))%7-3,-127,127)
        elif op==0x20: e=np.clip((a@E)//16,-127,127)
        elif op==0x50: h=np.clip((e@W)//16,-127,127)
        elif op==0x52: candidate=np.clip((h@D)//16,-127,127).astype(np.int32)
        elif op==0x53: nxt=np.argmax(candidate@O,axis=1).astype(np.int32)
        elif op==0x60:
            if candidate.shape!=(n,16) or np.any(abs(candidate)>127) or np.any((nxt<0)|(nxt>=32)): raise ValueError('CTR range failure')
            # Full scalar reference at first tick; one rotating cell at subsequent ticks.
            for i in (range(n) if t==0 else [(t*7919)%n]):
                u,v=scalar(i); checks+=1
                if u!=candidate[i].tolist() or v!=int(nxt[i]): raise ValueError('CTR reference failure')
        elif op==0x61:
            s,tokens=candidate,nxt; stream.update(tokens.astype('<u4').tobytes());t+=1
            if t in (1,10,100,rounds): trace.append(dict(iteration=t,emitted_tokens=t*n,state_sha256=sha(s.astype('<i4').tobytes()),token_stream_sha256=stream.hexdigest()))
        elif op==0x70:
            if t<rounds: pc=0
        elif op==0xff: break
    sec=time.perf_counter()-start
    return dict(side=side,streams=n,rounds=t,emitted_tokens=n*t,bytecode_dispatches=ops,scalar_cell_checks=checks,reference_errors=0,seconds=sec,measured_tokens_s=n*t/sec,token_stream_sha256=stream.hexdigest(),trace=trace,scope='32-symbol toy integer ANN; no language model or learned weights')
def main():
    a=argparse.ArgumentParser(); a.add_argument('--side',type=int,default=8);a.add_argument('--rounds',type=int,default=1000);a.add_argument('--seed',default='JARVIS-X');a.add_argument('--out');a.add_argument('--extract',nargs=2,metavar=('PDF','DIR'));args=a.parse_args()
    if args.extract:
        import fitz
        dest=Path(args.extract[1]);dest.mkdir(parents=True,exist_ok=True)
        with fitz.open(args.extract[0]) as d:
            for name in d.embfile_names(): (dest/Path(name).name).write_bytes(d.embfile_get(name))
        return
    result={'architecture':scale(),'execution':run(args.side,args.rounds,args.seed.encode())}; data=json.dumps(result,indent=2); print(data)
    if args.out: Path(args.out).write_text(data+'\n')
if __name__=='__main__':main()
