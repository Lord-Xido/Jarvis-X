#!/usr/bin/env python3
"""CPU reference for spatially sharded residual tokenisation; no GPU claims."""
import argparse
import hashlib
import json
import math
import random
from itertools import product

def digest(x):
    return hashlib.sha256(bytes(x)).hexdigest()

def run(n=16, p=2, steps=1000, seed=42, trace=None):
    if n < 4 or p < 1 or n % p or n // p < 2 or steps < 1:
        raise ValueError("require n>=4, p divides n, shard side>=2, steps>=1")
    b = n // p
    coords = list(product(range(n), repeat=3))
    def ix(r):
        x,y,z = r
        return (x*n+y)*n+z
    def neighbors(r):
        for axis in range(3):
            for d in (-1,1):
                q=list(r); q[axis]=(q[axis]+d)%n
                yield tuple(q)
    def owner(r):
        return tuple(v//b for v in r)
    shards = {g:[] for g in product(range(p), repeat=3)}
    for r in coords: shards[owner(r)].append(r)
    boundaries = {g:{q for r in rs for q in neighbors(r) if owner(q)!=g}
                  for g,rs in shards.items()}
    rng=random.Random(seed)
    y=[rng.randrange(256) for _ in coords]
    x=[0]*len(y)
    previous=x[:]
    # Fixed-point forward operator H0=current center, H1=lagged 6-neighbor average.
    # The observation has scale 16; residual is retained in integer arithmetic.
    def forward(r, a, old, read):
        return 10*a[ix(r)]+sum(read(q) for q in neighbors(r))
    def loss(a, old):
        return sum((16*y[ix(r)]-forward(r,a,old,lambda q:old[ix(q)]))**2
                   for r in coords)
    # Compare candidate and incumbent under the SAME frozen history.
    # Token codebook is a bounded scalar correction quantizer, not a trained ANN.
    codes=(-32,-16,-8,-4,-2,-1,0,1,2,4,8,16,32)
    def token(res):
        return min(range(len(codes)), key=lambda j:(abs(res-16*codes[j]),j))
    accepted=rejected=halo_values=0
    initial=loss(x,previous)
    chain="0"*64
    for k in range(1,steps+1):
        candidate=x[:]
        dense=x[:]
        tokens=[0]*len(x)
        for g,rs in shards.items():
            halo={q:previous[ix(q)] for q in boundaries[g]}
            halo_values+=len(halo)
            def read(q):
                return previous[ix(q)] if owner(q)==g else halo[q]
            for r in rs:
                i=ix(r)
                e=16*y[i]-forward(r,x,previous,read)
                j=token(e)
                tokens[i]=j
                candidate[i]=max(0,min(255,x[i]+codes[j]))
        for r in coords:
            i=ix(r)
            e=16*y[i]-forward(r,x,previous,lambda q:previous[ix(q)])
            dense[i]=max(0,min(255,x[i]+codes[token(e)]))
        if candidate!=dense:
            raise RuntimeError("sharded/dense transition mismatch")
        # Lossless side channel is separate from approximate inverse reconstruction.
        residual=bytes(a^v for a,v in zip(y,candidate))
        if bytes(a^v for a,v in zip(residual,candidate))!=bytes(y):
            raise RuntimeError("round trip failure")
        before=loss(x,previous)
        after=loss(candidate,previous)
        # Candidate history advances only on commit; verify next full history state too.
        next_loss=loss(candidate,x)
        commit=(after<before and next_loss<=before and k%97!=0)
        old_x=x[:]; old_prev=previous[:]
        if commit:
            previous,x=x,candidate
            accepted+=1
        else:
            rejected+=1
            if x!=old_x or previous!=old_prev:
                raise RuntimeError("rollback failure")
        record={"step":k,"before":before,"candidate":after,"next_loss":next_loss,
                "committed_loss":loss(x,previous),"commit":commit,
                "state_sha256":digest(x),"history_sha256":digest(previous),
                "token_sha256":digest(tokens),"residual_sha256":digest(residual)}
        chain=hashlib.sha256((chain+json.dumps(record,sort_keys=True)).encode()).hexdigest()
        record["chain"]=chain
        if trace: trace.write(json.dumps(record,sort_keys=True)+"\n")
    return {"n":n,"shards":p**3,"steps":steps,"accepted":accepted,"rejected":rejected,
            "initial_loss":initial,"final_loss":loss(x,previous),
            "halo_values":halo_values,"state_sha256":digest(x),"chain":chain}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--n",type=int,default=8)
    ap.add_argument("--p",type=int,default=2)
    ap.add_argument("--steps",type=int,default=1000)
    ap.add_argument("--trace",default="rraet_trace.jsonl")
    a=ap.parse_args()
    with open(a.trace,"w",encoding="utf-8") as f:
        result=run(a.n,a.p,a.steps,trace=f)
    replay=run(a.n,a.p,a.steps)
    if result!=replay: raise RuntimeError("non-deterministic replay")
    single=run(a.n,1,a.steps)
    for key in ("state_sha256","chain","accepted","rejected","final_loss"):
        if result[key]!=single[key]: raise RuntimeError("partition invariance: "+key)
    print(json.dumps(result,sort_keys=True,indent=2))
if __name__=="__main__":
    main()
