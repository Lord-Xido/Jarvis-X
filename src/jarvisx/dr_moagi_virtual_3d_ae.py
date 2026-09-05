"""Sparse virtual 3D bitstream AE/AD with inward fixed-point feedback.

The septillion^septillion value is a symbolic logical address-space contract;
only a finite active tile is ever materialized.
"""
from __future__ import annotations
import argparse, hashlib, json, math, random, time
from dataclasses import dataclass, asdict

Coord=tuple[int,int,int]
LOGICAL_STREAM_COUNT="(10^24)^(10^24) = 10^(24 * 10^24)"
VIRTUAL_SIDE_LENGTH="10^(8 * 10^24)"
N6=((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1))

@dataclass(frozen=True)
class Config:
    tile:int=8; bits:int=256; latent:int=32; passes:int=5
    alpha:float=.35; beta:float=.65; curvature:float=1.; seed:int=1337
    origin:Coord=(0,0,0); periodic:bool=True; epsilon:float=1e-3
    def __post_init__(self):
        if self.tile<1 or self.bits<2 or not 1<=self.latent<=self.bits or self.passes<1: raise ValueError("invalid dimensions")
        if any(not 0<=x<=1 for x in (self.alpha,self.beta,self.curvature)): raise ValueError("alpha,beta,curvature must be in [0,1]")
        if self.epsilon<0: raise ValueError("epsilon must be >= 0")

@dataclass(frozen=True)
class Metrics:
    pass_index:int; reconstruction_loss:float; cycle_loss:float; reality_gap:float
    changed_bits:int; active_streams:int; seconds:float; updates_per_second:float

def hd(a:int,b:int)->int:return (a^b).bit_count()
def hf(a:int,b:int,n:int)->float:return hd(a,b)/n if n else 0.

def stream(coord:Coord,n:int,seed:int)->int:
    need=(n+7)//8; raw=bytearray(); k=0
    while len(raw)<need:
        p=f"{seed}|{coord[0]}|{coord[1]}|{coord[2]}|{k}".encode(); k+=1
        raw.extend(hashlib.blake2b(p,digest_size=64,person=b"DM3D-AE-v1").digest())
    return int.from_bytes(raw[:need],"little")&((1<<n)-1)

def feedback_mask(n:int,beta:float,seed:int)->int:
    rng=random.Random(seed^0xD34D1A61); out=0
    for b in rng.sample(range(n),round(beta*n)):out|=1<<b
    return out

class Codec:
    def __init__(self,n:int,d:int):
        self.n=n;self.d=d;self.full=(1<<n)-1; q,r=divmod(n,d);s=0;self.groups=[]
        for j in range(d):
            w=q+(j<r);m=((1<<w)-1)<<s;s+=w;self.groups.append((w,m))
    def encode(self,x:int)->int:
        z=0
        for j,(w,m) in enumerate(self.groups):
            if (x&m).bit_count()*2>=w:z|=1<<j
        return z
    def decode(self,z:int)->int:
        x=0
        for j,(_,m) in enumerate(self.groups):
            if z>>j&1:x|=m
        return x&self.full
    def cycle_loss(self,z:int)->float:return hf(z,self.encode(self.decode(z)),self.d)

class Tile:
    def __init__(self,c:Config):
        self.c=c;ox,oy,oz=c.origin;n=c.tile
        self.coords=[(ox+x,oy+y,oz+z) for z in range(n) for y in range(n) for x in range(n)]
    def __len__(self):return len(self.coords)
    def local(self,p):ox,oy,oz=self.c.origin;return p[0]-ox,p[1]-oy,p[2]-oz
    def global_(self,p):ox,oy,oz=self.c.origin;return p[0]+ox,p[1]+oy,p[2]+oz
    def neighbours(self,p):
        x,y,z=self.local(p);n=self.c.tile;out=[]
        for dx,dy,dz in N6:
            q=(x+dx,y+dy,z+dz)
            if self.c.periodic:out.append(self.global_((q[0]%n,q[1]%n,q[2]%n)))
            elif all(0<=v<n for v in q):out.append(self.global_(q))
        return out

def couple(tile:Tile,Z:dict[Coord,int],d:int,a:float)->dict[Coord,int]:
    if not a:return dict(Z)
    out={}
    for p in tile.coords:
        ns=tile.neighbours(p);z=0
        for j in range(d):
            s=1 if Z[p]>>j&1 else -1
            m=sum(1 if Z[q]>>j&1 else -1 for q in ns)/len(ns) if ns else s
            if (1-a)*s+a*m>=0:z|=1<<j
        out[p]=z
    return out

def embed(local:Coord,n:int,k:float,r:float=22.,ratio:float=1.)->tuple[float,float,float]:
    if n<=1:x=y=z=0.
    else:x,y,z=((v/(n-1))*2-1 for v in local)
    if k<=1e-12:return x*r,y*r*.5,z*r*.5
    th=x*math.pi*k;minor=2.5*max(.15,ratio)
    return r*math.sin(th),y*8+math.cos(2*th)*1.5*k,-r*(1-math.cos(th))+z*minor

class DrMoagiVirtual3DAE:
    def __init__(self,c:Config|None=None):
        self.c=c or Config();self.tile=Tile(self.c);self.codec=Codec(self.c.bits,self.c.latent)
        self.full=(1<<self.c.bits)-1;self.mask=feedback_mask(self.c.bits,self.c.beta,self.c.seed)
        self.original:dict[Coord,int]={};self.state:dict[Coord,int]={};self.latent:dict[Coord,int]={};self.coupled:dict[Coord,int]={};self.decoded:dict[Coord,int]={}
    @property
    def active_streams(self):return len(self.tile)
    def materialize(self):
        for p in self.tile.coords:self.original[p]=self.state[p]=stream(p,self.c.bits,self.c.seed)
    def run(self)->list[Metrics]:
        if not self.state:self.materialize()
        hist=[]
        for k in range(self.c.passes):
            t=time.perf_counter();self.latent={p:self.codec.encode(x) for p,x in self.state.items()}
            self.coupled=couple(self.tile,self.latent,self.c.latent,self.c.alpha)
            self.decoded={p:self.codec.decode(z) for p,z in self.coupled.items()}
            rec=sum(hf(self.original[p],self.decoded[p],self.c.bits) for p in self.tile.coords)/len(self.tile)
            cyc=sum(self.codec.cycle_loss(self.coupled[p]) for p in self.tile.coords)/len(self.tile)
            nxt={};changed=0;anchor=self.full^self.mask
            for p in self.tile.coords:
                v=((self.original[p]&anchor)|(self.decoded[p]&self.mask))&self.full
                changed+=hd(self.state[p],v);nxt[p]=v
            self.state=nxt;gap=changed/(len(self.tile)*self.c.bits);dt=max(time.perf_counter()-t,1e-12)
            hist.append(Metrics(k,rec,cyc,gap,changed,len(self.tile),dt,len(self.tile)/dt))
            if gap<=self.c.epsilon:break
        return hist
    def geometry(self,count:int=6):
        ratio=max(.15,(self.c.latent/self.c.bits)**(1/3));out=[]
        for p in self.tile.coords[:count]:
            q=self.tile.local(p);out.append({"virtual_coord":p,"input_position_3d":embed(q,self.c.tile,self.c.curvature),"latent_position_3d":embed(q,self.c.tile,self.c.curvature,22*ratio,ratio)})
        return out
    def summary(self,h):
        return {"logical_stream_universe":LOGICAL_STREAM_COUNT,"virtual_cube_side":VIRTUAL_SIDE_LENGTH,"active_streams":len(self.tile),"codec":[self.c.bits,self.c.latent,self.c.bits],"compression_ratio":self.c.bits/self.c.latent,"alpha":self.c.alpha,"beta":self.c.beta,"curvature":self.c.curvature,"passes":len(h),"final":asdict(h[-1]) if h else None,"geometry":self.geometry()}

def _origin(s):
    p=tuple(map(int,s.split(',')))
    if len(p)!=3:raise argparse.ArgumentTypeError("origin must be x,y,z")
    return p

def main():
    p=argparse.ArgumentParser();p.add_argument('--tile',type=int,default=8);p.add_argument('--bits',type=int,default=256);p.add_argument('--latent',type=int,default=32);p.add_argument('--passes',type=int,default=5);p.add_argument('--alpha',type=float,default=.35);p.add_argument('--beta',type=float,default=.65);p.add_argument('--curvature',type=float,default=1.);p.add_argument('--seed',type=int,default=1337);p.add_argument('--origin',type=_origin,default=(0,0,0));p.add_argument('--epsilon',type=float,default=1e-3);p.add_argument('--no-periodic',action='store_true');p.add_argument('--json',action='store_true');a=p.parse_args()
    e=DrMoagiVirtual3DAE(Config(a.tile,a.bits,a.latent,a.passes,a.alpha,a.beta,a.curvature,a.seed,a.origin,not a.no_periodic,a.epsilon));h=e.run()
    for m in h:print(f"pass={m.pass_index:02d} rec={m.reconstruction_loss:.6f} cycle={m.cycle_loss:.6f} gap={m.reality_gap:.6f} changed={m.changed_bits:,} updates/s={m.updates_per_second:,.0f}")
    if a.json:print(json.dumps(e.summary(h),indent=2))
if __name__=='__main__':main()
