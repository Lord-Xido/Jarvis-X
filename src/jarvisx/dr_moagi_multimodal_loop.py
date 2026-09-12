"""DM-vOmegaXi+ bounded 3D multimodal autoencoding/generation reference loop.

`10**24 states/s` is virtual clock accounting, not measured hardware throughput.
Media adapters are codec-agnostic byte-to-volume projections; semantic codecs belong upstream.
"""
from __future__ import annotations

import argparse, json, math, random, time
from collections import deque
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Sequence

OPS_PER_SECOND = 10**24
VIRTUAL_STATES = 10**24


class Modality(str, Enum):
    VISUAL="visual"; AUDIO="audio"; TEXT="text"; VIDEO="video"; GENERIC="generic"

MODALITIES=tuple(Modality)


@dataclass
class Volume3D:
    edge:int
    values:list[float]
    def __post_init__(self):
        if self.edge<2 or len(self.values)!=self.edge**3: raise ValueError("invalid 3D volume")
        if not all(math.isfinite(v) for v in self.values): raise ValueError("non-finite 3D volume")
    def copy(self): return Volume3D(self.edge,self.values.copy())
    def _same(self,o):
        if self.edge!=o.edge: raise ValueError("3D shape mismatch")
    def mse(self,o):
        self._same(o); return sum((a-b)**2 for a,b in zip(self.values,o.values))/len(self.values)
    def energy(self): return sum(v*v for v in self.values)/len(self.values)
    def blend(self,o,a):
        self._same(o); a=max(0.0,min(1.0,float(a)))
        return Volume3D(self.edge,[(1-a)*x+a*y for x,y in zip(self.values,o.values)])


class Payload3DAdapter:
    PHASE={Modality.VISUAL:.17,Modality.AUDIO:.31,Modality.TEXT:.47,Modality.VIDEO:.63,Modality.GENERIC:.79}
    @classmethod
    def ingest(cls,payload:bytes,modality:Modality,edge:int):
        if edge<4 or edge%2: raise ValueError("edge must be even and >=4")
        payload=payload or b"\0"; n=edge**3; p=len(payload); out=[]; phase=cls.PHASE[modality]
        for i in range(n):
            b0=payload[(131*i+17)%p]; b1=payload[(197*i+43)%p]
            x=i%edge; y=(i//edge)%edge; z=i//(edge*edge)
            v=((.72*b0+.28*b1)/255.0)*2-1
            v+=.08*math.sin(phase+.37*x+.23*y+.19*z)
            if modality is Modality.AUDIO: v+=.07*math.sin(.83*x+.41*z)
            elif modality is Modality.TEXT: v+=.05*(1 if b0&(1<<(i%7)) else -1)
            elif modality is Modality.VIDEO: v+=.06*math.sin(.29*(x+y)+.71*z)
            elif modality is Modality.VISUAL: v+=.05*math.cos(.53*x-.47*y)
            out.append(math.tanh(v))
        return Volume3D(edge,out)
    @staticmethod
    def emit(volume:Volume3D,length:int):
        length=max(1,length); n=len(volume.values); out=bytearray(length)
        for i in range(length):
            v=max(-1.0,min(1.0,volume.values[min(n-1,i*n//length)]))
            out[i]=max(0,min(255,round((v+1)*127.5)))
        return bytes(out)


class Autoencoder3D:
    """Shared-weight 2x2x2 contraction/expansion neural codec with transactional updates."""
    PARAMS=("eg","eb","dg","db")
    def __init__(self,edge:int,seed:int):
        if edge<4 or edge%2: raise ValueError("edge must be even and >=4")
        self.edge=edge; self.le=edge//2; r=random.Random(seed)
        self.eg=.92+.16*r.random(); self.eb=(r.random()-.5)*.04
        self.dg=.92+.16*r.random(); self.db=(r.random()-.5)*.04
        self.accepted=0; self.rejected=0
    @staticmethod
    def idx(e,x,y,z): return (z*e+y)*e+x
    def encode(self,v:Volume3D):
        if v.edge!=self.edge: raise ValueError("encoder edge mismatch")
        o=[0.0]*(self.le**3)
        for z in range(self.le):
          for y in range(self.le):
            for x in range(self.le):
              s=0.0
              for dz in (0,1):
               for dy in (0,1):
                for dx in (0,1): s+=v.values[self.idx(self.edge,2*x+dx,2*y+dy,2*z+dz)]
              o[self.idx(self.le,x,y,z)]=math.tanh(self.eg*s/8+self.eb)
        return Volume3D(self.le,o)
    def decode(self,zv:Volume3D):
        if zv.edge!=self.le: raise ValueError("decoder edge mismatch")
        o=[0.0]*(self.edge**3)
        for z in range(self.le):
          for y in range(self.le):
            for x in range(self.le):
              q=zv.values[self.idx(self.le,x,y,z)]; base=math.tanh(self.dg*q+self.db)
              for dz in (0,1):
               for dy in (0,1):
                for dx in (0,1):
                 detail=.025*((dx+2*dy+3*dz)-3.0)
                 o[self.idx(self.edge,2*x+dx,2*y+dy,2*z+dz)]=math.tanh(base+detail*q)
        return Volume3D(self.edge,o)
    def loss(self,v): return v.mse(self.decode(self.encode(v)))
    def train_transactional(self,v,lr=.08,h=.002):
        base=self.loss(v); old={p:getattr(self,p) for p in self.PARAMS}; g={}
        for p in self.PARAMS:
            x=getattr(self,p); setattr(self,p,x+h); a=self.loss(v); setattr(self,p,x-h); b=self.loss(v); setattr(self,p,x)
            g[p]=(a-b)/(2*h)
        for p in self.PARAMS: setattr(self,p,max(-3,min(3,old[p]-lr*max(-4,min(4,g[p])))))
        if math.isfinite(self.loss(v)) and self.loss(v)<=base+1e-12: self.accepted+=1; return True
        for p,x in old.items(): setattr(self,p,x)
        self.rejected+=1; return False


class SeptillionAddressSpace:
    TOTAL=10**24; AXIS=10**8
    @classmethod
    def address(cls,index:int):
        n=int(index)%cls.TOTAL; a=[0,0,0]; place=[1,1,1]
        for level in range(24):
            axis=level%3; a[axis]+=(n%10)*place[axis]; place[axis]*=10; n//=10
        return tuple(a)
    @classmethod
    def normalized(cls,index:int):
        d=float(cls.AXIS-1); return tuple(2*v/d-1 for v in cls.address(index))


class VirtualSeptillionClock:
    def __init__(self,ops_per_second:int=OPS_PER_SECOND,now_ns:Callable[[],int]=time.monotonic_ns):
        if ops_per_second<=0: raise ValueError("ops_per_second must be positive")
        self.rate=int(ops_per_second); self.now_ns=now_ns; self.started=int(now_ns()); self.probes=0
    def virtual_ops(self): return max(0,int(self.now_ns())-self.started)*self.rate//10**9
    def sample(self):
        n=self.virtual_ops(); self.probes+=1; return n,SeptillionAddressSpace.address(n)


@dataclass
class ModalityMetrics:
    reconstruction_mse:float; cycle_mse:float; latent_energy:float; fusion_weight:float
    accepted_updates:int; rejected_updates:int

@dataclass
class LoopMetrics:
    cycle:int; virtual_ops:int; virtual_index:int; virtual_address:tuple[int,int,int]
    aggregate_reconstruction_mse:float; aggregate_cycle_mse:float; fixed_point_delta:float
    converged:bool; physical_probes:int; modalities:dict[str,ModalityMetrics]


class DrMoagiMultimodal3DLoop:
    def __init__(self,edge=8,temporal_depth=4,temporal_decay=.72,cross_modal_mix=.35,
                 coordinate_gain=.035,convergence_eps=1e-5,convergence_streak=3,
                 seed=0x4A415256495358,clock:Optional[VirtualSeptillionClock]=None):
        if edge<4 or edge%2: raise ValueError("edge must be even and >=4")
        if not 1<=temporal_depth<=64 or not 0<=temporal_decay<1 or not 0<=cross_modal_mix<=1: raise ValueError("invalid loop configuration")
        self.edge=edge; self.depth=temporal_depth; self.decay=temporal_decay; self.mix=cross_modal_mix
        self.coordinate_gain=coordinate_gain; self.eps=convergence_eps; self.streak_needed=convergence_streak
        self.clock=clock or VirtualSeptillionClock(); self.cycle=0; self.streak=0; self.prev: Optional[list[float]]=None
        self.models: dict[Modality,Autoencoder3D]={m:Autoencoder3D(edge,seed^(0x9E3779B97F4A7C15*(i+1))) for i,m in enumerate(MODALITIES)}
        self.inputs: dict[Modality,Volume3D]={}; self.lengths: dict[Modality,int]={}; self.history: dict[Modality,deque[Volume3D]]={m:deque(maxlen=temporal_depth) for m in MODALITIES}
        self.generated: dict[Modality,Volume3D]={}; self.fused: Optional[Volume3D]=None; self.last_metrics: Optional[LoopMetrics]=None
    def set_payload(self,modality:Modality|str,payload:bytes|str):
        m=modality if isinstance(modality,Modality) else Modality(modality); raw=payload.encode() if isinstance(payload,str) else bytes(payload)
        self.inputs[m]=Payload3DAdapter.ingest(raw,m,self.edge); self.lengths[m]=max(1,len(raw)); self.history[m].clear()
    def ensure_inputs(self):
        for m in MODALITIES:
            if m not in self.inputs: self.set_payload(m,f"DM-vOMEGA-XI+::{m.value}::3D recursive field")
    def temporal(self,m):
        h=self.history[m]; weights=[self.decay**i for i in range(len(h))]; total=sum(weights); o=[0.0]*len(h[0].values)
        for w,v in zip(weights,h):
            for i,x in enumerate(v.values): o[i]+=w*x/total
        return Volume3D(h[0].edge,o)
    @staticmethod
    def fuse(vols,weights):
        first=next(iter(vols.values())); out=[0.0]*len(first.values)
        for m,v in vols.items():
            first._same(v)
            for i,x in enumerate(v.values): out[i]+=weights[m]*x
        return Volume3D(first.edge,out)
    def condition(self,zv,index):
        cx,cy,cz=SeptillionAddressSpace.normalized(index); out=zv.copy(); e=zv.edge
        for i,v in enumerate(out.values):
            x=i%e; y=(i//e)%e; z=i//(e*e); phase=cx*(x+1)/e+cy*(y+1)/e+cz*(z+1)/e
            out.values[i]=math.tanh(v+self.coordinate_gain*math.sin(math.pi*phase))
        return out
    @staticmethod
    def signature(vols):
        out=[]
        for m in MODALITIES:
            v=vols[m].values; step=max(1,len(v)//16); out.extend(v[::step][:16])
        return out
    def step(self,virtual_ops:Optional[int]=None):
        self.ensure_inputs()
        if virtual_ops is None: virtual_ops,address=self.clock.sample()
        else: virtual_ops=int(virtual_ops); address=SeptillionAddressSpace.address(virtual_ops); self.clock.probes+=1
        index=virtual_ops%VIRTUAL_STATES
        for m in MODALITIES: self.models[m].train_transactional(self.inputs[m])
        temporal: dict[Modality,Volume3D]={}; losses: dict[Modality,float]={}
        for m in MODALITIES:
            model=self.models[m]; self.history[m].appendleft(model.encode(self.inputs[m])); temporal[m]=self.temporal(m)
            losses[m]=self.inputs[m].mse(model.decode(temporal[m]))
        raw={m:1/(1e-6+losses[m]) for m in MODALITIES}; s=sum(raw.values()); weights={m:raw[m]/s for m in MODALITIES}
        self.fused=self.condition(self.fuse(temporal,weights),index)
        generated: dict[Modality,Volume3D]={}; mm: dict[str,ModalityMetrics]={}; rs=cs=0.0
        for m in MODALITIES:
            model=self.models[m]; latent=temporal[m].blend(self.fused,self.mix); out=model.decode(latent); generated[m]=out
            cyc=latent.mse(model.encode(out)); rs+=losses[m]; cs+=cyc
            mm[m.value]=ModalityMetrics(losses[m],cyc,latent.energy(),weights[m],model.accepted,model.rejected)
        self.generated=generated; sig=self.signature(generated)
        delta=math.inf if self.prev is None else math.sqrt(sum((a-b)**2 for a,b in zip(sig,self.prev))/len(sig)); self.prev=sig
        self.streak=self.streak+1 if math.isfinite(delta) and delta<self.eps else 0; self.cycle+=1
        self.last_metrics=LoopMetrics(self.cycle,virtual_ops,index,address,rs/len(MODALITIES),cs/len(MODALITIES),delta,self.streak>=self.streak_needed,self.clock.probes,mm)
        return self.last_metrics
    def run(self,cycles:int,deterministic_virtual_stride:Optional[int]=None):
        if cycles<1: raise ValueError("cycles must be >=1")
        last=None
        for i in range(cycles): last=self.step(None if deterministic_virtual_stride is None else i*int(deterministic_virtual_stride))
        return last
    def generated_payload(self,modality:Modality|str):
        m=modality if isinstance(modality,Modality) else Modality(modality)
        if m not in self.generated: raise RuntimeError("run before generation")
        return Payload3DAdapter.emit(self.generated[m],self.lengths.get(m,self.edge**3))
    def export(self,directory:str|Path):
        if self.last_metrics is None or self.fused is None: raise RuntimeError("run before export")
        p=Path(directory); p.mkdir(parents=True,exist_ok=True)
        for m in MODALITIES: (p/f"generated-{m.value}.raw").write_bytes(self.generated_payload(m))
        (p/"metrics.json").write_text(json.dumps(asdict(self.last_metrics),indent=2,sort_keys=True)+"\n")
        lines=["# Jarvis-X fused 3D latent field"]; e=self.fused.edge
        for i,v in enumerate(self.fused.values): lines.append(f"v {i%e} {(i//e)%e} {i//(e*e)} {0.5+0.5*max(-1,min(1,v)):.6f}")
        (p/"fused-latent.obj").write_text("\n".join(lines)+"\n")


def parser():
    p=argparse.ArgumentParser(description="DM-vOmegaXi+ full 3D multimodal recursive loop")
    p.add_argument("--cycles",type=int,default=4); p.add_argument("--edge",type=int,default=8); p.add_argument("--temporal-depth",type=int,default=4)
    p.add_argument("--mix",type=float,default=.35); p.add_argument("--input",action="append",default=[],metavar="MODALITY=PATH")
    p.add_argument("--text"); p.add_argument("--out",default="./dm-multimodal3d-out"); p.add_argument("--deterministic-stride",type=int); p.add_argument("--quiet",action="store_true")
    return p


def main(argv:Optional[Sequence[str]]=None):
    a=parser().parse_args(argv); engine=DrMoagiMultimodal3DLoop(a.edge,a.temporal_depth,cross_modal_mix=a.mix)
    try:
        for spec in a.input:
            name,path=spec.split("=",1); engine.set_payload(Modality(name.lower()),Path(path).read_bytes())
        if a.text is not None: engine.set_payload(Modality.TEXT,a.text)
        m=engine.run(a.cycles,a.deterministic_stride); engine.export(a.out)
    except (OSError,ValueError,RuntimeError) as exc: raise SystemExit(f"error: {exc}") from exc
    if not a.quiet:
        d="inf" if not math.isfinite(m.fixed_point_delta) else f"{m.fixed_point_delta:.6g}"
        print(f"cycle={m.cycle} virtual={m.virtual_ops} index={m.virtual_index} address={m.virtual_address} recon={m.aggregate_reconstruction_mse:.6g} cycle_mse={m.aggregate_cycle_mse:.6g} delta={d} converged={m.converged} probes={m.physical_probes}")
        print(f"output={Path(a.out).resolve()}"); print("virtual-rate=1e24 states/s accounting only; physical work is the measured loop/probe count")
    return 0

if __name__=="__main__": raise SystemExit(main())