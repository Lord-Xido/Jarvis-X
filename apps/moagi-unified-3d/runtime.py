#!/usr/bin/env python3
"""Jarvis-X / Moagi Unified 3D Runtime.
Sparse virtual 1000^3 -> residual pyramid -> 64-fold fixed point -> decode -> Top-K correct -> recur.
"""
from __future__ import annotations
import argparse, hashlib, math, time
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class Config:
    world:int=1000; factor:int=10; levels:int=3; channels:int=4
    latent:int=64; folds:int=64; rho:float=.30; spectral:float=.82; beta:float=.88
    quantum:float=.02; lam:float=.35; tau0:float=.15; tau_decay:float=.60; topk:float=.12
    cycles:int=8; target_mse:float=1e-10; seed:int=7

@dataclass
class Field:
    xyz:np.ndarray; val:np.ndarray; size:int
    def __post_init__(self):
        self.xyz=np.asarray(self.xyz,np.int32); self.val=np.asarray(self.val,np.float32)
        if self.xyz.ndim!=2 or self.xyz.shape[1]!=3 or self.val.ndim!=2 or len(self.xyz)!=len(self.val):
            raise ValueError('Field shapes must be xyz=(N,3), val=(N,C)')
    @property
    def n(self): return len(self.xyz)
    @property
    def c(self): return self.val.shape[1]
    def map(self): return {tuple(map(int,p)):v for p,v in zip(self.xyz,self.val)}

@dataclass
class Pyramid:
    levels:list[Field]; residuals:list[Field]

@dataclass
class State:
    cycle:int; mse:float; psnr:float; threshold:float; selected:np.ndarray
    latent:np.ndarray; omega:np.ndarray; cube64:int; delta64:float; elapsed_ms:float
    converged:bool; done:bool

class Ingest:
    def __init__(self,c:Config): self.c=c
    def text(self,text:str)->Field:
        pts=[]; vals=[]
        for i,tok in enumerate(text.split()):
            d=hashlib.sha256(tok.encode()).digest(); lanes=[int.from_bytes(d[j:j+8],'little') for j in (0,8,16)]
            mix=(0x9E3779B185EBCA87,0xC2B2AE3D27D4EB4F,0x165667B19E3779F9)
            pts.append([(lanes[j]^(i*mix[j]))%self.c.world for j in range(3)])
            vals.append(np.resize(np.frombuffer(d,np.uint8).astype(np.float32)/127.5-1,self.c.channels))
        return Field(np.asarray(pts,np.int32),np.asarray(vals,np.float32),self.c.world)
    def synthetic(self,n:int)->Field:
        r=np.random.default_rng(self.c.seed)
        centers=np.array([[220,260,260],[720,330,670],[520,760,420],[500,500,500]],np.float32)
        xyz=centers[r.integers(0,len(centers),n)]+r.normal(0,75,(n,3))
        xyz=np.unique(np.clip(np.rint(xyz),0,self.c.world-1).astype(np.int32),axis=0)
        q=xyz.astype(np.float32)/self.c.world; x,y,z=q.T
        v=np.stack([np.sin(2*np.pi*x),np.cos(2*np.pi*y),np.sin(2*np.pi*z),np.cos(2*np.pi*(x+y+z))],1)
        return Field(xyz,np.resize(v,(len(xyz),self.c.channels)).astype(np.float32),self.c.world)

class Multiresolution:
    def __init__(self,c:Config): self.c=c
    def contract(self,f:Field)->Field:
        p=f.xyz//self.c.factor; u,inv=np.unique(p,axis=0,return_inverse=True)
        a=np.zeros((len(u),f.c),np.float64); count=np.zeros(len(u),np.int64)
        np.add.at(a,inv,f.val); np.add.at(count,inv,1); a/=count[:,None]
        return Field(u,a.astype(np.float32),max(1,f.size//self.c.factor))
    def encode(self,f:Field)->Pyramid:
        levels=[f]; residuals=[]; cur=f
        for _ in range(self.c.levels):
            parent=self.contract(cur); pm=parent.map(); pred=np.zeros_like(cur.val)
            for i,p in enumerate(cur.xyz): pred[i]=pm[tuple(map(int,p//self.c.factor))]
            res=cur.val-pred
            if self.c.quantum>0: res=np.round(res/self.c.quantum)*self.c.quantum
            residuals.append(Field(cur.xyz.copy(),res.astype(np.float32),cur.size)); levels.append(parent); cur=parent
        return Pyramid(levels,residuals)
    def decode(self,root:np.ndarray,residuals:list[Field],xyz:np.ndarray)->np.ndarray:
        out=np.repeat(np.asarray(root,np.float32)[None,:],len(xyz),0)
        for level in reversed(range(len(residuals))):
            m=residuals[level].map(); q=xyz//(self.c.factor**level)
            for i,p in enumerate(q):
                v=m.get(tuple(map(int,p)))
                if v is not None: out[i]+=v
        return out

class Core:
    def __init__(self,c:Config):
        self.c=c; r=np.random.default_rng(c.seed); d=c.latent
        W=r.normal(0,1/math.sqrt(d),(d,d)).astype(np.float32); s=np.linalg.svd(W,compute_uv=False)[0]
        self.W=W*(c.spectral/max(float(s),1e-9)); self.O=r.normal(0,.25/math.sqrt(d),(d,d)).astype(np.float32)
        self.b=r.normal(0,.01,d).astype(np.float32); self.omega=np.zeros(d,np.float32)
    def lift(self,root):
        i=np.arange(self.c.latent,dtype=np.float32)
        z=np.zeros(self.c.latent,np.float32)
        for j,v in enumerate(np.asarray(root).ravel()): z+=np.sin((j+1)*.173*i+v)+.5*np.cos((j+1)*.071*i-v)
        return np.tanh(z/max(1,1.5*len(root))).astype(np.float32)
    def fold(self,z):
        delta=0.
        for _ in range(self.c.folds):
            p=np.tanh(self.W@z+self.O@self.omega+self.b); zn=((1-self.c.rho)*z+self.c.rho*p).astype(np.float32)
            delta=float(np.linalg.norm(zn-z)); z=zn
        self.omega=(self.c.beta*self.omega+(1-self.c.beta)*z).astype(np.float32)
        return z,delta
    def root(self,z,channels): return np.array([np.mean(a) for a in np.array_split(z,channels)],np.float32)

class Engine:
    def __init__(self,c:Config):
        self.c=c; self.ingest=Ingest(c); self.mr=Multiresolution(c); self.core=Core(c)
        self.src=None; self.reco=None; self.cycle=0; self.last=None
    def _cube(self,z,res):
        rb=np.concatenate([r.val.ravel() for r in res]) if res else np.empty(0,np.float32)
        h=lambda a:int.from_bytes(hashlib.sha256(np.ascontiguousarray(a).tobytes()).digest()[:2],'little')
        n=0 if self.src is None else self.src.n; summary=min(4095,int(round(math.log2(max(n,1))*128)))
        fields=((0xA1,8),(0xF,4),(1,4),(self.c.levels,4),(summary,12),(h(z),16),(h(rb),16)); word=0
        for v,b in fields: word=(word<<b)|(v&((1<<b)-1))
        return word
    def _model(self,f:Field):
        p=self.mr.encode(f); root=np.mean(p.levels[-1].val,axis=0); z,delta=self.core.fold(self.core.lift(root))
        pred=self.mr.decode(self.core.root(z,f.c),p.residuals,f.xyz); return p,z,delta,pred,self._cube(z,p.residuals)
    def _select(self,e,tau):
        score=np.linalg.norm(e,axis=1); eligible=np.flatnonzero(score>tau)
        if not len(eligible): return eligible
        k=min(len(eligible),max(1,math.ceil(self.c.topk*len(eligible)))); part=np.argpartition(score[eligible],-k)[-k:]
        return eligible[part[np.argsort(score[eligible][part])[::-1]]]
    def _state(self,z,delta,cube,ms):
        e=self.src.val-self.reco; mse=float(np.mean(np.square(e,dtype=np.float64))); psnr=float('inf') if mse<=0 else 10*math.log10(4/mse)
        tau=self.c.tau0*self.c.tau_decay**self.cycle; sel=self._select(e,tau); conv=mse<=self.c.target_mse or len(sel)==0; done=conv or self.cycle>=self.c.cycles
        self.last=State(self.cycle,mse,psnr,tau,sel,z.copy(),self.core.omega.copy(),cube,delta,ms,conv,done); return self.last
    def load(self,f:Field):
        self.src=f; self.cycle=0; self.core.omega[:]=0; t=time.perf_counter(); _,z,d,p,cube=self._model(f); self.reco=p.astype(np.float32)
        return self._state(z,d,cube,(time.perf_counter()-t)*1000)
    def step(self):
        if self.last.done:return self.last
        t=time.perf_counter(); e=self.src.val-self.reco; sel=self._select(e,self.c.tau0*self.c.tau_decay**self.cycle)
        if len(sel): self.reco[sel]+=self.c.lam*e[sel]
        recurrent=Field(self.src.xyz.copy(),self.reco.copy(),self.src.size); _,z,d,p,cube=self._model(recurrent)
        self.reco=(.65*self.reco+.35*p).astype(np.float32); self.cycle+=1
        return self._state(z,d,cube,(time.perf_counter()-t)*1000)
    def run(self):
        while not self.last.done: self.step()
        return self.last

class Viewer:
    def __init__(self,e:Engine,f:Field):
        import tkinter as tk
        self.tk=tk; self.e=e; self.f=f; self.s=e.load(f); self.yaw=.65; self.pitch=-.4; self.running=False
        self.root=tk.Tk(); self.root.title('Jarvis-X — Moagi Unified 3D Runtime'); self.root.geometry('1280x820')
        self.cv=tk.Canvas(self.root,bg='#020617',highlightthickness=0); self.cv.pack(fill='both',expand=True)
        bar=tk.Frame(self.root,bg='#0b1220'); bar.place(x=15,y=15)
        self.lbl=tk.Label(bar,bg='#0b1220',fg='#e5f3ff',justify='left',font=('Courier',10)); self.lbl.pack(padx=12,pady=8)
        tk.Button(bar,text='RUN',command=self.toggle).pack(side='left'); tk.Button(bar,text='STEP',command=self.one).pack(side='left')
        self.cv.bind('<B1-Motion>',self.drag); self.cv.bind('<ButtonPress-1>',lambda ev:setattr(self,'last',(ev.x,ev.y))); self.cv.bind('<Configure>',lambda ev:self.draw()); self.last=None; self.draw()
    def proj(self,p):
        w=max(1,self.cv.winfo_width()); h=max(1,self.cv.winfo_height()); cy,sy=math.cos(self.yaw),math.sin(self.yaw); cp,sp=math.cos(self.pitch),math.sin(self.pitch)
        R=np.array([[cy,0,sy],[sp*sy,cp,-sp*cy],[-cp*sy,sp,cp*cy]],np.float32); q=(p-.5)@R.T; d=np.maximum(q[:,2]+2.7,.1); sc=min(w,h)*.78
        return np.c_[w/2+sc*q[:,0]/d,h/2-sc*q[:,1]/d]
    def drag(self,ev):
        if self.last:self.yaw+=(ev.x-self.last[0])*.008; self.pitch=np.clip(self.pitch+(ev.y-self.last[1])*.008,-1.4,1.4)
        self.last=(ev.x,ev.y); self.draw()
    def one(self):
        if not self.s.done:self.s=self.e.step(); self.draw()
    def toggle(self):self.running=not self.running; self.loop()
    def loop(self):
        if not self.running or self.s.done:return
        self.s=self.e.step(); self.yaw+=.04; self.draw(); self.root.after(180,self.loop)
    def draw(self):
        self.cv.delete('all'); V=np.array([[x,y,z] for z in (0,1) for y in (0,1) for x in (0,1)],np.float32); P=self.proj(V)
        edges=((0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7))
        for a,b in edges:self.cv.create_line(*P[a],*P[b],fill='#155e75',width=2)
        n=min(4000,self.f.n); idx=np.argpartition(np.linalg.norm(self.f.val-self.e.reco,axis=1),-n)[-n:] if self.f.n>n else np.arange(self.f.n)
        Q=self.proj(self.f.xyz[idx]/(self.e.c.world-1)); selected=set(map(int,self.s.selected))
        for j,i in enumerate(idx):
            col='#ff3344' if int(i) in selected else '#f59e0b'; x,y=Q[j]; self.cv.create_oval(x-2,y-2,x+2,y+2,fill=col,outline='')
        c=self.proj(np.array([[.5,.5,.5]],np.float32))[0]; self.cv.create_oval(c[0]-9,c[1]-9,c[0]+9,c[1]+9,fill='#a78bfa',outline='')
        ps='inf' if not math.isfinite(self.s.psnr) else f'{self.s.psnr:.2f}'
        self.lbl.config(text=f'virtual {self.e.c.world}³ | active {self.f.n:,}\ncycle {self.s.cycle}/{self.e.c.cycles} | selected {len(self.s.selected):,}\nMSE {self.s.mse:.3e} | PSNR {ps} dB\nCube64 0x{self.s.cube64:016X}\nΩ {np.linalg.norm(self.s.omega):.4f} | Δ64 {self.s.delta64:.2e}')
    def run(self):self.root.mainloop()

def main():
    a=argparse.ArgumentParser(); a.add_argument('--headless',action='store_true'); a.add_argument('--points',type=int,default=10000); a.add_argument('--cycles',type=int,default=8); a.add_argument('--text'); a.add_argument('--residual-quantum',type=float,default=.02); ns=a.parse_args()
    c=Config(cycles=max(1,ns.cycles),quantum=max(0,ns.residual_quantum)); e=Engine(c); f=e.ingest.text(ns.text) if ns.text else e.ingest.synthetic(max(1,ns.points)); s=e.load(f)
    if ns.headless:
        print(f'virtual={c.world}^3 active={f.n:,} materialized=no')
        while True:
            ps='inf' if not math.isfinite(s.psnr) else f'{s.psnr:.3f}'; print(f'cycle={s.cycle:02d} selected={len(s.selected):5d} MSE={s.mse:.6e} PSNR={ps}dB Cube64=0x{s.cube64:016X}')
            if s.done:break
            s=e.step()
    else: Viewer(e,f).run()
if __name__=='__main__': main()
