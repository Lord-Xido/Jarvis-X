#!/usr/bin/env python3
"""V4-compatible codec subset required by the isolated DM3D v5 experiment.

Functions and parameters copied from the local v4 reference. This module is
not the complete original v4 CLI, benchmark engine or archival format.
"""
from __future__ import annotations
import hashlib, json, math, struct, zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import numpy as np
from PIL import Image, ImageFilter
from reportlab.lib.colors import HexColor

VERSION = '4.0.0'
GENE_SHAPE = (8,8,12)
BASELINE = (7,0,0)
FRACTIONS = (0.0, .0625, .125, .25, .375, .5, .75, 1.0)
QUANT = (1,2,3,4,6,8,12,16)
OFFSETS = ((0,0,0),(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,1,1),(2,0,0),(0,2,0),(0,0,2),(2,2,2))
TEAL=HexColor('#40d6f2'); MINT=HexColor('#61e5b8')


def sha(data: bytes) -> str: return hashlib.sha256(data).hexdigest()

def canonical(o): return json.dumps(o,sort_keys=True,separators=(',',':')).encode('utf-8')

def sample_volume(image_path: Path, n=24) -> np.ndarray:
    if n % 2 or n<8 or n>96: raise ValueError('n must be even and 8..96')
    if image_path and image_path.is_file():
        im=Image.open(image_path).convert('L').resize((n,n),Image.Resampling.BICUBIC)
        im=im.filter(ImageFilter.GaussianBlur(radius=.85))
        a=np.asarray(im,dtype=np.float32)/255.
    else:
        u=np.linspace(0,2*np.pi,n,endpoint=False,dtype=np.float32)
        x,y=np.meshgrid(u,u,indexing='ij')
        a=.5+.25*np.cos(x)+.20*np.sin(y)
    u=np.arange(n,dtype=np.float32)/n * (2*np.pi)
    x,y,z=np.meshgrid(u,u,u,indexing='ij')
    radial=np.sqrt((np.sqrt((x-np.pi)**2+(y-np.pi)**2)-1.2)**2+(z-np.pi)**2)
    torus=np.exp(-4*radial**2)
    slices=np.empty((n,n,n),dtype=np.float32)
    for k in range(n):
        slices[:,:,k]=np.roll(a,(k//5,k//7),(0,1))*(.46+.14*np.cos(u[k]))
    signal=np.clip(slices + .31*torus + .12*(1+np.sin(2*x+y+z))/2,0,1)
    return np.rint(signal*255).astype(np.uint8)

def neighbors_3d(v: np.ndarray):
    return sum((np.roll(v,1,i)+np.roll(v,-1,i) for i in range(3)))/6.

def inward_field(state: np.ndarray, target: np.ndarray, steps=4, memory=.57, diffusion=.13):
    """Contraction: Lipschitz in infinity norm <=memory+diffusion<1."""
    if memory<0 or diffusion<0 or memory+diffusion>=1: raise ValueError('not contractive')
    f=np.asarray(state,dtype=np.float32).copy()
    target=np.asarray(target,dtype=np.float32)
    changes=[]
    for _ in range(steps):
        nxt=memory*f+diffusion*neighbors_3d(f)+(1-memory-diffusion)*target
        changes.append(float(np.max(np.abs(nxt-f))))
        f=nxt
    if not np.isfinite(f).all():raise ArithmeticError('Non-finite volume state')
    return f,changes

@dataclass
class Encoded:
    raw: bytes
    coarse: np.ndarray
    positions: np.ndarray
    corrections: np.ndarray
    gene: Tuple[int,int,int]

def encode3d(volume: np.ndarray, gene: Tuple[int,int,int]):
    v=np.asarray(volume)
    if v.ndim!=3 or len(set(v.shape))!=1 or v.shape[0]%2 or v.dtype!=np.uint8:
        raise ValueError('uint8 even cubic volume expected')
    n=v.shape[0]
    if any(not 0<=g<d for g,d in zip(gene,GENE_SHAPE)):raise ValueError('invalid genotype')
    roll=OFFSETS[gene[2]]
    shifted=np.roll(v,roll,axis=(0,1,2))
    coarse=np.rint(shifted.reshape(n//2,2,n//2,2,n//2,2).mean(axis=(1,3,5))).astype(np.uint8)
    approx=coarse.repeat(2,0).repeat(2,1).repeat(2,2)
    residual=(shifted.astype(np.int16)-approx.astype(np.int16)).ravel()
    keep=round(FRACTIONS[gene[0]]*residual.size)
    if keep==0:
        positions=np.empty(0,dtype=np.uint16)
    elif keep==residual.size:
        positions=np.arange(residual.size,dtype=np.uint16)
    else:
        positions=np.sort(np.argpartition(np.abs(residual),-keep)[-keep:]).astype(np.uint16)
    q=QUANT[gene[1]]
    corr=np.rint(residual[positions]/q).astype(np.int16)*q
    meta={'version':VERSION,'n':n,'gene':list(gene),'count':len(positions),'dtype':'u1,i2,i2'}
    head=canonical(meta)
    content=struct.pack('<I',len(head))+head+coarse.tobytes()+positions.tobytes()+corr.astype('<i2').tobytes()
    return Encoded(zlib.compress(content,6),coarse,positions,corr,gene)

def decode3d(blob: bytes) -> np.ndarray:
    data=zlib.decompress(blob)
    if len(data)<4:raise ValueError('truncated payload')
    mlen=struct.unpack('<I',data[:4])[0]
    if mlen>4096 or len(data)<4+mlen: raise ValueError('invalid header length')
    m=json.loads(data[4:4+mlen]);n=int(m['n']);gene=tuple(m['gene']);k=int(m['count'])
    if n%2 or n<8 or n>96 or k<0 or k>n**3 or n**3>65535 or len(gene)!=3:raise ValueError('invalid dimensions')
    if any(not 0<=g<d for g,d in zip(gene,GENE_SHAPE)):raise ValueError('invalid gene')
    cs=(n//2)**3;off=4+mlen;expected=off+cs+4*k
    if len(data)!=expected:raise ValueError('payload size mismatch')
    coarse=np.frombuffer(data,dtype=np.uint8,count=cs,offset=off).reshape((n//2,)*3)
    off+=cs
    positions=np.frombuffer(data,dtype='<u2',count=k,offset=off).astype(np.int64)
    off+=2*k
    vals=np.frombuffer(data,dtype='<i2',count=k,offset=off)
    if k and (int(positions.max())>=n**3 or len(np.unique(positions))!=k):raise ValueError('invalid residual indices')
    a=coarse.repeat(2,0).repeat(2,1).repeat(2,2).astype(np.int16)
    a.reshape(-1)[positions]+=vals
    restored=np.roll(np.clip(a,0,255).astype(np.uint8),tuple(-s for s in OFFSETS[gene[2]]),(0,1,2))
    return restored

def distortion(original,restored):
    delta=original.astype(np.float32)-restored.astype(np.float32)
    mse=float(np.mean(delta**2))
    return mse, float('inf') if mse==0 else float(10*math.log10((255**2)/mse)),int(np.max(np.abs(delta)))

def canonical_gene(g):
    """When no residuals survive, quantization is provably irrelevant."""
    return (g[0],0,g[2]) if g[0]==0 else tuple(g)

def iso_points(c, volume, x0=250,y0=160,scale=7.4,limit=200):
    """Projected X/Y/Z points from actual 3D ndarray; perspective drawing is diagrammatic."""
    n=volume.shape[0]
    coords=np.argwhere(volume>np.percentile(volume,93))
    if len(coords)>limit:coords=coords[np.linspace(0,len(coords)-1,limit,dtype=int)]
    pts=[]
    for x,y,z in coords:
        xx=x0+(x-y)*scale*.68
        yy=y0+(x+y)*scale*.31+z*scale*.69
        pts.append((x+y+z,xx,yy,float(volume[x,y,z])/255))
    for _,xx,yy,v in sorted(pts):
        c.setFillColor(HexColor('#52b8ed') if v<.68 else MINT)
        c.circle(xx,yy,1.9,stroke=0,fill=1)
    def proj(x,y,z):return x0+(x-y)*scale*.68, y0+(x+y)*scale*.31+z*scale*.69
    V=[proj(a,b,d) for a in (0,n) for b in (0,n) for d in (0,n)]
    c.setStrokeColor(TEAL);c.setLineWidth(.65)
    for i,p in enumerate(V):
        for j,q in enumerate(V):
            if i<j and bin(i^j).count('1')==1:c.line(*p,*q)
