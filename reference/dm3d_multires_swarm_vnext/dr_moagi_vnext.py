#!/usr/bin/env python3
"""DM3D vNext: bounded, reproducible 3D multimodal codec + ANN swarm.

Capabilities: symbolic virtual coordinates, reversible 3D multilevel Haar basis,
9 geometric ANN agents, analytic train gradients, transactional train/validation
CTR, encoder-only inward ridge correction, independently decodable packets,
progressive sparse detail planes, integrity digest, deterministic rate/quality
search, and safe finite resource caps.

This is an experimental numerical codec, not general intelligence, a global
astronomical simulation, or a proof of unseen-input safety.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
import numpy as np

MODALITIES = ("text", "image", "audio", "video")
AGENTS = ("identity", "x+", "x-", "y+", "y-", "z+", "z-", "periodic_diffusion", "inward")
AXIS_EXPONENT = 6_000_000
VOLUME_EXPONENT = 18_000_000
SYMBOLIC_ITERATIONS = "10^(24*10^24)"  # (septillion)^(septillion), virtual only
PACKET_VERSION = 1


def tile_address(i: int) -> dict:
    if not 0 <= i < 1_000_000:
        raise ValueError("bounded tile index required")
    return {"x_prefix": f"{i:06d}", "y_prefix": f"{(17*i+211)%1_000_000:06d}",
            "z_prefix": f"{(37*i+877)%1_000_000:06d}",
            "free_decimal_digits_per_axis": AXIS_EXPONENT-6}


def hadamard_8() -> np.ndarray:
    # Exactly orthonormal in real arithmetic; floating-point inversion is near exact.
    i = np.arange(8, dtype=np.uint8)
    v = np.empty((8,8), dtype=np.float64)
    for a in range(8):
        for b in range(8):
            v[a,b] = (1 if ((a & b).bit_count() % 2)==0 else -1) / math.sqrt(8)
    return v

H8 = hadamard_8()


def split_level(x: np.ndarray):
    if x.ndim != 5 or len(set(x.shape[1:4])) != 1 or x.shape[1] % 2:
        raise ValueError("expected (B, even N, even N, even N, C)")
    b,n,_,_,c = x.shape
    k=n//2
    samples = x.reshape(b,k,2,k,2,k,2,c).transpose(0,1,3,5,2,4,6,7).reshape(b,k,k,k,8,c)
    coefficients = np.einsum('...ic,ij->...jc', samples, H8, optimize=True)
    return coefficients[...,0,:].copy(), coefficients[...,1:,:].copy()


def join_level(coarse: np.ndarray, details: np.ndarray):
    b,k,_,_,c = coarse.shape
    if details.shape != (b,k,k,k,7,c):
        raise ValueError("invalid 3D detail shape")
    coeff = np.concatenate([coarse[...,None,:],details],axis=-2)
    samples = np.einsum('...jc,ij->...ic', coeff, H8, optimize=True)
    return samples.reshape(b,k,k,k,2,2,2,c).transpose(0,1,4,2,5,3,6,7).reshape(b,2*k,2*k,2*k,c)


def pyramid(x: np.ndarray, levels: int):
    if levels < 1 or x.shape[1] % (2**levels) or x.shape[1]//(2**levels) < 2:
        raise ValueError("levels require a divisible edge and coarsest edge >= 2")
    c=x
    details=[]
    for _ in range(levels):
        c,d=split_level(c)
        details.append(d)
    return c, details


def inverse_pyramid(coarse: np.ndarray, details: list[np.ndarray]):
    x=coarse
    for d in reversed(details):
        x=join_level(x,d)
    return x


def softmax(q):
    e=np.exp(q-np.max(q))
    return e/e.sum()


class Geometry:
    def __init__(self,n, inward_ratio=0.65):
        if n < 2 or not (0 < inward_ratio <= 1):
            raise ValueError("bad latent geometry")
        mesh=np.indices((n,n,n), dtype=np.float64)
        center=(n-1)/2
        sample=np.rint(center+inward_ratio*(mesh-center)).astype(int)%n
        self.idx=np.ravel_multi_index(tuple(sample),(n,n,n)).ravel()
    @staticmethod
    def diffusion(x):
        out=x.copy()
        for axis in (1,2,3):
            out += np.roll(x,1,axis) + np.roll(x,-1,axis)
        return out/7
    def apply(self,j,x):
        if j==0: return x
        if j in (1,2,3,4,5,6):
            return np.roll(x,1 if j%2 else -1,axis=(j-1)//2+1)
        if j==7: return self.diffusion(x)
        if j==8:
            b,_,_,_,c=x.shape
            return x.reshape(b,-1,c)[:,self.idx,:].reshape(x.shape)
        raise ValueError(j)
    def adjoint(self,j,x):
        if j==0: return x
        if j in (1,2,3,4,5,6):
            return np.roll(x,-1 if j%2 else 1,axis=(j-1)//2+1)
        if j==7: return self.diffusion(x)
        if j==8:
            b,_,_,_,c=x.shape
            flat=x.reshape(b,-1,c)
            out=np.zeros_like(flat)
            for k in range(b):
                np.add.at(out[k],self.idx,flat[k])
            return out.reshape(x.shape)
        raise ValueError(j)


@dataclass
class Config:
    edge:int=8
    tiles:int=8
    levels:int=2
    hidden:int=12
    train_steps:int=80
    lr:float=0.8
    memory_gain:float=0.04
    memory_decay:float=0.98
    latent_steps:int=12
    latent_mix:float=0.5
    ridge:float=0.001
    seed:int=7
    max_voxels:int=500_000
    def validate(self):
        if not (4<=self.edge<=64 and 2<=self.tiles<=128 and 1<=self.levels<=5 and
                self.edge%(2**self.levels)==0 and self.edge//(2**self.levels)>=2):
            raise ValueError("bounded edge/tiles/levels required; coarsest edge >= 2")
        if self.tiles*self.edge**3 > self.max_voxels:
            raise ValueError("materialization exceeds max_voxels")
        if not 1<=self.hidden<=256 or not 0<=self.train_steps<=20000 or not 0<=self.latent_steps<=500:
            raise ValueError("invalid hidden/iteration budget")
        if not (0 < self.lr <= 5 and 0<=self.memory_gain<=1 and 0<=self.memory_decay<1 and
                0<self.latent_mix<=1 and self.ridge>0):
            raise ValueError("invalid optimizer settings")


class Swarm:
    def __init__(self, cfg:Config, channels=4):
        cfg.validate()
        self.cfg=cfg
        self.channels=channels
        self.geometry=Geometry(cfg.edge//2**cfg.levels)
        rng=np.random.default_rng(cfg.seed)
        self.We=rng.normal(0,.35,(channels,cfg.hidden))
        self.be=np.zeros(cfg.hidden)
        self.Wd=rng.normal(0,.35,(cfg.hidden,channels))
        self.bd=np.zeros(channels)
        self.q=np.array([3.6,-.8,-.8,-.8,-.8,-.8,-.8,.5,-.8],dtype=float)
        self.omega=np.zeros(cfg.hidden)
        self.accepted=0
        self.rejected=0
        self.history=[]
    def params(self):
        return {k:getattr(self,k) for k in ('We','be','Wd','bd','q','omega')}
    def forward(self,c,cache=False):
        h=np.tanh(c@self.We+self.be+self.cfg.memory_gain*self.omega)
        a=softmax(self.q)
        fields=[self.geometry.apply(j,h) for j in range(9)]
        z=sum(a[j]*fields[j] for j in range(9))
        output=z@self.Wd+self.bd
        if cache: return output,z,(h,a,fields)
        return output,z
    def mse(self,c):
        y,_=self.forward(c)
        return float(np.mean((y-c)**2))
    def gradients(self,c):
        y,z,(h,a,fields)=self.forward(c,True)
        dy=2*(y-c)/c.size
        dw=z.reshape(-1,self.cfg.hidden).T@dy.reshape(-1,self.channels)
        db=dy.sum(axis=(0,1,2,3))
        dz=dy@self.Wd.T
        dh=np.zeros_like(h)
        dq=np.zeros(9)
        for j in range(9):
            dq[j]=a[j]*np.sum(dz*(fields[j]-z))
            dh += a[j]*self.geometry.adjoint(j,dz)
        dpre=dh*(1-h*h)
        dwe=c.reshape(-1,self.channels).T@dpre.reshape(-1,self.cfg.hidden)
        dbe=dpre.sum(axis=(0,1,2,3))
        return {'We':dwe, 'be':dbe, 'Wd':dw, 'bd':db, 'q':dq}
    def train(self,c_train,c_val):
        # Both datasets are fixed across the run; only bounded/verified steps are accepted.
        train0=self.mse(c_train); val0=self.mse(c_val)
        for step in range(1,self.cfg.train_steps+1):
            before_tr=self.mse(c_train)
            before_val=self.mse(c_val)
            old={k:v.copy() for k,v in self.params().items()}
            grad=self.gradients(c_train)
            norm=math.sqrt(sum(float((g*g).sum()) for g in grad.values()))
            scale=min(1.,1./max(norm,1e-12))
            _,_,(h,_,_)=self.forward(c_train,True)
            proposed_memory=self.cfg.memory_decay*old['omega']+(1-self.cfg.memory_decay)*h.mean(axis=(0,1,2,3))
            ok=False
            for backtrack in range(12):
                eta=self.cfg.lr*scale*(.5**backtrack)
                for name,g in grad.items():
                    getattr(self,name)[...]=old[name]-eta*g
                self.omega[...]=proposed_memory
                tr=self.mse(c_train)
                val=self.mse(c_val)
                if (np.isfinite(tr) and np.isfinite(val) and
                        tr<=before_tr+1e-12 and val<=before_val+1e-12):
                    ok=True
                    break
            if not ok:
                for name,a in old.items():
                    getattr(self,name)[...]=a
                self.rejected+=1
            else:
                self.accepted+=1
            self.history.append({'step':step,'train_mse':self.mse(c_train),
                                 'validation_mse':self.mse(c_val),'accepted':ok})
        return {'train_initial':train0,'validation_initial':val0,
                'train_final':self.mse(c_train),'validation_final':self.mse(c_val),
                'accepted':self.accepted,'rejected':self.rejected}
    def checkpoint_bytes(self):
        """Serialize mutable weights, routing, optimizer telemetry and memory."""
        cfg_json=json.dumps(self.cfg.__dict__,sort_keys=True)
        arrays={'configuration':np.array(cfg_json),
                'accepted':np.array(self.accepted), 'rejected':np.array(self.rejected)}
        arrays.update({k:v.copy() for k,v in self.params().items()})
        with io.BytesIO() as buf:
            np.savez_compressed(buf,**arrays)
            return buf.getvalue()

    @classmethod
    def from_checkpoint(cls,packet:bytes):
        with np.load(io.BytesIO(packet),allow_pickle=False) as d:
            cfg=Config(**json.loads(str(d['configuration'].item())))
            model=cls(cfg)
            for k,dst in model.params().items():
                src=d[k]
                if src.shape!=dst.shape or not np.isfinite(src).all():
                    raise ValueError('checkpoint parameter corruption')
                dst[...]=src
            model.accepted=int(d['accepted'].item())
            model.rejected=int(d['rejected'].item())
            if model.accepted<0 or model.rejected<0:raise ValueError('invalid counters')
            return model

    def encode_coarse(self,c):
        _,z=self.forward(c)
        # Encoder has access to c; decoder only receives z and Wd/bd.
        matrix=self.Wd@self.Wd.T + self.cfg.ridge*np.eye(self.cfg.hidden)
        rhs=(c-self.bd)@self.Wd.T+self.cfg.ridge*z
        target=np.linalg.solve(matrix,rhs.reshape(-1,self.cfg.hidden).T).T.reshape(z.shape)
        prior_loss=float(np.mean((c-(z@self.Wd+self.bd))**2))
        steps=0
        for _ in range(self.cfg.latent_steps):
            candidate=z+self.cfg.latent_mix*(target-z)
            loss=float(np.mean((c-(candidate@self.Wd+self.bd))**2))
            if not np.isfinite(loss) or loss > prior_loss+1e-12:
                break
            z=candidate;prior_loss=loss;steps+=1
            if float(np.max(np.abs(target-z))) < 1e-8: break
        return z,steps


def digest_arrays(coef, weights, biases, idxs, vals, correction):
    h=hashlib.sha256()
    for a in (coef,weights,biases,*idxs,*vals):
        ar=np.ascontiguousarray(a)
        h.update(str(ar.shape).encode());h.update(str(ar.dtype).encode());h.update(ar.tobytes())
    if correction is not None:
        ar=np.ascontiguousarray(correction)
        h.update(str(ar.shape).encode());h.update(ar.tobytes())
    return h.hexdigest()


def encode_packet(x, model:Swarm, keep:float, include_base_correction=False) -> bytes:
    if not 0<=keep<=1: raise ValueError("detail retention fraction must be [0,1]")
    c,details=pyramid(x,model.cfg.levels)
    z,_=model.encode_coarse(c)
    decoded_coarse=z@model.Wd+model.bd
    correction=c-decoded_coarse if include_base_correction else None
    flattened=np.concatenate([d.reshape(-1) for d in details])
    counts=[d.size for d in details]
    k=min(len(flattened),math.ceil(keep*len(flattened)))
    chosen=np.zeros(len(flattened),dtype=bool)
    if k:
        # Stable, deterministic top-k by magnitude, then array index.
        order=np.lexsort((np.arange(len(flattened)),-np.abs(flattened)))
        chosen[order[:k]]=True
    indices=[];values=[];offset=0
    for d,n in zip(details,counts):
        local_idx=np.flatnonzero(chosen[offset:offset+n]).astype(np.int32)
        indices.append(local_idx)
        values.append(d.reshape(-1)[local_idx].copy())
        offset+=n
    meta={'version':PACKET_VERSION,'edge':model.cfg.edge,'tiles':int(x.shape[0]),
          'levels':model.cfg.levels,'channels':int(x.shape[-1]),'hidden':model.cfg.hidden,
          'detail_keep':keep,'base_correction':bool(include_base_correction),
          'detail_shapes':[list(d.shape) for d in details],
          'symbolic_volume_exponent':VOLUME_EXPONENT,
          'addresses':[tile_address(i) for i in range(len(x))]}
    checksum=digest_arrays(z,model.Wd,model.bd,indices,values,correction)
    meta['sha256']=checksum
    arrays={'meta':np.array(json.dumps(meta,sort_keys=True)),
            'Z':np.asarray(z,dtype=np.float64),'Wd':model.Wd.copy(),'bd':model.bd.copy()}
    for level,(ind,val) in enumerate(zip(indices,values)):
        arrays[f'index_{level}']=ind
        arrays[f'value_{level}']=val
    if correction is not None: arrays['base_correction']=correction
    with io.BytesIO() as buf:
        np.savez_compressed(buf,**arrays)
        return buf.getvalue()


def decode_packet(packet:bytes, max_voxels=500_000):
    if len(packet)>256_000_000: raise ValueError("packet exceeds 256 MB cap")
    # Guard against compressed ZIP bombs before NumPy decompresses an array.
    try:
        with zipfile.ZipFile(io.BytesIO(packet)) as archive:
            members=archive.infolist()
            if len(members)>40 or sum(m.file_size for m in members)>128_000_000:
                raise ValueError("packet uncompressed size exceeds safety cap")
    except zipfile.BadZipFile as exc:
        raise ValueError('invalid ZIP packet') from exc
    try:
        with np.load(io.BytesIO(packet),allow_pickle=False) as data:
            meta=json.loads(str(data['meta'].item()))
            if meta['version']!=PACKET_VERSION: raise ValueError("unknown packet version")
            edge=int(meta['edge']);b=int(meta['tiles']);channels=int(meta['channels']);levels=int(meta['levels'])
            h=int(meta['hidden'])
            if not (1<=b<=128 and 4<=edge<=64 and 1<=channels<=16 and 1<=h<=256 and
                    1<=levels<=5 and edge%(2**levels)==0 and edge//(2**levels)>=2 and
                    b*edge**3 <= max_voxels):
                raise ValueError("invalid or oversized decoded tensor")
            n=edge//2**levels
            z=data['Z'].copy(); wd=data['Wd'].copy(); bd=data['bd'].copy()
            if z.shape!=(b,n,n,n,h) or wd.shape!=(h,channels) or bd.shape!=(channels,):
                raise ValueError("invalid decoder geometry")
            details=[];indices=[];values=[]
            for level in range(levels):
                k=edge//(2**(level+1))
                shape=(b,k,k,k,7,channels)
                if tuple(meta['detail_shapes'][level]) != shape:
                    raise ValueError("invalid sparse detail shape")
                ind=data[f'index_{level}'].copy();val=data[f'value_{level}'].copy()
                if ind.ndim!=1 or val.shape!=ind.shape or np.any(ind<0) or np.any(ind>=np.prod(shape)):
                    raise ValueError("invalid sparse indices")
                if len(np.unique(ind))!=len(ind): raise ValueError("duplicate sparse indices")
                full=np.zeros(math.prod(shape),dtype=np.float64)
                full[ind]=val
                details.append(full.reshape(shape))
                indices.append(ind);values.append(val)
            correction=data['base_correction'].copy() if meta['base_correction'] else None
            if correction is not None and correction.shape!=(b,n,n,n,channels):
                raise ValueError("bad base correction")
            if digest_arrays(z,wd,bd,indices,values,correction)!=meta['sha256']:
                raise ValueError("packet integrity check failed")
            if not np.isfinite(z).all() or not np.isfinite(wd).all() or not np.isfinite(bd).all() or any(not np.isfinite(v).all() for v in values):
                raise ValueError("nonfinite packet content")
            coarse=z@wd+bd
            if correction is not None:
                if not np.isfinite(correction).all(): raise ValueError("nonfinite correction")
                coarse=coarse+correction
            output=inverse_pyramid(coarse,details)
            return output,meta
    except (KeyError,IndexError,TypeError) as exc:
        raise ValueError("invalid codec packet") from exc


def candidate_search(model,validation_x,target_mse:float,keep_rates=(0.,.125,.25,.5,.75,1.)):
    if target_mse<0: raise ValueError("target_mse must be >= 0")
    rows=[];chosen=None
    for retain in keep_rates:
        for correct in (False,True):
            blob=encode_packet(validation_x,model,float(retain),correct)
            y,_=decode_packet(blob,max_voxels=model.cfg.max_voxels)
            mse=float(np.mean((validation_x-y)**2))
            row={'keep':float(retain),'base_correction':correct,
                 'mse':mse,'bytes':len(blob),'meets_target':bool(mse<=target_mse)}
            rows.append(row)
    feasible=[r for r in rows if r['meets_target']]
    if feasible:
        chosen=min(feasible,key=lambda r:(r['bytes'],r['mse']))
    else:
        chosen=min(rows,key=lambda r:(r['mse'],r['bytes']))
    return chosen,rows


def demo_data(cfg):
    # Reproducible heterogeneous fields, never claimed as real text/audio/video.
    v=np.indices((cfg.edge,cfg.edge,cfg.edge),dtype=float)
    z,y,x=v/max(1,cfg.edge-1)
    base=np.stack([
        np.where((((v[1]//2+v[2]//2)%3)==0)&(v[0]%4<2),.8,-.2),
        np.exp(-((x-.35)**2+(y-.63)**2)/.09)*1.7-.6,
        np.sin(2*math.pi*(2*x+1.1*y+.6*z)),
        2*np.exp(-((x-(.15+.7*z))**2+(y-.5)**2)/.06)-.6],axis=-1)
    return np.stack([np.roll(base,(i%3,i//3%3,i//5%3),(0,1,2)) for i in range(cfg.tiles)])


def save_atomic(filename:Path,data:bytes):
    filename.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=filename.parent,delete=False) as f:
        tmp=Path(f.name)
        f.write(data)
    try: os.replace(tmp,filename)
    finally:
        if tmp.exists(): tmp.unlink()


def run(cfg:Config,out:Path,target_mse=.015,inputs=None,model=None):
    cfg.validate()
    x=demo_data(cfg) if inputs is None else np.asarray(inputs,dtype=np.float64)
    if x.shape!=(cfg.tiles,cfg.edge,cfg.edge,cfg.edge,4) or not np.isfinite(x).all():
        raise ValueError("expected finite 4-channel 3D input tiles")
    # Keep an independent test slice whenever tiles >= 4. Development validation
    # is used for backtracking and codec selection; do not call it unseen test.
    if cfg.tiles >= 4:
        test_x=x[-1:]
        nval=max(1, (cfg.tiles-1)//4)
        val_x=x[-1-nval:-1]
        train_x=x[:-1-nval]
    else:
        test_x=None
        train_x=x[:1]
        val_x=x[1:]
    c_tr,_=pyramid(train_x,cfg.levels)
    c_val,_=pyramid(val_x,cfg.levels)
    model=Swarm(cfg,channels=4) if model is None else model
    if model.cfg!=cfg or model.channels!=4:
        raise ValueError('resumed checkpoint architecture does not match the request')
    learned=model.train(c_tr,c_val)
    selection,search=candidate_search(model,val_x,target_mse)
    blob=encode_packet(x,model,selection['keep'],selection['base_correction'])
    y,meta=decode_packet(blob,max_voxels=cfg.max_voxels)
    if test_x is not None:
        test_blob=encode_packet(test_x,model,selection['keep'],selection['base_correction'])
        test_dec,_=decode_packet(test_blob,max_voxels=cfg.max_voxels)
        independent_test={'tiles':len(test_x),'mse':float(np.mean((test_x-test_dec)**2)),
                          'max_abs_error':float(np.max(np.abs(test_x-test_dec))),
                          'packet_bytes':len(test_blob),
                          'meets_target':bool(np.mean((test_x-test_dec)**2)<=target_mse)}
    else:
        independent_test=None
    actual_mse=float(np.mean((x-y)**2))
    maxerr=float(np.max(np.abs(x-y)))
    rawbytes=x.nbytes
    final={'version':'DM3D-vNext-1.0','geometry':{'logical_sites':'10^18000000',
           'virtual_iterations':SYMBOLIC_ITERATIONS,'materialized_voxels':cfg.tiles*cfg.edge**3},
           'train':learned,'selected':selection,'independent_test':independent_test,
           'full_mse':actual_mse,
           'max_abs_error':maxerr,'raw_bytes':rawbytes,'packet_bytes':len(blob),
           'measured_raw_to_packet_ratio':rawbytes/len(blob),
           'routing_weights':{name:float(v) for name,v in zip(AGENTS,softmax(model.q))},
           'packet_checksum_sha256':meta['sha256'],
           'limitations':['finite NumPy simulation','symbolic virtual address space',
                          'local verification is not a generalization guarantee',
                          'numerically near-lossless mode is not bitwise identical',
                          'packet bytes include sparse indices and embedded decoder weights']}
    out.mkdir(parents=True,exist_ok=True)
    save_atomic(out/'encoded.dm3d.npz',blob)
    save_atomic(out/'model_checkpoint.npz',model.checkpoint_bytes())
    np.savez_compressed(out/'reconstructed.npz',input=x,decoded=y,residual=x-y)
    (out/'summary.json').write_text(json.dumps(final,indent=2),encoding='utf-8')
    (out/'candidate_search.json').write_text(json.dumps(search,indent=2),encoding='utf-8')
    (out/'training_history.json').write_text(json.dumps(model.history,indent=2),encoding='utf-8')
    return final


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--edge',type=int,default=8)
    ap.add_argument('--tiles',type=int,default=8)
    ap.add_argument('--levels',type=int,default=2)
    ap.add_argument('--hidden',type=int,default=12)
    ap.add_argument('--steps',type=int,default=80)
    ap.add_argument('--target-mse',type=float,default=.015)
    ap.add_argument('--out',type=Path,default=Path('dm3d_vnext_output'))
    ap.add_argument('--decode',type=Path,help='independently decode a .dm3d.npz packet (no input or trained encoder needed)')
    ap.add_argument('--resume',type=Path,help='resume from an earlier optimizer/memory checkpoint')
    ap.add_argument('--encode-npz',type=Path,help='preassembled 4-channel tensor in .npz under key input')
    ap.add_argument('--text',help='UTF-8 string for text modality')
    ap.add_argument('--image',help='Image filename, optional Pillow')
    ap.add_argument('--audio',help='PCM WAV filename')
    ap.add_argument('--video',help='Video filename, optional OpenCV')
    args=ap.parse_args(argv)
    if args.decode:
        if args.decode.stat().st_size>256_000_000:
            raise ValueError('file is over packet byte budget')
        data=args.decode.read_bytes()
        y,meta=decode_packet(data)
        args.out.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(args.out/'decoded_only.npz',decoded=y)
        print(json.dumps({'decoded_shape':list(y.shape), 'checksum':meta['sha256'],
                          'output':str(args.out/'decoded_only.npz')},indent=2))
        return
    model=None
    if args.resume:
        if args.resume.stat().st_size>256_000_000: raise ValueError('checkpoint too large')
        model=Swarm.from_checkpoint(args.resume.read_bytes())
        cfg=model.cfg
    else:
        cfg=Config(edge=args.edge,tiles=args.tiles,levels=args.levels,hidden=args.hidden,train_steps=args.steps)
    x=None
    if args.encode_npz:
        with np.load(args.encode_npz,allow_pickle=False) as f:
            x=f['input'].copy()
    if x is None and any(v is not None for v in (args.text,args.image,args.audio,args.video)):
        from dr_moagi_inward_intelligence import Configuration,make_dataset
        legacy=Configuration(n_tiles=args.tiles,edge=args.edge,active_tiles=min(4,args.tiles),steps=0)
        x,_=make_dataset(legacy,args.text,args.image,args.audio,args.video)
    metrics=run(cfg,args.out,args.target_mse,x,model)
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':
    main()
