#!/usr/bin/env python3
"""Ω-ANN v2: trained, separable 3-D neural transform + int16 quantized latent.

An executable research prototype (not a general-purpose or exavoxel codec).
Independent, self-contained binary decoding needs NumPy but not PyTorch.
PyTorch is needed only for model training.

Model: learned four rank-1 spatial basis fields (4 x 3 x 16 factor vectors)
with a learned nonlinear 4 -> 8 -> 4 tanh decoder correction.
Latents: 4 int16 coefficients per 4096-voxel float32 tile.
Both learned model and per-volume quantization scales are stored in each file.
"""
from __future__ import annotations
import argparse, json, math, struct, time, zlib
from pathlib import Path
from typing import Iterable
import numpy as np

TILE, K, HIDDEN = 16, 4, 8
ELEMENTS = TILE**3
MAGIC, VERSION = b'ONN3', 2
# magic, version, z/y/x (uint32), tile, K, num_tiles, model bytes, payload bytes, crc32
HEADER = struct.Struct('<4sB3IBBIIII')
PARAM_COUNT = K*3*TILE + HIDDEN*K + HIDDEN + K*HIDDEN + K + K
MODEL_BYTES = 4*PARAM_COUNT
MAX_VOXELS = 2**31  # safety check for this in-memory demonstrator

def default_factors() -> np.ndarray:
    c = np.ones(TILE, dtype=np.float32); c /= np.linalg.norm(c)
    t = np.linspace(-1, 1, TILE, dtype=np.float32); t /= np.linalg.norm(t)
    return np.stack([[c,c,c],[t,c,c],[c,t,c],[c,c,t]],axis=0).astype(np.float32)

def normalized_factors(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    return raw / np.maximum(np.linalg.norm(raw,axis=-1,keepdims=True),1e-12)

def basis_from_factors(factors: np.ndarray) -> np.ndarray:
    f = normalized_factors(factors)
    return np.einsum('ki,kj,kl->kijl',f[:,0],f[:,1],f[:,2]).reshape(K,ELEMENTS)

def to_tiles(volume: np.ndarray) -> np.ndarray:
    if volume.ndim != 3 or any(s <= 0 or s % TILE for s in volume.shape):
        raise ValueError('3D volume dimensions must be positive multiples of 16')
    a,b,c=volume.shape
    return volume.reshape(a//TILE,TILE,b//TILE,TILE,c//TILE,TILE).transpose(0,2,4,1,3,5).reshape(-1,ELEMENTS)

def from_tiles(tiles: np.ndarray,shape: tuple[int,int,int]) -> np.ndarray:
    a,b,c=shape
    return tiles.reshape(a//TILE,b//TILE,c//TILE,TILE,TILE,TILE).transpose(0,3,1,4,2,5).reshape(shape)

def get_params(model) -> dict[str,np.ndarray]:
    return {name: obj.detach().cpu().numpy().copy().astype(np.float32) for name,obj in
            [('factors',model.factors),('w1',model.w1.weight),('b1',model.w1.bias),
             ('w2',model.w2.weight),('b2',model.w2.bias)]}

def params_to_bytes(params: dict[str,np.ndarray],scales: np.ndarray) -> bytes:
    vals = [params['factors'].reshape(-1),params['w1'].reshape(-1),params['b1'].reshape(-1),
            params['w2'].reshape(-1),params['b2'].reshape(-1),scales.reshape(-1)]
    return np.ascontiguousarray(np.concatenate(vals),dtype='<f4').tobytes()

def params_from_bytes(buf: bytes) -> tuple[dict[str,np.ndarray],np.ndarray]:
    if len(buf) != MODEL_BYTES: raise ValueError('Bad embedded model length')
    a = np.frombuffer(buf,dtype='<f4')
    if not np.isfinite(a).all(): raise ValueError('Model contains nonfinite values')
    cursor=0
    def take(n,shape):
        nonlocal cursor
        part=a[cursor:cursor+n].reshape(shape).copy(); cursor+=n;return part
    params={'factors':take(K*3*TILE,(K,3,TILE)),
            'w1':take(HIDDEN*K,(HIDDEN,K)),'b1':take(HIDDEN,(HIDDEN,)),
            'w2':take(K*HIDDEN,(K,HIDDEN)),'b2':take(K,(K,))}
    scales=take(K,(K,))
    if np.any(scales <= 0):raise ValueError('Bad quantization scales')
    return params,scales

def apply_decoder(codes: np.ndarray, params: dict[str,np.ndarray]) -> np.ndarray:
    c=np.asarray(codes,dtype=np.float64)
    h=np.tanh((c/32.) @ params['w1'].astype(np.float64).T + params['b1'])
    offset=h @ params['w2'].astype(np.float64).T + params['b2']
    basis=basis_from_factors(params['factors'])
    return ((c+0.05*offset) @ basis).astype(np.float32)

def encode_codes(tiles:np.ndarray,params:dict[str,np.ndarray]) -> np.ndarray:
    b=basis_from_factors(params['factors'])
    gram=b@b.T
    observations = np.asarray(tiles,dtype=np.float64) @ b.T
    return np.linalg.solve(gram, observations.T).T

def pack(volume: np.ndarray, params: dict[str,np.ndarray]) -> tuple[bytes,dict]:
    vol=np.asarray(volume,dtype=np.float32)
    if not np.isfinite(vol).all():raise ValueError('Input nonfinite')
    tiles=to_tiles(vol)
    codes=encode_codes(tiles,params)
    scales=(np.maximum(np.max(np.abs(codes),axis=0),1e-6) / 32760.).astype(np.float32)
    q=np.clip(np.rint(codes/scales),-32768,32767).astype('<i2')
    model_bytes=params_to_bytes(params,scales)
    payload=q.tobytes(order='C')
    crc=zlib.crc32(model_bytes+payload)
    header=HEADER.pack(MAGIC,VERSION,*vol.shape,TILE,K,len(tiles),len(model_bytes),len(payload),crc)
    return header+model_bytes+payload, {'tiles':len(tiles),'saturated_coefficients':int(np.sum((codes/scales)>32767)+np.sum((codes/scales)<-32768))}

def parse_bitstream(buf:bytes):
    if len(buf)<HEADER.size:raise ValueError('Truncated stream header')
    magic,ver,a,b,c,tile,k,n,m_len,p_len,crc=HEADER.unpack(buf[:HEADER.size])
    if (magic,ver,tile,k)!=(MAGIC,VERSION,TILE,K):raise ValueError('Bad stream magic/version/shape')
    if min(a,b,c)<=0 or any(d%TILE for d in (a,b,c)):raise ValueError('Invalid dimensions')
    if a*b*c>MAX_VOXELS:raise ValueError('Volume exceeds this demonstrator safety limit')
    expected=(a//TILE)*(b//TILE)*(c//TILE)
    if n!=expected or m_len!=MODEL_BYTES or p_len!=n*K*2:raise ValueError('Invalid stream metadata')
    if len(buf)!=HEADER.size+m_len+p_len:raise ValueError('Truncated or extra stream bytes')
    body=buf[HEADER.size:]
    if zlib.crc32(body)!=crc:raise ValueError('CRC32 mismatch')
    params,scales=params_from_bytes(body[:m_len])
    q=np.frombuffer(body[m_len:],dtype='<i2').reshape(n,K)
    return (a,b,c),params,scales,q

def decode_tiles_stream(buf:bytes,batch_tiles=64) -> Iterable[tuple[int,np.ndarray]]:
    dims,params,scales,q=parse_bitstream(buf)
    for i in range(0,len(q),batch_tiles):
        latents=q[i:i+batch_tiles].astype(np.float64)*scales
        yield i,apply_decoder(latents,params)

def unpack(buf:bytes,batch_tiles:int=64) -> np.ndarray:
    dims,_,_,q=parse_bitstream(buf)
    out_tiles=np.empty((len(q),ELEMENTS),dtype=np.float32)
    for i,chunk in decode_tiles_stream(buf,batch_tiles):
        out_tiles[i:i+len(chunk)]=chunk
    return from_tiles(out_tiles,dims)

def quality(vol:np.ndarray,decoded:np.ndarray)->dict:
    if decoded.shape!=vol.shape:raise ValueError('Reconstruction has wrong shape')
    orig=to_tiles(np.asarray(vol,dtype=np.float32)).astype(np.float64)
    recon=to_tiles(np.asarray(decoded,dtype=np.float32)).astype(np.float64)
    err=orig-recon
    rmses=np.sqrt(np.mean(err*err,axis=1))
    ranges=np.ptp(orig,axis=1)
    global_range=max(float(np.ptp(orig)),1e-12)
    scales=np.where(ranges>1e-12,ranges,global_range)
    qs=np.clip(1-rmses/scales,0,1)
    rmse=float(np.sqrt(np.mean(err*err)))
    return {'mean_q':float(np.mean(qs)),'min_q':float(np.min(qs)),
            'global_q':float(max(0,1-rmse/global_range)),
            'rmse':rmse,'psnr_db':float(20*math.log10(global_range/max(rmse,1e-15))),
            'max_absolute_error':float(np.max(np.abs(err)))}

def gate(stats:dict,mean_threshold=0.98,min_tile_threshold=0.98,ratio=1000.,actual_ratio=0.)->dict:
    return {'quality_accepted':stats['mean_q']>=mean_threshold and stats['min_q']>=min_tile_threshold,
            'rate_accepted':actual_ratio>=ratio,
            'target_met':stats['mean_q']>=mean_threshold and stats['min_q']>=min_tile_threshold and actual_ratio>=ratio}

def run_trial(volume:np.ndarray,params:dict,case:str,output_dir:Path|None=None,mean_threshold=.98,min_tile_threshold=.98,target_ratio=1000.,write_rejected=False)->dict:
    t0=time.perf_counter()
    vol=np.asarray(volume,dtype=np.float32)
    stream,extra=pack(vol,params)
    t1=time.perf_counter()
    reconstructed=unpack(stream)
    t2=time.perf_counter()
    stats=quality(vol,reconstructed)
    actual_ratio=vol.nbytes/len(stream)
    gates=gate(stats,mean_threshold,min_tile_threshold,target_ratio,actual_ratio)
    result={'case':case,'shape':list(vol.shape),'raw_bytes':int(vol.nbytes),'stream_bytes':len(stream),
            'embedded_model_bytes':MODEL_BYTES,'header_bytes':HEADER.size,'latent_payload_bytes':len(stream)-MODEL_BYTES-HEADER.size,
            'ratio_with_embedded_model':actual_ratio,'ratio_shared_model_excluded':vol.nbytes/(len(stream)-MODEL_BYTES),
            'quality':stats,'gates':gates,'quantization':extra,
            'encode_seconds':float(t1-t0),'decode_seconds':float(t2-t1),
            'lossless_zlib_bytes':len(zlib.compress(vol.tobytes(),level=6)),
            'baseline_zlib_ratio':vol.nbytes/len(zlib.compress(vol.tobytes(),level=6))}
    if output_dir and (gates['quality_accepted'] or write_rejected):
        output_dir.mkdir(parents=True,exist_ok=True)
        path=output_dir/f'{case}.onn3';path.write_bytes(stream)
        result['stream_path']=str(path)
    return result

def build_model(seed=4,perturbation=.02):
    import torch
    from torch import nn
    from torch.nn import functional as F
    torch.manual_seed(seed)
    class ANN(nn.Module):
        def __init__(self):
            super().__init__()
            init=default_factors()+perturbation*np.random.default_rng(seed).normal(size=(K,3,TILE)).astype(np.float32)
            self.factors=nn.Parameter(torch.tensor(init))
            self.w1=nn.Linear(K,HIDDEN)
            self.w2=nn.Linear(HIDDEN,K)
            nn.init.zeros_(self.w2.weight);nn.init.zeros_(self.w2.bias)
        def forward(self,x):
            f=F.normalize(self.factors,dim=-1)
            b=torch.einsum('ki,kj,kl->kijl',f[:,0],f[:,1],f[:,2]).reshape(K,ELEMENTS)
            coeff=torch.linalg.solve(b@b.T,(x@b.T).T).T
            correction=self.w2(torch.tanh(self.w1(coeff/32.)))
            return (coeff+0.05*correction)@b
    return ANN()

def train_model(steps:int=220,seed:int=4,train_samples:int=1024,batch_size:int=64)->tuple[dict,dict]:
    import torch
    torch.set_num_threads(min(4,torch.get_num_threads()))
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    coefs=np.empty((train_samples,4),dtype=np.float32)
    coefs[:,0]=rng.uniform(.15,.85,size=train_samples)
    coefs[:,1:]=rng.uniform(-.23,.23,size=(train_samples,3))
    import omega_codec_probe as old
    tiles=(coefs@old.BASIS[:4]).astype(np.float32)
    data=torch.from_numpy(tiles)
    model=build_model(seed)
    optim=torch.optim.Adam(model.parameters(),lr=.02)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(optim,T_max=max(steps,1),eta_min=.0008)
    model.eval()
    with torch.no_grad():
        before=float(torch.mean((model(data[:64])-data[:64])**2).item())
    model.train()
    t0=time.perf_counter()
    for step in range(steps):
        ix=torch.randint(0,train_samples,(batch_size,))
        batch=data[ix]
        y=model(batch)
        loss=torch.mean((y-batch)**2)
        optim.zero_grad();loss.backward();optim.step();sched.step()
    model.eval()
    with torch.no_grad():
        after=float(torch.mean((model(data[:64])-data[:64])**2).item())
    info={'training_samples':train_samples,'steps':steps,'seed':seed,'loss_before':before,
          'loss_after':after,'train_seconds':time.perf_counter()-t0,
          'model_trained':steps>0,'network':'trainable 3D rank-one tensor spatial factors + nonlinear latent MLP'}
    return get_params(model),info

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps',type=int,default=220)
    p.add_argument('--side',type=int,default=128)
    p.add_argument('--seed',type=int,default=4)
    p.add_argument('--output-dir',type=Path,default=Path('results'))
    p.add_argument('--input-npy',type=Path)
    p.add_argument('--mean-quality',type=float,default=.98)
    p.add_argument('--min-tile-quality',type=float,default=.98)
    p.add_argument('--ratio',type=float,default=1000.)
    args=p.parse_args()
    if args.side%TILE:raise ValueError('--side must be multiple of 16')
    params,train=train_model(steps=args.steps,seed=args.seed)
    import omega_codec_probe as old
    cases=[(args.input_npy.stem,np.load(args.input_npy,allow_pickle=False))] if args.input_npy else [
        (case,old.generate_volume(case,args.side,seed=11)) for case in ('affine','smooth','noise','mixed')]
    results=[]
    for case,vol in cases:
        r=run_trial(vol,params,case,args.output_dir,args.mean_quality,args.min_tile_quality,args.ratio)
        results.append(r)
        print(f'{case:7} ratio={r["ratio_with_embedded_model"]:10.2f}x Qmean={r["quality"]["mean_q"]:.7f} Qmin={r["quality"]["min_q"]:.7f} PSNR={r["quality"]["psnr_db"]:.2f}dB target={r["gates"]["target_met"]}')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    report={'train':train,'settings':vars(args).copy(),'results':results,'limitations':['Not exavoxel scale','Not a generic dataset benchmark','No zfp/SZ3 baseline unless independently installed','CRC32 is not authenticated encryption']}
    report['settings']['output_dir']=str(args.output_dir)
    report['settings']['input_npy']=str(args.input_npy) if args.input_npy else None
    filename=args.output_dir/'benchmark.json'
    filename.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('TRAIN',train,'REPORT',filename)

if __name__=='__main__':main()
