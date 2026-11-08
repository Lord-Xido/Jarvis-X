#!/usr/bin/env python3
"""Memory-bounded tile-by-tile ONN3 encoder, validator and .npy decoder.

Supports np.load(..., mmap_mode='r') to avoid materializing full source volumes.
All tiles are visited, never subsampled. The entire encoded file is CRC-checked
before a stream is trusted. This is single-node CPU streaming, not distributed.
"""
from __future__ import annotations
import argparse, json, os, struct, sys, zlib
from pathlib import Path
import numpy as np
from omega_ann_codec import (
    TILE,K,ELEMENTS,MAGIC,VERSION,HEADER,MODEL_BYTES,MAX_VOXELS,
    encode_codes,params_to_bytes,params_from_bytes,apply_decoder,unpack,
)

def validate_shape(volume):
    if volume.ndim != 3 or any(d<=0 or d%TILE for d in volume.shape):
        raise ValueError('Expected a nonempty 3D volume with dimensions divisible by 16')
    if volume.size>MAX_VOXELS:raise ValueError('In-memory demonstrator voxel safety limit exceeded')

def tile_batches(volume, batch_tiles=32):
    validate_shape(volume)
    if batch_tiles<=0:raise ValueError('batch_tiles must be positive')
    nz,ny,nx=(v//TILE for v in volume.shape)
    batch=[]; start=0; total=0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                tile=np.asarray(volume[iz*TILE:(iz+1)*TILE,
                                       iy*TILE:(iy+1)*TILE,
                                       ix*TILE:(ix+1)*TILE],dtype=np.float32).reshape(ELEMENTS)
                if not np.isfinite(tile).all():raise ValueError('Nonfinite input values')
                batch.append(tile)
                total+=1
                if len(batch)>=batch_tiles:
                    yield start,np.stack(batch)
                    start=total;batch=[]
    if batch:yield start,np.stack(batch)

def stream_encode(volume,params,path:Path,batch_tiles=32):
    validate_shape(volume)
    peak=np.full(K,1e-6,dtype=np.float64)
    n=0
    for _,tiles in tile_batches(volume,batch_tiles):
        codes=encode_codes(tiles,params)
        peak=np.maximum(peak,np.max(np.abs(codes),axis=0))
        n+=len(tiles)
    scales=(peak/32760.).astype(np.float32)
    model=params_to_bytes(params,scales)
    if len(model)!=MODEL_BYTES:raise AssertionError('Unexpected model size')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload_bytes=n*K*2
    if payload_bytes>0xffffffff:raise ValueError('This format uses a uint32 payload byte field')
    checksum=zlib.crc32(model)
    with path.open('wb') as f:
        f.write(bytes(HEADER.size))
        f.write(model)
        actual=0
        for _,tiles in tile_batches(volume,batch_tiles):
            codes=encode_codes(tiles,params)
            q=np.clip(np.rint(codes/scales),-32768,32767).astype('<i2')
            by=q.tobytes()
            f.write(by);checksum=zlib.crc32(by,checksum);actual+=len(by)
        if actual!=payload_bytes:raise AssertionError('Incomplete payload')
        hdr=HEADER.pack(MAGIC,VERSION,*volume.shape,TILE,K,n,len(model),payload_bytes,checksum)
        f.seek(0);f.write(hdr)
    return {'file':str(path),'tiles':n,'raw_bytes':int(np.dtype(np.float32).itemsize*volume.size),
            'stream_bytes':path.stat().st_size,'ratio_with_model':(volume.size*4)/path.stat().st_size,
            'model_bytes':len(model),'peak_code_magnitudes':peak.tolist()}

def verified_reader(path:Path,chunk_bytes=1<<20):
    path=Path(path)
    with path.open('rb') as f:
        hdr=f.read(HEADER.size)
        if len(hdr)!=HEADER.size:raise ValueError('Truncated header')
        magic,ver,a,b,c,tile,k,n,m_len,p_len,crc=HEADER.unpack(hdr)
        if (magic,ver,tile,k)!=(MAGIC,VERSION,TILE,K):raise ValueError('Invalid format')
        if min(a,b,c)<=0 or any(d%TILE for d in (a,b,c)):raise ValueError('Invalid shape')
        if a*b*c>MAX_VOXELS:raise ValueError('Volume too large for demonstrator')
        if n!=(a//TILE)*(b//TILE)*(c//TILE) or m_len!=MODEL_BYTES or p_len!=n*K*2:
            raise ValueError('Invalid declared payload length')
        if path.stat().st_size!=HEADER.size+m_len+p_len:raise ValueError('Invalid file length')
        computed=0
        for chunk in iter(lambda:f.read(chunk_bytes),b''):
            computed=zlib.crc32(chunk,computed)
        if computed!=crc:raise ValueError('CRC32 integrity check failed')
        f.seek(HEADER.size)
        model=f.read(m_len)
        params,scales=params_from_bytes(model)
        yield (a,b,c),n,params,scales,f

def iterate_decoded(path:Path,batch_tiles=32):
    for dims,n,params,scales,f in verified_reader(path):
        for start in range(0,n,batch_tiles):
            count=min(batch_tiles,n-start)
            raw=f.read(count*K*2)
            if len(raw)!=count*K*2:raise ValueError('Unexpected truncated payload')
            q=np.frombuffer(raw,dtype='<i2').reshape(count,K)
            recon=apply_decoder(q.astype(np.float64)*scales,params)
            yield dims,start,recon

def tile_slice(index:int,dims:tuple[int,int,int]):
    ny,nx=dims[1]//TILE,dims[2]//TILE
    iz=index//(ny*nx)
    iy=(index//nx)%ny
    ix=index%nx
    return (slice(iz*TILE,(iz+1)*TILE),slice(iy*TILE,(iy+1)*TILE),slice(ix*TILE,(ix+1)*TILE))

def stream_decode_to_npy(path:Path,out_path:Path,batch_tiles=32):
    out_path=Path(out_path);out_path.parent.mkdir(parents=True,exist_ok=True)
    writer=None
    try:
        for dims,start,tiles in iterate_decoded(path,batch_tiles):
            if writer is None:
                writer=np.lib.format.open_memmap(out_path,mode='w+',dtype=np.float32,shape=dims)
            for off,tile in enumerate(tiles):
                writer[tile_slice(start+off,dims)]=tile.reshape(TILE,TILE,TILE)
        if writer is None:raise ValueError('Empty bitstream')
        writer.flush()
    finally:
        if writer is not None:del writer
    return str(out_path)

def stream_verify_full(volume,path:Path,batch_tiles=32):
    validate_shape(volume)
    global_min=np.inf;global_max=-np.inf
    for _,src in tile_batches(volume,batch_tiles):
        global_min=min(global_min,float(np.min(src)))
        global_max=max(global_max,float(np.max(src)))
    global_range=max(global_max-global_min,1e-12)
    sse=0.;sum_quality=0.;min_quality=1.;max_error=0.;total_tiles=0;total_values=0
    for dims,start,decoded in iterate_decoded(path,batch_tiles):
        if tuple(dims)!=tuple(volume.shape):raise ValueError('Decoded dimensions differ from input')
        src=np.stack([np.asarray(volume[tile_slice(start+i,dims)],dtype=np.float32).reshape(ELEMENTS)
                      for i in range(len(decoded))]).astype(np.float64)
        residual=src-decoded.astype(np.float64)
        squared=np.sum(residual*residual,axis=1)
        ranges=np.ptp(src,axis=1)
        scales=np.where(ranges>1e-12,ranges,global_range)
        qs=np.clip(1-np.sqrt(squared/ELEMENTS)/scales,0.,1.)
        sum_quality+=float(np.sum(qs));min_quality=min(min_quality,float(np.min(qs)))
        sse+=float(np.sum(squared));max_error=max(max_error,float(np.max(np.abs(residual))))
        total_tiles+=len(decoded);total_values+=decoded.size
    expected=volume.size//ELEMENTS
    if total_tiles!=expected:raise ValueError('Decoded tile coverage incomplete')
    rmse=float(np.sqrt(sse/total_values))
    ratio=(volume.size*4)/Path(path).stat().st_size
    return {'coverage_fraction':total_values/volume.size,'tiles_verified':total_tiles,
            'ratio_with_embedded_model':ratio,'mean_q':sum_quality/total_tiles,'min_q':min_quality,
            'global_q':max(0,1-rmse/global_range),'rmse':rmse,
            'psnr_db':20*np.log10(global_range/max(rmse,1e-15)),'max_absolute_error':max_error,
            'mean_quality_pass':sum_quality/total_tiles>=.98,
            'minimum_quality_pass':min_quality>=.98,
            'rate_pass':ratio>=1000.}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='cmd',required=True)
    enc=sub.add_parser('encode');enc.add_argument('input_npy',type=Path);enc.add_argument('output_onn3',type=Path)
    enc.add_argument('--steps',type=int,default=220)
    dec=sub.add_parser('decode');dec.add_argument('input_onn3',type=Path);dec.add_argument('output_npy',type=Path)
    val=sub.add_parser('verify');val.add_argument('input_npy',type=Path);val.add_argument('input_onn3',type=Path)
    args=parser.parse_args()
    if args.cmd=='encode':
        from omega_ann_codec import train_model
        params,_=train_model(steps=args.steps)
        result=stream_encode(np.load(args.input_npy,mmap_mode='r',allow_pickle=False),params,args.output_onn3)
    elif args.cmd=='decode':result={'decoded_path':stream_decode_to_npy(args.input_onn3,args.output_npy)}
    else:result=stream_verify_full(np.load(args.input_npy,mmap_mode='r',allow_pickle=False),args.input_onn3)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
