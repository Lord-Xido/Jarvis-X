#!/usr/bin/env python3
"""Ω-ENGINE quality + bitrate commit controller.

Commit an ONN3 trained codec bitstream only after complete tile-by-tile
verification. If it misses a target, test deterministic polynomial v1 as an
explicitly labeled fallback. Failed candidates are removed.
"""
from __future__ import annotations
import json,os,sys,tempfile
from pathlib import Path
import numpy as np
from omega_ann_codec import train_model
from omega_stream import stream_encode,stream_verify_full
import omega_codec_probe as old

def negotiate(volume:np.ndarray,params:dict,outdir:Path,name:str='volume',mean_q=.98,min_q=.98,target_ratio=1000.)->dict:
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    vol=np.asarray(volume,dtype=np.float32)
    if not np.isfinite(vol).all():raise ValueError('Nonfinite volume')
    tmp_path=outdir/(name+'.pending.onn3')
    accepted_path=outdir/(name+'.onn3')
    legacy_path=outdir/(name+'.om3d')
    result={'name':name,'requested':{'mean_q':mean_q,'min_q':min_q,'ratio':target_ratio},'candidates':[]}
    try:
        stream_encode(vol,params,tmp_path)
        measured=stream_verify_full(vol,tmp_path)
        allowed=measured['mean_q']>=mean_q and measured['min_q']>=min_q and measured['ratio_with_embedded_model']>=target_ratio
        result['candidates'].append({'codec':'trained_ANN_int16','measurement':measured,'target_met':allowed})
        if allowed:
            tmp_path.replace(accepted_path)
            result.update(status='COMMITTED',codec='trained_ANN_int16',file=str(accepted_path),
                          ratio=measured['ratio_with_embedded_model'],min_quality=measured['min_q'])
            return result
    finally:
        tmp_path.unlink(missing_ok=True)
    prior=old.run_volume(vol,name,mean_q,min_q,target_ratio)
    fallback={'codec':'deterministic_polynomial_v1','status':prior['status'],
              'K':prior.get('K'),'ratio':prior.get('compression_ratio'),
              'quality':prior.get('quality_after_roundtrip'),
              'target_met':prior.get('target_met',False)}
    result['candidates'].append(fallback)
    if prior.get('target_met'):
        coeff,_,_=old.encode_coefficients(old.to_tiles(vol))
        raw=old.serialize_coefficients(coeff,tuple(vol.shape),prior['K'])
        recon,_=old.decode_bitstream(raw)
        if recon.shape!=vol.shape:raise AssertionError('Fallback decode failed')
        if abs(vol.nbytes/len(raw)-prior['compression_ratio'])>1e-6:raise AssertionError('Rate mismatch')
        legacy_path.write_bytes(raw)
        result.update(status='COMMITTED',codec='deterministic_polynomial_v1',file=str(legacy_path),
                      ratio=prior['compression_ratio'],min_quality=prior['quality_after_roundtrip']['quality_min_tile'])
        return result
    result.update(status='REJECTED',codec=None,reason='Neither candidate satisfied both quality and bitrate.')
    return result

def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_npy',type=Path)
    parser.add_argument('--output-dir',type=Path,default=Path('results'))
    parser.add_argument('--train-steps',type=int,default=220)
    args=parser.parse_args()
    params,_=train_model(steps=args.train_steps)
    volume=np.load(args.input_npy,mmap_mode='r',allow_pickle=False)
    res=negotiate(volume,params,args.output_dir,args.input_npy.stem)
    print(json.dumps(res,indent=2))

if __name__=='__main__':main()
