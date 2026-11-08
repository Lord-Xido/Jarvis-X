#!/usr/bin/env python3
"""Reproduce held-out synthetic ONN3 versus polynomial v1 and zlib controls.
zfp and SZ3 are not executed. Neural codec is lossy; zlib is lossless.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from omega_ann_codec import train_model, run_trial
from omega_stream import stream_encode,stream_verify_full,stream_decode_to_npy
import omega_codec_probe as old

def main():
    output=Path(__file__).resolve().parent/'results'
    output.mkdir(exist_ok=True)
    params,train=train_model(steps=220,seed=4)
    full_report=[]
    for case in ('affine','smooth','noise','mixed'):
        vol=old.generate_volume(case,128,seed=11)
        trial=run_trial(vol,params,case,output)
        prior=old.run_volume(vol,case,.98,.98,1000.)
        prior_brief={'status':prior['status'],'K':prior.get('K'),'ratio':prior.get('compression_ratio'),
                     'quality':prior.get('quality_after_roundtrip')}
        data_path=output/f'{case}_source.npy'
        np.save(data_path,vol)
        mmap=np.load(data_path,mmap_mode='r',allow_pickle=False)
        stream_path=output/f'{case}_streamed.onn3'
        stream_meta=stream_encode(mmap,params,stream_path,batch_tiles=17)
        verified=stream_verify_full(mmap,stream_path,batch_tiles=31)
        if case=='affine':
            stream_decode_to_npy(stream_path,output/'affine_reconstructed.npy',batch_tiles=19)
        match={key:(abs(verified[key]-trial['quality'][key]) if key in trial['quality'] else None)
               for key in ('mean_q','min_q','rmse','psnr_db','max_absolute_error')}
        full_report.append({'case':case,'ann_v2':trial,'prior_polynomial_v1':prior_brief,
                            'streaming_metadata':stream_meta,'streaming_verified':verified,
                            'differences_stream_vs_in_memory':match})
        print(f'{case:6} neural={trial["ratio_with_embedded_model"]:8.2f}x mean={verified["mean_q"]:.6f} min={verified["min_q"]:.6f} covered={verified["tiles_verified"]}')
    report={'method':'trained separable 3D neural spatial factors + nonlinear decoder MLP',
            'training':train,'quality_definition':'clip(1-tile_RMSE/tile_range,0,1); constant tiles use whole-volume range',
            'thresholds':{'mean_quality':.98,'minimum_tile_quality':.98,'compression_ratio':1000},
            'evaluated_tiles':512,'side':128,
            'metric_note':'Custom normalized RMSE, not SSIM or clinical fidelity.',
            'baselines_executed':['zlib lossless','polynomial v1'],
            'baselines_unavailable':['zfp','SZ3'],'cases':full_report}
    (output/'benchmark_streaming.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Wrote:',output/'benchmark_streaming.json')

if __name__=='__main__':main()
