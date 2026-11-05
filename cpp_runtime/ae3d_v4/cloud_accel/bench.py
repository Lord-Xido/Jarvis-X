"""Reproducible C++ baseline vs Torch device comparison, with correctness gates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile
import time

import numpy as np
import torch
from jx_accel import Archive, NeuralCodec, Volume, read_bounded


def bench(codec: NeuralCodec, volume: Volume, cpp_binary: str, model_path: str, repetitions: int = 5, cpp_kernel: str | None = None) -> dict:
    if repetitions < 3:
        raise ValueError('at least 3 runs are needed')
    input_blocks=volume.blocks()
    timings={'cpp_encode_s':[],'torch_encode_s':[],'torch_decode_s':[]}
    cpp_blob=None
    with tempfile.TemporaryDirectory() as directory:
        input_path=Path(directory)/'input.jxvol'
        output_path=Path(directory)/'output.jxpkt'
        input_path.write_bytes(volume.serialize())
        # Warmup both kernels, outside the measured window.
        codec.encode(volume, False)
        subprocess.run([cpp_binary,'encode',str(input_path),str(model_path),str(output_path),'lossy'],check=True,capture_output=True)
        for _ in range(repetitions):
            t0=time.perf_counter()
            subprocess.run([cpp_binary,'encode',str(input_path),str(model_path),str(output_path),'lossy'],check=True,capture_output=True)
            timings['cpp_encode_s'].append(time.perf_counter()-t0)
            cpp_blob=output_path.read_bytes()
            t0=time.perf_counter()
            a=codec.encode(volume,False)
            timings['torch_encode_s'].append(time.perf_counter()-t0)
            t0=time.perf_counter()
            _=codec.decode(a)
            timings['torch_decode_s'].append(time.perf_counter()-t0)
    cpp_arc=Archive.parse(cpp_blob)
    torch_arc=codec.encode(volume,False)
    q_equal = cpp_arc.q == torch_arc.q
    # Check C++ predictions against PyTorch on exactly the SAME C++ quantized codes.
    with tempfile.TemporaryDirectory() as directory:
        archive_path=Path(directory)/'from_cpp.jxpkt'; out_path=Path(directory)/'decoded_cpp.jxvol'
        archive_path.write_bytes(cpp_blob)
        subprocess.run([cpp_binary,'decode',str(archive_path),str(model_path),str(out_path)],check=True,capture_output=True)
        cpp_decoded=Volume.parse(out_path.read_bytes())
    torch_decoded=codec.decode(cpp_arc)
    max_error=int(np.abs(torch_decoded.voxels.astype(np.int16)-cpp_decoded.voxels.astype(np.int16)).max())
    if not q_equal or max_error:
        raise AssertionError(f'cross-runtime parity failed: equal_latents={q_equal}, max_decode_delta={max_error}')
    tcpp=statistics.median(timings['cpp_encode_s']); ttorch=statistics.median(timings['torch_encode_s'])
    result={'hardware':platform.platform(),'torch':torch.__version__,'device':codec.device,
            'cuda_available':torch.cuda.is_available(),'model_sha256':codec.sha256,
            'dims':list(volume.dims),'voxels':int(volume.voxels.size),'blocks':int(input_blocks.shape[0]),
            'repetitions':repetitions,'median_cpp_encode_s':tcpp,'median_torch_encode_s':ttorch,
            'median_torch_decode_s':statistics.median(timings['torch_decode_s']),
            'relative_cpp_to_torch_encode':tcpp/ttorch,'latent_byte_identical':q_equal,
            'max_prediction_byte_difference':max_error,
            'scope':'End-to-end local encode incl. subprocess startup and disk I/O for C++; in-memory Torch encode; not a controlled kernel-only comparison',
            'gpu_measured':codec.device=='cuda'}
    if cpp_kernel:
        with tempfile.TemporaryDirectory() as directory:
            input_path=Path(directory)/'input.jxvol'
            input_path.write_bytes(volume.serialize())
            proc=subprocess.run([cpp_kernel,str(model_path),str(input_path),str(repetitions),'1'],
                                check=True,capture_output=True,text=True)
            cpp_stats=json.loads(proc.stdout)
        core_times=[]
        # Warmed-up Torch tensor kernel, excluding model loading and file I/O.
        for _ in range(3): codec.encode_blocks(input_blocks)
        for _ in range(repetitions):
            if codec.device=='cuda': torch.cuda.synchronize()
            t0=time.perf_counter()
            codec.encode_blocks(input_blocks)
            if codec.device=='cuda': torch.cuda.synchronize()
            core_times.append(time.perf_counter()-t0)
        t_cpp=cpp_stats['median_kernel_seconds'];t_torch=statistics.median(core_times)
        result['kernel_benchmark']={'cpp_v4_median_s':t_cpp,'torch_median_s':t_torch,
                                     'relative_cpp_to_torch':t_cpp/t_torch,
                                     'scoped_to':'C++ per-block v4 encode_q loop versus batched Torch encode_blocks; both exclude file I/O/model load',
                                     'cpp_build':'-O3 -march=native on this CPU',
                                     'torch_threads':torch.get_num_threads()}
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--volume',required=True)
    p.add_argument('--cpp',required=True)
    p.add_argument('--cpp-kernel',default=None)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--repetitions',type=int,default=5)
    p.add_argument('--out',default='bench.json')
    p.add_argument('--threads',type=int,default=1)
    args=p.parse_args()
    torch.set_num_threads(args.threads)
    codec=NeuralCodec(read_bounded(args.model),device=args.device)
    result=bench(codec,Volume.parse(read_bounded(args.volume)),args.cpp,args.model,args.repetitions,args.cpp_kernel)
    Path(args.out).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
