#!/usr/bin/env python3
"""Measured one-shot native CPU benchmark. These are local sample results, not a GPU claim."""
import argparse
import ctypes as C
import json
import time
from serve import Core, Stats

parser=argparse.ArgumentParser()
parser.add_argument('--dimension',type=int,default=10)
parser.add_argument('--iterations',type=int,default=1000)
args=parser.parse_args()
engine=Core(args.dimension,20261104)
try:
    with engine.lock:
        engine.auto=False
        start=time.perf_counter()
        for i in range(0,args.iterations,10000):
            count=min(10000,args.iterations-i)
            engine.lib.jx_step(engine.pointer,count)
        elapsed=time.perf_counter()-start
        stats=Stats();engine.lib.jx_stats(engine.pointer,C.byref(stats))
        verified=engine.verify()
        route=engine.route(0,engine.n-1)
    print(json.dumps({
        'engine':'C++20 native CPU','dimension':args.dimension,'voxels':engine.n,
        'iterations':args.iterations,'measured_total_s':round(elapsed,6),
        'cell_updates_per_sec':int((engine.n*args.iterations)/elapsed),
        'codec_raw_bytes':verified['raw_bytes'],'codec_encoded_bytes':verified['encoded_bytes'],
        'codec_byte_exact':verified['verified'],'native_a_star_hops':route['hops'],
        'route_latency_ms':route['measured_ms'],
        'fixed_point_last_max_delta':stats.last_max_delta
    },indent=2))
finally:engine.close()
