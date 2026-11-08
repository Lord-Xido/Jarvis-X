#!/usr/bin/env python3
"""Omega 3D codec proof-of-concept: 3D projection, binary serialization,
round-trip decoding, byte-counted rate and reconstruction quality measurement.

This is a deterministic low-order polynomial transform codec, NOT a trained
ANN, not a 10^18-voxel benchmark and not a state-of-the-art claim.
"""
from __future__ import annotations
import argparse
import json
import math
import struct
import zlib
from pathlib import Path
import numpy as np

TILE = 16
SIDE = TILE**3
MAGIC = b"OM3D"
VERSION = 1
HEADER = struct.Struct("<4sB3IBBII")

def make_basis() -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-1., 1., TILE, dtype=np.float64)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    X2, Y2, Z2 = (t*t - np.mean(t*t) for t in (X, Y, Z))
    basis = np.stack([
        np.ones_like(X), X, Y, Z,
        X*Y, X*Z, Y*Z,
        X2, Y2, Z2,
        X*Y*Z, X2*Y,
    ]).reshape(12, SIDE)
    norms = (basis*basis).sum(axis=1)
    assert np.all(norms > 0)
    gram = basis @ basis.T
    assert np.allclose(gram, np.diag(norms), rtol=1e-12, atol=1e-10)
    return basis, norms

BASIS, NORMS = make_basis()

def to_tiles(volume: np.ndarray) -> np.ndarray:
    if volume.ndim != 3 or any(d % TILE != 0 for d in volume.shape):
        raise ValueError("Expected a 3D array whose dimensions are multiples of 16")
    z,y,x = volume.shape
    return (volume.reshape(z//TILE, TILE, y//TILE, TILE, x//TILE, TILE)
            .transpose(0,2,4,1,3,5).reshape(-1, SIDE))

def from_tiles(tiles: np.ndarray, shape: tuple[int,int,int]) -> np.ndarray:
    z,y,x = shape
    expected = (z//TILE)*(y//TILE)*(x//TILE)
    if tiles.shape != (expected, SIDE):
        raise ValueError("Tile dimensions do not match volume header")
    return (tiles.reshape(z//TILE, y//TILE, x//TILE, TILE, TILE, TILE)
            .transpose(0,3,1,4,2,5).reshape(shape).astype(np.float32))

def encode_coefficients(tiles: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    source = np.asarray(tiles, dtype=np.float64)
    dot = source @ BASIS.T
    coeff = (dot / NORMS[None, :]).astype(np.float32)
    orig_sse = np.einsum('ij,ij->i', source, source)
    return coeff, dot, orig_sse

def projected_error_sse(coeff: np.ndarray, dot: np.ndarray,
                        orig_sse: np.ndarray, k: int) -> np.ndarray:
    c = coeff[:, :k].astype(np.float64)
    explained = 2.0*np.sum(c*dot[:, :k], axis=1) - np.sum(c*c*NORMS[:k], axis=1)
    return np.maximum(orig_sse - explained, 0)

def quality_stats(tiles: np.ndarray, errors_sse: np.ndarray) -> dict:
    ranges = np.ptp(tiles, axis=1).astype(np.float64)
    global_range = float(np.ptp(tiles))
    local_scale = np.where(ranges > 1e-12, ranges, max(global_range, 1e-12))
    rmses = np.sqrt(errors_sse / SIDE)
    qs = np.clip(1.0 - rmses / local_scale, 0., 1.)
    global_rmse = math.sqrt(float(np.mean(errors_sse)) / SIDE)
    global_q = max(0.0, 1.0 - global_rmse / max(global_range, 1e-12))
    return {"quality_mean_tile": float(np.mean(qs)),
            "quality_min_tile": float(np.min(qs)),
            "quality_global": float(global_q),
            "rmse_global": float(global_rmse),
            "max_abs_error_bound_not_guaranteed": True}

def serialize_coefficients(coeff: np.ndarray, dims: tuple[int,int,int], k: int) -> bytes:
    if k < 1 or k > BASIS.shape[0]:
        raise ValueError("K out of range")
    payload = np.ascontiguousarray(coeff[:, :k], dtype='<f4').tobytes(order='C')
    header = HEADER.pack(MAGIC, VERSION, *dims, TILE, k, len(coeff), zlib.crc32(payload))
    return header + payload

def decode_bitstream(data: bytes) -> tuple[np.ndarray,dict]:
    if len(data) < HEADER.size:
        raise ValueError("Truncated bitstream header")
    magic,ver,z,y,x,tile,k,n,crc = HEADER.unpack(data[:HEADER.size])
    if magic != MAGIC or ver != VERSION or tile != TILE:
        raise ValueError("Unrecognized OM3D bitstream version")
    if not 1 <= k <= BASIS.shape[0] or min(z,y,x) <= 0:
        raise ValueError("Invalid dimensions or K")
    if any(d % tile for d in (z,y,x)):
        raise ValueError("Shape incompatible with tile size")
    if n != (z//tile)*(y//tile)*(x//tile):
        raise ValueError("Tile count inconsistent with dimensions")
    payload = data[HEADER.size:]
    if len(payload) != n*k*4 or zlib.crc32(payload) != crc:
        raise ValueError("Corrupted/truncated coefficient bitstream")
    coeff = np.frombuffer(payload, dtype='<f4').reshape(n,k)
    reconstructed_tiles = (coeff.astype(np.float64) @ BASIS[:k]).astype(np.float32)
    return from_tiles(reconstructed_tiles,(z,y,x)), {"shape":(z,y,x),"k":k,"tiles":n}

def generate_volume(case: str, side: int, seed: int = 7) -> np.ndarray:
    if side % TILE:
        raise ValueError("Side must be a multiple of 16")
    rng = np.random.default_rng(seed)
    if case in ('affine', 'mixed'):
        n_tiles = (side//TILE)**3
        coef = np.empty((n_tiles,4),dtype=np.float64)
        coef[:,0] = rng.uniform(0.1, 0.9, size=n_tiles)
        coef[:,1:] = rng.uniform(-0.2, 0.2, size=(n_tiles,3))
        tiles = (coef @ BASIS[:4]).astype(np.float32)
        if case == 'mixed':
            tiles[-1,:] = rng.uniform(0,1,SIDE).astype(np.float32)
        return from_tiles(tiles, (side,)*3)
    if case == 'smooth':
        a = np.arange(side, dtype=np.float32)
        Z,Y,X = np.meshgrid(a,a,a,indexing='ij')
        return (0.5 + 0.2*np.sin(2*np.pi*X/side)
                    + 0.15*np.cos(2*np.pi*Y/side)
                    + 0.1*np.sin(2*np.pi*Z/side)).astype(np.float32)
    if case == 'noise':
        return rng.uniform(0.,1.,(side,side,side)).astype(np.float32)
    raise ValueError(f"Unknown case: {case}")

def run_volume(volume: np.ndarray, case: str, threshold: float,
               min_tile_threshold: float, target_ratio: float,
               output_path: Path | None = None) -> dict:
    if volume.ndim != 3 or not np.isfinite(volume).all():
        raise ValueError("Input must be a finite 3D numeric volume")
    volume = np.asarray(volume, dtype=np.float32)
    tiles = to_tiles(volume)
    coeff,dot,source_sse = encode_coefficients(tiles)
    records = []
    accepted = None
    for k in range(BASIS.shape[0], 0, -1):
        error = projected_error_sse(coeff,dot,source_sse,k)
        stats = quality_stats(tiles,error)
        provisional_stream = serialize_coefficients(coeff, tuple(volume.shape), k)
        ratio = volume.nbytes / len(provisional_stream)
        ok = stats['quality_mean_tile'] >= threshold and stats['quality_min_tile'] >= min_tile_threshold
        records.append({"K":k,"bytes":len(provisional_stream),"ratio":ratio,
                        "mean_q":stats['quality_mean_tile'],"min_q":stats['quality_min_tile'],
                        "accept":ok})
        if not ok:
            break
        accepted = (k,provisional_stream)
        if ratio >= target_ratio:
            break
    result = {"case":case,"shape":list(volume.shape),"voxel_count":int(volume.size),
              "raw_bytes":int(volume.nbytes),"target_ratio":target_ratio,"threshold_mean":threshold,
              "threshold_min_tile":min_tile_threshold,"tested":records,
              "accepted": accepted is not None}
    if accepted is None:
        result['reason'] = 'No K between 12 and 1 satisfied both quality constraints.'
        result['status'] = 'REJECTED'
        return result
    k,stream = accepted
    reconstructed,meta = decode_bitstream(stream)
    delta = volume.astype(np.float64)-reconstructed.astype(np.float64)
    tile_errors = np.sum(np.square(to_tiles(delta.astype(np.float32)), dtype=np.float64),axis=1)
    actual_q = quality_stats(tiles,tile_errors)
    result.update({"K":k,"status": "TARGET_MET" if volume.nbytes / len(stream) >= target_ratio else "QUALITY_ONLY",
                   "target_met": bool(volume.nbytes / len(stream) >= target_ratio),
                   "serialized_bytes":len(stream),
                   "compression_ratio":float(volume.nbytes / len(stream)),
                   "quality_after_roundtrip":actual_q,
                   "max_abs_error":float(np.max(np.abs(delta))),
                   "psnr_db":float(20*np.log10(max(float(np.ptp(volume)),1e-12) /
                                               max(actual_q['rmse_global'],1e-15))),
                   "decoder_meta":meta})
    if (actual_q['quality_mean_tile'] < threshold or
            actual_q['quality_min_tile'] < min_tile_threshold):
        result['accepted'] = False
        result['status'] = 'REJECTED'
        result['reason'] = 'Serialized round-trip failed quality revalidation.'
        return result
    if output_path is not None:
        output_path.parent.mkdir(parents=True,exist_ok=True)
        output_path.write_bytes(stream)
        result['bitstream_path'] = str(output_path)
    return result

def run_one(case: str, side: int, threshold: float,
            min_tile_threshold: float, target_ratio: float,
            output_path: Path | None = None) -> dict:
    return run_volume(generate_volume(case, side), case, threshold,
                      min_tile_threshold, target_ratio, output_path)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=('affine','smooth','noise','mixed','all'),default='all')
    p.add_argument('--side',type=int,default=128)
    p.add_argument('--quality',type=float,default=.98)
    p.add_argument('--min-tile-quality',type=float,default=.98)
    p.add_argument('--ratio',type=float,default=1000.)
    p.add_argument('--output-dir',type=Path,default=Path('.'))
    p.add_argument('--input-npy',type=Path,help='Optional arbitrary 3D float-compatible .npy volume instead of synthetic cases')
    args=p.parse_args()
    cases = ('affine','smooth','noise','mixed') if args.case=='all' else (args.case,)
    if args.input_npy is not None:
        cases=('input',)
    report=[]
    for case in cases:
        out=args.output_dir/f'{case}_{args.side}.om3d'
        if args.input_npy is not None:
            volume = np.load(args.input_npy, allow_pickle=False)
            out=args.output_dir/f'{args.input_npy.stem}.om3d'
            result = run_volume(volume, args.input_npy.stem, args.quality,
                                args.min_tile_quality, args.ratio, out)
        else:
            result = run_one(case,args.side,args.quality,args.min_tile_quality,args.ratio,out)
        report.append(result)
        if result['accepted']:
            print(f'{case:8s} {result["status"]:12s} K={result["K"]:2d}  ratio={result["compression_ratio"]:9.2f}x  '
                  f'mean={result["quality_after_roundtrip"]["quality_mean_tile"]:.8f}  '
                  f'min={result["quality_after_roundtrip"]["quality_min_tile"]:.8f}  '
                  f'PSNR={result["psnr_db"]:.2f} dB')
        else:
            print(f'{case:8s} REJECT: {result["reason"]}  '
                  f'K={result["tested"][-1]["K"]}  '
                  f'mean={result["tested"][-1]["mean_q"]:.6f}  '
                  f'min={result["tested"][-1]["min_q"]:.6f}')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    report_path = args.output_dir/f'omega_codec_probe_{args.side}.json' if args.input_npy is None else args.output_dir/f'omega_codec_probe_{args.input_npy.stem}.json'
    report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'Report: {report_path}')

if __name__=='__main__': main()
