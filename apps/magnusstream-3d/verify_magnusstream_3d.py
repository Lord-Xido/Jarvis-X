#!/usr/bin/env python3
"""Independent 3D reference codec verifying the fixed-point F = D o M o E.
Not extracted from, nor asserted to be implemented by, the uploaded MagnusStream artifact.
"""
import argparse, json, math, hashlib, struct
from pathlib import Path
import numpy as np

N = 16
B = 2
L = N // B
ALPHA = 0.25

def make_field(n=N):
    # Spatially varied deterministic 3D test signal, bounded to [0,1].
    z, y, x = np.indices((n, n, n), dtype=np.float64)
    r2 = ((x - 7.5) / 4.2)**2 + ((y - 7.5) / 4.2)**2 + ((z - 7.5) / 4.2)**2
    v = (0.18 + 0.53*np.exp(-r2) + 0.14*np.sin((x+1)*0.54)*np.cos((z+1)*0.4)
         + 0.08*np.sin((y+z+1)*0.9) + 0.04*((x+y+z) % 2))
    return np.clip(v, 0., 1.).astype(np.float32).astype(np.float64)

def encode(x):
    assert x.shape == (N, N, N)
    return x.reshape(L, B, L, B, L, B).mean(axis=(1, 3, 5))

def manifold(z, alpha=ALPHA):
    assert z.shape == (L, L, L)
    neighbor_avg = sum(np.roll(z, k, axis=ax) for ax in range(3) for k in [-1, 1]) / 6.0
    return (1-alpha)*z + alpha*neighbor_avg

def decode(z):
    assert z.shape == (L, L, L)
    return np.repeat(np.repeat(np.repeat(z, B, axis=0), B, axis=1), B, axis=2)

def F(x, alpha=ALPHA):
    return decode(manifold(encode(x),alpha))

def metrics(a, b):
    d = a - b
    mse = float(np.mean(d*d))
    denom = max(float(np.linalg.norm(a)),1e-15)
    return {'mse':mse,'rmse':float(np.sqrt(mse)),
            'psnr_db':float(-10*math.log10(max(mse,1e-30))),
            'residual':float(np.linalg.norm(d)/denom)}


def pack_latent(z):
    """Versioned fixed-width transport for the 8^3 float32 latent volume."""
    if z.shape != (L, L, L) or not np.isfinite(z).all():
        raise ValueError("Expected a finite 8x8x8 latent field")
    return b'MS3D' + struct.pack('<7I', 1, N, N, N, L, L, L) + np.asarray(z, dtype='<f4').tobytes(order='C')


def unpack_latent(payload):
    """Fail closed on wrong magic, dimensions, version, length or nonfinite data."""
    if len(payload) != 32 + L**3*4 or payload[:4] != b'MS3D':
        raise ValueError("Invalid MS3D payload length or magic")
    if struct.unpack('<7I', payload[4:32]) != (1, N, N, N, L, L, L):
        raise ValueError("Invalid MS3D version or spatial shape")
    z = np.frombuffer(payload, dtype='<f4', count=L**3, offset=32).astype(np.float64).reshape((L, L, L))
    if not np.isfinite(z).all():
        raise ValueError("Nonfinite encoded sample")
    return z

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--loops',type=int,default=1000)
    p.add_argument('--json', default=None, help='Optional JSON receipt destination')
    a=p.parse_args()
    if not 1 <= a.loops <= 20000: raise SystemExit('loops must be between 1 and 20000')
    source=make_field()
    x=source.copy()
    checkpoints={}
    initial_mean=float(source.mean())
    variance_previous=float(x.var())
    monotone_violations=0
    for i in range(1,a.loops+1):
        xnext=F(x)
        variance_next=float(xnext.var())
        if variance_next > variance_previous + 1e-12: monotone_violations += 1
        if i in {1,2,10,50,100,250,500,1000,a.loops}:
            checkpoints[str(i)] = {
                'mse_vs_source':metrics(source,xnext)['mse'],
                'relative_step':metrics(xnext,x)['residual'],
                'variance':variance_next,
                'mean_drift':abs(float(xnext.mean())-initial_mean)
            }
        x,variance_previous=xnext,variance_next
    constant=np.full((N,N,N),0.625, dtype=np.float64)
    f_constant=F(constant)
    max_constant_error=float(np.max(np.abs(f_constant-constant)))
    global_fixed=np.full((N,N,N),initial_mean,dtype=np.float64)
    max_global_fixed_error=float(np.max(np.abs(F(global_fixed)-global_fixed)))
    reverse_error = float(np.max(np.abs(decode(encode(constant))-constant)))
    lat=encode(x).astype('<f4')
    payload=pack_latent(lat)
    assert np.array_equal(unpack_latent(payload).astype('<f4'), lat)
    # 32-byte header: 4-byte magic + 7 x uint32
    assert len(payload)==32+L**3*4
    native_bytes=int(source.size*4)
    result={
        'type':'independent_reference_3d_feedback_codec',
        'original_artifact_verified_as_real_codec':False,
        'algorithm':'F(X) = D(M(E(X))); E=2x2x2 block mean, M=6-neighbor periodic diffusion, D=nearest-neighbor 3D reconstruction',
        'grid_original':[N,N,N], 'latent_grid':[L,L,L],
        'original_voxels':int(source.size),'latent_values':int(lat.size),
        'iterations':a.loops,'alpha':ALPHA,
        'raw_bytes':native_bytes,'encoded_bytes_with_header':len(payload),
        'serialized_payload_ratio':native_bytes/len(payload),
        'source_sha256_float32':hashlib.sha256(source.astype('<f4').tobytes()).hexdigest(),
        'encoded_sha256':hashlib.sha256(payload).hexdigest(),
        'constant_preservation_error':max_constant_error,
        'constant_codec_roundtrip_error':reverse_error,
        'global_fixed_point_error':max_global_fixed_error,
        'final_relative_residual':metrics(x,F(x))['residual'],
        'final_reconstruction_metrics':metrics(source,x),
        'mean_preservation_error':abs(float(x.mean())-initial_mean),
        'variance_nonincrease_violations':monotone_violations,
        'checkpoints':checkpoints,
        'checks':{
            'shape':source.shape==(16,16,16) and lat.shape==(8,8,8),
            'constant_invariant':max_constant_error<1e-12,
            'global_fixed_invariant':max_global_fixed_error<1e-12,
            'mean_preserved':abs(float(x.mean())-initial_mean)<1e-10,
            'variance_nonincreasing':monotone_violations==0,
            'latent_payload_smaller':len(payload)<native_bytes,
            'recursive_convergence':metrics(x,F(x))['residual']<1e-8 if a.loops>=1000 else None
        }
    }
    if a.json:
        dest = Path(a.json)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['iterations','original_voxels','latent_values','serialized_payload_ratio','final_relative_residual','mean_preservation_error','variance_nonincrease_violations','checks','checkpoints']},indent=2))
if __name__=='__main__':main()