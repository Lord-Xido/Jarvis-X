"""CMX3: bounded procedural + residual 3D volumetric seed codec.

CMX3 stores a Mandelbulb-like procedural prior plus an optional quantized
residual lattice in a versioned, CRC-checked seed. The encoder never truncates
payloads: candidates larger than ``max_seed_bytes`` are rejected.
"""
from __future__ import annotations

import math
import struct
import zlib
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

SEED_MAX_BYTES = 1024


@dataclass(frozen=True)
class MandelbulbConfig:
    power: float = 8.0
    max_iter: int = 24
    inward: float = 1.2
    julia: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bound: float = 1.5


class Mandelbulb3D:
    """Vectorized Mandelbulb-like scalar field with optional spherical inversion."""

    def __init__(self, config: MandelbulbConfig = MandelbulbConfig()):
        if config.power <= 1.0:
            raise ValueError("power must be > 1")
        if not 1 <= config.max_iter <= 65535:
            raise ValueError("max_iter must be in [1, 65535]")
        if config.bound <= 0:
            raise ValueError("bound must be > 0")
        self.config = config

    def generate(self, res: int, dtype: np.dtype = np.float32) -> np.ndarray:
        if res < 4:
            raise ValueError("resolution must be >= 4")
        c = self.config
        axis = -c.bound + np.arange(res, dtype=np.float64) * (2.0 * c.bound / res)
        zz, yy, xx = np.meshgrid(axis, axis, axis, indexing="ij")
        zx, zy, zz1 = xx.copy(), yy.copy(), zz.copy()

        if c.inward > 1e-12:
            r2 = zx * zx + zy * zy + zz1 * zz1
            mask = r2 > 1e-12
            inv = np.ones_like(r2)
            inv[mask] = c.inward / r2[mask]
            zx *= inv; zy *= inv; zz1 *= inv

        dr = np.ones_like(zx)
        active = np.ones_like(zx, dtype=bool)
        jx, jy, jz = c.julia
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            for _ in range(c.max_iter):
                r = np.sqrt(zx*zx + zy*zy + zz1*zz1)
                active &= np.isfinite(r) & (r <= 4.0)
                if not np.any(active):
                    break
                safe = np.where(r > 1e-30, r, 1.0)
                theta = np.where(r > 1e-30, np.arccos(np.clip(zz1/safe, -1.0, 1.0)), 0.0)
                phi = np.arctan2(zy, zx)
                drn = np.power(r, c.power-1.0) * c.power * dr + 1.0
                zr = np.power(r, c.power)
                tp, pp = theta*c.power, phi*c.power
                st = np.sin(tp)
                nx = zr*st*np.cos(pp) + xx + jx
                ny = zr*st*np.sin(pp) + yy + jy
                nz = zr*np.cos(tp) + zz + jz
                zx = np.where(active, nx, zx)
                zy = np.where(active, ny, zy)
                zz1 = np.where(active, nz, zz1)
                dr = np.where(active, drn, dr)

            r = np.sqrt(zx*zx + zy*zy + zz1*zz1)
            out = np.zeros_like(r)
            valid = np.isfinite(r) & np.isfinite(dr) & (r > 0) & (dr > 0)
            out[valid] = 0.5 * np.log(r[valid]) * r[valid] / dr[valid]
        return np.maximum(np.nan_to_num(out), 0.0).astype(dtype)


class ResidualLattice:
    @staticmethod
    def sample(v: np.ndarray, n: int) -> np.ndarray:
        if n <= 0:
            return np.empty((0, 0, 0), dtype=np.float32)
        idx = np.rint(np.linspace(0, v.shape[0]-1, n)).astype(np.int64)
        return v[np.ix_(idx, idx, idx)].astype(np.float32, copy=False)

    @staticmethod
    def resize(cube: np.ndarray, target: int) -> np.ndarray:
        n = cube.shape[0]
        if n == 0:
            return np.zeros((target, target, target), dtype=np.float32)
        if n == target:
            return cube.astype(np.float32, copy=True)
        pos = np.linspace(0, n-1, target)
        lo = np.floor(pos).astype(int); hi = np.minimum(lo+1, n-1); w = pos-lo
        a = cube[lo,:,:]*(1-w)[:,None,None] + cube[hi,:,:]*w[:,None,None]
        b = a[:,lo,:]*(1-w)[None,:,None] + a[:,hi,:]*w[None,:,None]
        return (b[:,:,lo]*(1-w)[None,None,:] + b[:,:,hi]*w[None,None,:]).astype(np.float32)

    @staticmethod
    def quantize(cube: np.ndarray) -> tuple[bytes, float]:
        if cube.size == 0:
            return b"", 0.0
        scale = float(np.max(np.abs(cube)))
        if scale <= 1e-20:
            return b"", 0.0
        q = np.rint(np.clip(cube/scale, -1.0, 1.0)*127.0).astype(np.int8)
        return q.tobytes(order="C"), scale

    @staticmethod
    def dequantize(raw: bytes, n: int, scale: float) -> np.ndarray:
        expected = n**3
        q = np.frombuffer(raw, dtype=np.int8)
        if q.size != expected:
            raise ValueError(f"residual length mismatch: expected {expected}, got {q.size}")
        return (q.astype(np.float32).reshape(n,n,n) * (scale/127.0)).astype(np.float32)


def reconstruction_metrics(source: np.ndarray, recon: np.ndarray) -> dict[str, float]:
    a = np.asarray(source, dtype=np.float64); b = np.asarray(recon, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("shape mismatch")
    d = a-b; mse = float(np.mean(d*d)); peak = max(float(np.max(np.abs(a))), 1e-12)
    psnr = 100.0 if mse == 0.0 else 10.0*math.log10((peak*peak)/mse)
    return {"mse": mse, "rmse": math.sqrt(mse), "psnr_db": psnr,
            "max_abs_error": float(np.max(np.abs(d)))}


@dataclass(frozen=True)
class Candidate:
    anchor_res: int
    level: int
    seed_size: int
    mse: float
    scale: float
    payload: bytes


class CMX3Codec:
    """Versioned <=1 KiB procedural/residual codec."""

    MAGIC = b"CMX3"
    VERSION = 2
    FLAG_RESIDUAL = 1
    HEADER_FMT = ">4sBBfHfHfffBBfII"
    HEADER_SIZE = struct.calcsize(HEADER_FMT)

    def __init__(self, max_seed_bytes: int = SEED_MAX_BYTES):
        if max_seed_bytes < self.HEADER_SIZE:
            raise ValueError("seed budget smaller than header")
        self.max_seed_bytes = int(max_seed_bytes)

    @classmethod
    def _header(cls, cfg: MandelbulbConfig, res: int, n: int, level: int,
                scale: float, payload: bytes) -> bytes:
        return struct.pack(cls.HEADER_FMT, cls.MAGIC, cls.VERSION,
            cls.FLAG_RESIDUAL if payload else 0, cfg.power, cfg.max_iter, cfg.inward,
            res, cfg.julia[0], cfg.julia[1], cfg.julia[2], n, level, scale,
            len(payload), zlib.crc32(payload) & 0xFFFFFFFF)

    def encode(self, generator: Mandelbulb3D, res: int, source: Optional[np.ndarray] = None,
               anchors: Sequence[int] = (0,4,5,6,8,10,12),
               levels: Sequence[int] = (6,9)) -> tuple[bytes, dict[str, Any]]:
        cfg = generator.config
        if source is None:
            h = self._header(cfg, res, 0, 9, 0.0, b"")
            return h, {"mode":"procedural-only", "seed_size":len(h), "anchor_res":0}
        src = np.asarray(source, dtype=np.float32)
        if src.shape != (res,res,res):
            raise ValueError(f"source must have shape {(res,res,res)}")
        prior = generator.generate(res)
        candidates: list[Candidate] = []
        for n in dict.fromkeys((0, *anchors)):
            if not 0 <= n <= res:
                continue
            for level in levels:
                if n == 0:
                    payload, scale, recon = b"", 0.0, prior
                else:
                    raw, scale = ResidualLattice.quantize(ResidualLattice.sample(src-prior, n))
                    payload = zlib.compress(raw, level) if raw else b""
                    recon = prior if not payload else prior + ResidualLattice.resize(
                        ResidualLattice.dequantize(zlib.decompress(payload), n, scale), res)
                h = self._header(cfg, res, n if payload else 0, level, scale, payload)
                size = len(h)+len(payload)
                if size <= self.max_seed_bytes:
                    candidates.append(Candidate(n if payload else 0, level, size,
                        reconstruction_metrics(src, recon)["mse"], scale, payload))
        if not candidates:
            raise ValueError("no codec candidate fits seed budget")
        best = min(candidates, key=lambda c:(round(c.mse,15), c.seed_size, c.anchor_res))
        seed = self._header(cfg, res, best.anchor_res, best.level, best.scale, best.payload)+best.payload
        return seed, {"mode":"procedural+residual" if best.payload else "procedural-only",
            "seed_size":len(seed), "anchor_res":best.anchor_res, "zlib_level":best.level,
            "payload_size":len(best.payload), "candidate_count":len(candidates), "predicted_mse":best.mse}

    def decode(self, seed: bytes, target_res: Optional[int] = None) -> dict[str, Any]:
        if len(seed) < self.HEADER_SIZE:
            raise ValueError("seed shorter than header")
        f = struct.unpack(self.HEADER_FMT, seed[:self.HEADER_SIZE])
        magic, ver, flags, power, iters, inward, source_res, jx, jy, jz, n, level, scale, plen, crc = f
        if magic != self.MAGIC or ver != self.VERSION:
            raise ValueError("invalid or unsupported CMX3 seed")
        payload = seed[self.HEADER_SIZE:]
        if plen != len(payload):
            raise ValueError("payload length mismatch")
        if (zlib.crc32(payload) & 0xFFFFFFFF) != crc:
            raise ValueError("payload CRC32 mismatch")
        res = int(source_res if target_res is None else target_res)
        gen = Mandelbulb3D(MandelbulbConfig(power, iters, inward, (jx,jy,jz)))
        recon = gen.generate(res)
        if flags & self.FLAG_RESIDUAL:
            try:
                raw = zlib.decompress(payload)
            except zlib.error as exc:
                raise ValueError("invalid residual payload") from exc
            recon = recon + ResidualLattice.resize(ResidualLattice.dequantize(raw, n, scale), res)
        return {"resolution":res, "source_resolution":int(source_res), "anchor_res":int(n),
                "zlib_level":int(level), "residual_used":bool(flags & self.FLAG_RESIDUAL),
                "grid":recon, "config":gen.config}
