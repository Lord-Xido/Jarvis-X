#!/usr/bin/env python3
"""
terabyte_3d_engine.py
=====================

Jarvis-X / Moagi hierarchical terabyte-scale 3D autoencoding reference runtime.

Operational geometry
--------------------
1000 GB logical stream
    -> 62,500 x 16 MB blocks (100^3 x 4 float32)
    -> per-block 3D contraction: 100^3 -> 10^3 -> 1^3
    -> per-block 64-D fixed-point latent + Omega memory
    -> residual-preserving reconstruction
    -> threshold + Top-K selective correction
    -> local latent aggregation
    -> global 64-D inward fixed-point latent
    -> telemetry + optional 3D lattice viewer

The engine is streaming: it never allocates the full 1000 GB dataset.

Synthetic mode generates each 100^3 block procedurally. File mode can memory-map
a raw float32 stream whose size is an integer multiple of one 16 MB block.

This is a software reference engine. Throughput projections are derived from
measured host throughput; hypothetical EM/photonic timing is not asserted.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass
class TBConfig:
    logical_gb: float = 1000.0
    world_side: int = 1000
    block_side: int = 100
    factor: int = 10
    channels: int = 4
    dtype_bytes: int = 4
    latent_dim: int = 64
    folds: int = 64
    rho: float = 0.30
    spectral_target: float = 0.82
    beta: float = 0.88
    residual_quantum: float = 0.02
    root_quantum: float = 0.01
    correction_lambda: float = 0.35
    tau0: float = 0.15
    tau_decay: float = 0.60
    topk_fraction: float = 0.12
    correction_cycles: int = 5
    seed: int = 7

    @property
    def block_voxels(self) -> int:
        return self.block_side ** 3

    @property
    def block_bytes(self) -> int:
        return self.block_voxels * self.channels * self.dtype_bytes

    @property
    def block_mb(self) -> float:
        return self.block_bytes / 1e6

    @property
    def blocks_per_axis(self) -> int:
        if self.world_side % self.block_side:
            raise ValueError("world_side must be divisible by block_side")
        return self.world_side // self.block_side

    @property
    def blocks_per_volume(self) -> int:
        return self.blocks_per_axis ** 3

    @property
    def volume_bytes(self) -> int:
        return self.world_side ** 3 * self.channels * self.dtype_bytes

    @property
    def logical_bytes(self) -> int:
        return int(round(self.logical_gb * 1e9))

    @property
    def logical_blocks(self) -> int:
        return math.ceil(self.logical_bytes / self.block_bytes)

    @property
    def logical_volumes(self) -> float:
        return self.logical_bytes / self.volume_bytes


@dataclass(frozen=True)
class BlockAddress:
    block_id: int
    volume: int
    bx: int
    by: int
    bz: int

    @classmethod
    def from_id(cls, block_id: int, cfg: TBConfig) -> "BlockAddress":
        local = block_id % cfg.blocks_per_volume
        volume = block_id // cfg.blocks_per_volume
        a = cfg.blocks_per_axis
        bx = local // (a * a)
        rem = local % (a * a)
        by = rem // a
        bz = rem % a
        return cls(block_id, volume, bx, by, bz)


class BlockSource:
    def __init__(self, cfg: TBConfig):
        self.cfg = cfg

    def get(self, block_id: int) -> np.ndarray:
        raise NotImplementedError

    def available_blocks(self) -> Optional[int]:
        return None


class SyntheticBlockSource(BlockSource):
    def __init__(self, cfg: TBConfig):
        super().__init__(cfg)
        s = cfg.block_side
        q = (np.arange(s, dtype=np.float32) + 0.5) / s
        self.x = q[:, None, None]
        self.y = q[None, :, None]
        self.z = q[None, None, :]

    def get(self, block_id: int) -> np.ndarray:
        addr = BlockAddress.from_id(block_id, self.cfg)
        a = self.cfg.blocks_per_axis
        gx = (addr.bx + self.x) / a
        gy = (addr.by + self.y) / a
        gz = (addr.bz + self.z) / a
        phase = (addr.volume % 17) * 0.113
        c0 = np.sin(2 * np.pi * (gx + 0.17 * gy) + phase)
        c1 = np.cos(2 * np.pi * (gy + 0.11 * gz) - phase)
        c2 = np.sin(2 * np.pi * (gz + 0.13 * gx) + 0.5 * phase)
        c3 = np.cos(2 * np.pi * (gx + gy + gz) + 0.25 * phase)
        block = np.stack(
            [
                np.broadcast_to(c0, (self.cfg.block_side,) * 3),
                np.broadcast_to(c1, (self.cfg.block_side,) * 3),
                np.broadcast_to(c2, (self.cfg.block_side,) * 3),
                np.broadcast_to(c3, (self.cfg.block_side,) * 3),
            ],
            axis=-1,
        ).astype(np.float32, copy=False)
        if self.cfg.channels != 4:
            block = np.resize(
                block,
                (self.cfg.block_side, self.cfg.block_side, self.cfg.block_side, self.cfg.channels),
            ).astype(np.float32)
        return np.ascontiguousarray(np.tanh(block), dtype=np.float32)


class RawMemmapBlockSource(BlockSource):
    def __init__(self, cfg: TBConfig, path: str):
        super().__init__(cfg)
        self.path = Path(path)
        size = self.path.stat().st_size
        if size % cfg.block_bytes:
            raise ValueError(f"raw file size {size} is not divisible by block size {cfg.block_bytes}")
        self.n_blocks = size // cfg.block_bytes
        self.mm = np.memmap(self.path, mode="r", dtype=np.float32)

    def available_blocks(self) -> Optional[int]:
        return int(self.n_blocks)

    def get(self, block_id: int) -> np.ndarray:
        if block_id < 0 or block_id >= self.n_blocks:
            raise IndexError(block_id)
        n = self.cfg.block_voxels * self.cfg.channels
        start = block_id * n
        stop = start + n
        return np.asarray(self.mm[start:stop]).reshape(
            self.cfg.block_side,
            self.cfg.block_side,
            self.cfg.block_side,
            self.cfg.channels,
        )


@dataclass
class QuantizedResidual:
    q: float
    codes: Optional[np.ndarray] = None
    exact: Optional[np.ndarray] = None
    clipped: int = 0

    @classmethod
    def encode(cls, x: np.ndarray, q: float) -> "QuantizedResidual":
        x = np.asarray(x, np.float32)
        if q <= 0:
            return cls(q=0.0, exact=x.copy())
        raw = np.rint(x / q)
        clipped = int(np.count_nonzero((raw < -127) | (raw > 127)))
        codes = np.clip(raw, -127, 127).astype(np.int8)
        return cls(q=float(q), codes=codes, clipped=clipped)

    def decode(self) -> np.ndarray:
        if self.q <= 0:
            assert self.exact is not None
            return self.exact.astype(np.float32, copy=False)
        assert self.codes is not None
        return self.codes.astype(np.float32) * self.q

    @property
    def encoded_bytes(self) -> int:
        if self.q <= 0:
            assert self.exact is not None
            return int(self.exact.nbytes)
        assert self.codes is not None
        return int(self.codes.nbytes + 4)


@dataclass
class EncodedBlock:
    address: BlockAddress
    latent: np.ndarray
    omega: np.ndarray
    root_model: np.ndarray
    root_residual: QuantizedResidual
    coarse_residual: QuantizedResidual
    fine_residual: QuantizedResidual
    cube64: int
    delta64: float
    original_bytes: int
    encoded_bytes: int


class FixedPointCore:
    def __init__(self, cfg: TBConfig, seed_offset: int = 0):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed + seed_offset)
        d = cfg.latent_dim
        W = rng.normal(0.0, 1.0 / math.sqrt(d), (d, d)).astype(np.float32)
        sigma = float(np.linalg.svd(W, compute_uv=False)[0])
        self.Wz = W * (cfg.spectral_target / max(sigma, 1e-9))
        self.Wo = rng.normal(0.0, 0.25 / math.sqrt(d), (d, d)).astype(np.float32)
        self.b = rng.normal(0.0, 0.01, d).astype(np.float32)
        self.omega = np.zeros(d, dtype=np.float32)
        self.idx = np.arange(d, dtype=np.float32)

    @property
    def relaxed_bound(self) -> float:
        return 1.0 - self.cfg.rho + self.cfg.rho * self.cfg.spectral_target

    def lift4(self, root: np.ndarray) -> np.ndarray:
        root = np.asarray(root, np.float32).ravel()
        j = np.arange(1, len(root) + 1, dtype=np.float32)
        sin_a = np.outer(j * 0.173, self.idx) + root[:, None]
        cos_a = np.outer(j * 0.071, self.idx) - root[:, None]
        z = np.sum(np.sin(sin_a) + 0.5 * np.cos(cos_a), axis=0)
        z /= max(1.0, 1.5 * len(root))
        return np.tanh(z).astype(np.float32)

    def fold(self, z0: np.ndarray) -> Tuple[np.ndarray, float]:
        z = np.asarray(z0, np.float32).copy()
        invariant = self.Wo @ self.omega + self.b
        delta = 0.0
        for _ in range(self.cfg.folds):
            proposal = np.tanh(self.Wz @ z + invariant)
            zn = ((1.0 - self.cfg.rho) * z + self.cfg.rho * proposal).astype(np.float32)
            delta = float(np.linalg.norm(zn - z))
            z = zn
        self.omega = (self.cfg.beta * self.omega + (1.0 - self.cfg.beta) * z).astype(np.float32)
        return z, delta

    def root4(self, z: np.ndarray) -> np.ndarray:
        return np.array([np.mean(a) for a in np.array_split(z, self.cfg.channels)], np.float32)


class BlockCodec3D:
    def __init__(self, cfg: TBConfig):
        self.cfg = cfg
        self.core = FixedPointCore(cfg, seed_offset=0)

    def contract_100_to_10(self, block: np.ndarray) -> np.ndarray:
        f = self.cfg.factor
        s = self.cfg.block_side
        if s != f * f:
            raise ValueError("reference codec expects block_side == factor^2")
        c = self.cfg.channels
        return block.reshape(f, f, f, f, f, f, c).mean(axis=(1, 3, 5), dtype=np.float64).astype(np.float32)

    def expand_10_to_100(self, coarse: np.ndarray) -> np.ndarray:
        f = self.cfg.factor
        return np.repeat(np.repeat(np.repeat(coarse, f, axis=0), f, axis=1), f, axis=2)

    @staticmethod
    def _checksum16(*arrays: np.ndarray) -> int:
        h = hashlib.sha256()
        for a in arrays:
            h.update(np.ascontiguousarray(a).tobytes())
        return int.from_bytes(h.digest()[:2], "little")

    def _cube64(self, address: BlockAddress, z: np.ndarray, fine: QuantizedResidual, coarse: QuantizedResidual) -> int:
        op = 0xB3
        volume = address.volume & 0x3FF
        bx, by, bz = address.bx & 0xF, address.by & 0xF, address.bz & 0xF
        latent_hash = self._checksum16(z)
        res_hash = self._checksum16(fine.decode().ravel()[:4096], coarse.decode().ravel()) & 0x3FFF
        flags = (1 if self.cfg.residual_quantum > 0 else 0) | (2 if fine.clipped or coarse.clipped else 0)
        fields = ((op, 8), (volume, 10), (bx, 4), (by, 4), (bz, 4), (latent_hash, 16), (res_hash, 14), (flags, 4))
        word = 0
        for value, bits in fields:
            word = (word << bits) | (int(value) & ((1 << bits) - 1))
        return word

    def encode(self, address: BlockAddress, block: np.ndarray) -> EncodedBlock:
        block = np.asarray(block, np.float32)
        expected = (self.cfg.block_side,) * 3 + (self.cfg.channels,)
        if block.shape != expected:
            raise ValueError(f"block shape {block.shape} != {expected}")
        coarse = self.contract_100_to_10(block)
        root = coarse.mean(axis=(0, 1, 2), dtype=np.float64).astype(np.float32)
        z0 = self.core.lift4(root)
        z, delta = self.core.fold(z0)
        root_model = self.core.root4(z)
        root_r = QuantizedResidual.encode(root - root_model, self.cfg.root_quantum)
        coarse_r = QuantizedResidual.encode(coarse - root[None, None, None, :], self.cfg.residual_quantum)
        fine_pred = self.expand_10_to_100(coarse)
        fine_r = QuantizedResidual.encode(block - fine_pred, self.cfg.residual_quantum)
        cube = self._cube64(address, z, fine_r, coarse_r)
        encoded_bytes = z.nbytes + self.core.omega.nbytes + root_model.nbytes + root_r.encoded_bytes + coarse_r.encoded_bytes + fine_r.encoded_bytes + 8
        return EncodedBlock(address, z, self.core.omega.copy(), root_model, root_r, coarse_r, fine_r, cube, delta, int(block.nbytes), int(encoded_bytes))

    def decode(self, encoded: EncodedBlock) -> np.ndarray:
        root = encoded.root_model + encoded.root_residual.decode()
        coarse = root[None, None, None, :] + encoded.coarse_residual.decode()
        block = self.expand_10_to_100(coarse) + encoded.fine_residual.decode()
        return block.astype(np.float32, copy=False)

    def correct(self, target: np.ndarray, reconstruction: np.ndarray):
        reco = np.asarray(reconstruction, np.float32).copy()
        target = np.asarray(target, np.float32)
        history = []
        flat_target = target.reshape(-1, self.cfg.channels)
        flat_reco = reco.reshape(-1, self.cfg.channels)
        for t in range(self.cfg.correction_cycles):
            error = flat_target - flat_reco
            score = np.linalg.norm(error, axis=1)
            tau = self.cfg.tau0 * (self.cfg.tau_decay ** t)
            eligible = np.flatnonzero(score > tau)
            mse_before = float(np.mean(np.square(error, dtype=np.float64)))
            if len(eligible) == 0:
                history.append((t, 0, mse_before, mse_before))
                break
            k = min(len(eligible), max(1, math.ceil(self.cfg.topk_fraction * len(eligible))))
            candidate = score[eligible]
            if k < len(eligible):
                part = np.argpartition(candidate, -k)[-k:]
                selected = eligible[part]
            else:
                selected = eligible
            flat_reco[selected] += self.cfg.correction_lambda * error[selected]
            mse_after = float(np.mean(np.square(flat_target - flat_reco, dtype=np.float64)))
            history.append((t, int(len(selected)), mse_before, mse_after))
        return reco, history


class GlobalLatentReducer:
    def __init__(self, cfg: TBConfig):
        self.cfg = cfg
        self.core = FixedPointCore(cfg, seed_offset=1009)
        self.sum = np.zeros(cfg.latent_dim, dtype=np.float64)
        self.count = 0
        self.last_global = np.zeros(cfg.latent_dim, dtype=np.float32)

    def add(self, local_z: np.ndarray) -> None:
        self.sum += np.asarray(local_z, np.float64)
        self.count += 1

    def solve(self) -> Tuple[np.ndarray, float]:
        if self.count == 0:
            return self.last_global.copy(), 0.0
        z0 = (self.sum / self.count).astype(np.float32)
        z, delta = self.core.fold(z0)
        self.last_global = z
        return z, delta


@dataclass
class BlockMetrics:
    block_id: int
    volume: int
    coord: Tuple[int, int, int]
    mse_before_correction: float
    mse_after_correction: float
    psnr_after_db: float
    local_delta64: float
    cube64: int
    compression_ratio: float
    clipped_values: int
    elapsed_ms: float


@dataclass
class RunMetrics:
    processed_blocks: int
    processed_bytes: int
    elapsed_s: float
    measured_gbps: float
    projected_1000gb_s: float
    mean_mse: float
    mean_compression_ratio: float
    global_delta64: float
    global_norm: float


class Terabyte3DEngine:
    def __init__(self, cfg: TBConfig, source: BlockSource):
        self.cfg = cfg
        self.source = source
        self.codec = BlockCodec3D(cfg)
        self.global_reducer = GlobalLatentReducer(cfg)
        self.block_metrics = []
        self.processed_addresses = []

    @staticmethod
    def _psnr(mse: float, peak: float = 2.0) -> float:
        if mse <= 0:
            return float("inf")
        return 10.0 * math.log10((peak * peak) / mse)

    def process_block(self, block_id: int) -> BlockMetrics:
        t0 = time.perf_counter()
        block = self.source.get(block_id)
        address = BlockAddress.from_id(block_id, self.cfg)
        encoded = self.codec.encode(address, block)
        decoded = self.codec.decode(encoded)
        mse0 = float(np.mean(np.square(block - decoded, dtype=np.float64)))
        corrected, hist = self.codec.correct(block, decoded)
        mse1 = float(np.mean(np.square(block - corrected, dtype=np.float64)))
        self.global_reducer.add(encoded.latent)
        self.processed_addresses.append(address)
        clipped = encoded.root_residual.clipped + encoded.coarse_residual.clipped + encoded.fine_residual.clipped
        ratio = encoded.original_bytes / max(encoded.encoded_bytes, 1)
        m = BlockMetrics(block_id, address.volume, (address.bx, address.by, address.bz), mse0, mse1, self._psnr(mse1), encoded.delta64, encoded.cube64, float(ratio), int(clipped), (time.perf_counter() - t0) * 1000.0)
        self.block_metrics.append(m)
        return m

    def run(self, block_ids: Sequence[int]) -> RunMetrics:
        if not block_ids:
            raise ValueError("no block ids")
        available = self.source.available_blocks()
        if available is not None:
            for bid in block_ids:
                if bid >= available:
                    raise IndexError(f"block {bid} exceeds file source block count {available}")
        t0 = time.perf_counter()
        for bid in block_ids:
            self.process_block(int(bid))
        elapsed = time.perf_counter() - t0
        global_z, global_delta = self.global_reducer.solve()
        processed_bytes = len(block_ids) * self.cfg.block_bytes
        gbps = processed_bytes / max(elapsed, 1e-12) / 1e9
        projected = self.cfg.logical_gb / gbps if gbps > 0 else float("inf")
        return RunMetrics(len(block_ids), processed_bytes, elapsed, gbps, projected, float(np.mean([m.mse_after_correction for m in self.block_metrics])), float(np.mean([m.compression_ratio for m in self.block_metrics])), float(global_delta), float(np.linalg.norm(global_z)))


class GeometryViewer:
    def __init__(self, cfg: TBConfig, addresses: Sequence[BlockAddress]):
        import tkinter as tk
        self.tk = tk
        self.cfg = cfg
        self.addresses = list(addresses)
        self.root = tk.Tk()
        self.root.title("Jarvis-X — 1000 GB Hierarchical 3D Engine")
        self.root.geometry("1280x840")
        self.root.configure(bg="#020617")
        self.canvas = tk.Canvas(self.root, bg="#020617", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.yaw, self.pitch, self.zoom, self.last = 0.65, -0.42, 2.7, None
        self.canvas.bind("<ButtonPress-1>", self._down)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.canvas.bind("<Configure>", lambda e: self.draw())
        self.draw()

    def _down(self, e): self.last = (e.x, e.y)

    def _drag(self, e):
        if self.last:
            self.yaw += (e.x - self.last[0]) * 0.008
            self.pitch = float(np.clip(self.pitch + (e.y - self.last[1]) * 0.008, -1.4, 1.4))
        self.last = (e.x, e.y)
        self.draw()

    def _wheel(self, e):
        self.zoom *= 1.08 if e.delta > 0 else 0.92
        self.zoom = float(np.clip(self.zoom, 1.4, 7.0))
        self.draw()

    def _project(self, p: np.ndarray) -> np.ndarray:
        w, h = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        cy, sy, cp, sp = math.cos(self.yaw), math.sin(self.yaw), math.cos(self.pitch), math.sin(self.pitch)
        R = np.array([[cy, 0, sy], [sp * sy, cp, -sp * cy], [-cp * sy, sp, cp * cy]], np.float32)
        q = (p - 0.5) @ R.T
        d = np.maximum(q[:, 2] + self.zoom, 0.1)
        sc = min(w, h) * 0.80
        return np.c_[w / 2 + sc * q[:, 0] / d, h / 2 - sc * q[:, 1] / d]

    def draw(self):
        self.canvas.delete("all")
        a = self.cfg.blocks_per_axis
        V = np.array([[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1)], np.float32)
        P = self._project(V)
        edges = ((0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7))
        for i, j in edges: self.canvas.create_line(*P[i], *P[j], fill="#155E75", width=2)
        centers = np.array([[(x+.5)/a,(y+.5)/a,(z+.5)/a] for x in range(a) for y in range(a) for z in range(a)], np.float32)
        CP = self._project(centers)
        for x, y in CP: self.canvas.create_oval(x-1,y-1,x+1,y+1,fill="#334155",outline="")
        if self.addresses:
            vol = self.addresses[0].volume
            shown = [r for r in self.addresses if r.volume == vol]
            pts = np.array([[(r.bx+.5)/a,(r.by+.5)/a,(r.bz+.5)/a] for r in shown], np.float32)
            if len(pts):
                PP = self._project(pts)
                core = self._project(np.array([[.5,.5,.5]],np.float32))[0]
                for x,y in PP:
                    self.canvas.create_line(x,y,core[0],core[1],fill="#0EA5E9")
                    self.canvas.create_oval(x-4,y-4,x+4,y+4,fill="#F59E0B",outline="")
        core = self._project(np.array([[.5,.5,.5]],np.float32))[0]
        for r,col in ((18,"#312E81"),(11,"#7C3AED"),(5,"#E9D5FF")):
            self.canvas.create_oval(core[0]-r,core[1]-r,core[0]+r,core[1]+r,fill=col,outline="")
        self.canvas.create_text(20,20,anchor="nw",fill="#E5F3FF",font=("Courier",11),text=(f"virtual dataset: {self.cfg.logical_gb:.0f} GB\nblock: {self.cfg.block_side}³ x {self.cfg.channels} float32 = {self.cfg.block_mb:.0f} MB\nlogical blocks: {self.cfg.logical_blocks:,}\nblocks/1000³ volume: {self.cfg.blocks_per_volume:,}\nprocessed in view: {len(self.addresses):,}\norange = streamed block\npurple = global latent core"))

    def run(self): self.root.mainloop()


def choose_block_ids(cfg: TBConfig, count: int, start: int, stride: int):
    if count <= 0: raise ValueError("count must be positive")
    return [i for i in (start + n * stride for n in range(count)) if i < cfg.logical_blocks]


def print_header(cfg: TBConfig, source: BlockSource) -> None:
    print("=" * 96)
    print("MOAGI / JARVIS-X — HIERARCHICAL 1000 GB 3D AUTOENCODING ENGINE")
    print("=" * 96)
    print(f"logical dataset         : {cfg.logical_gb:.3f} GB")
    print(f"virtual 1000^3 volumes : {cfg.logical_volumes:.3f}")
    print(f"block geometry          : {cfg.block_side}^3 x {cfg.channels} float32")
    print(f"block bytes             : {cfg.block_bytes:,} ({cfg.block_mb:.3f} MB)")
    print(f"logical blocks          : {cfg.logical_blocks:,}")
    print(f"blocks per 1000^3       : {cfg.blocks_per_volume:,}")
    print(f"latent                  : {cfg.latent_dim}-D, {cfg.folds} folds")
    print(f"proposal spectral norm  : {cfg.spectral_target:.3f}")
    print(f"relaxed contraction q<= : {1-cfg.rho+cfg.rho*cfg.spectral_target:.6f}")
    print(f"q^64 theoretical bound : {(1-cfg.rho+cfg.rho*cfg.spectral_target)**cfg.folds:.6e}")
    print(f"residual quantum        : {cfg.residual_quantum}")
    print(f"source                  : {type(source).__name__}")
    print("=" * 96)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Hierarchical streaming 1000 GB 3D autoencoding reference engine")
    p.add_argument("--logical-gb", type=float, default=1000.0)
    p.add_argument("--blocks", type=int, default=4, help="blocks to physically execute")
    p.add_argument("--start-block", type=int, default=0)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--input", type=str, default=None, help="raw float32 block stream")
    p.add_argument("--residual-quantum", type=float, default=0.02)
    p.add_argument("--correction-cycles", type=int, default=5)
    p.add_argument("--gui", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    ns = p.parse_args(argv)
    cfg = TBConfig(logical_gb=max(0.016, ns.logical_gb), residual_quantum=max(0.0, ns.residual_quantum), correction_cycles=max(0, ns.correction_cycles), seed=ns.seed)
    source = RawMemmapBlockSource(cfg, ns.input) if ns.input else SyntheticBlockSource(cfg)
    block_ids = choose_block_ids(cfg, ns.blocks, ns.start_block, max(1, ns.stride))
    if not block_ids:
        print("No block ids fall inside the logical dataset.", file=sys.stderr)
        return 2
    print_header(cfg, source)
    engine = Terabyte3DEngine(cfg, source)
    run = engine.run(block_ids)
    print()
    print(f"{'block':>7} {'vol':>4} {'xyz':>11} {'MSE0':>12} {'MSE1':>12} {'PSNR':>8} {'ratio':>8} {'delta64':>11} {'clip':>7} {'ms':>9}")
    for m in engine.block_metrics:
        xyz = f"{m.coord[0]},{m.coord[1]},{m.coord[2]}"
        ps = "inf" if not math.isfinite(m.psnr_after_db) else f"{m.psnr_after_db:.2f}"
        print(f"{m.block_id:7d} {m.volume:4d} {xyz:>11} {m.mse_before_correction:12.3e} {m.mse_after_correction:12.3e} {ps:>8} {m.compression_ratio:8.3f} {m.local_delta64:11.3e} {m.clipped_values:7d} {m.elapsed_ms:9.2f}")
    print()
    print("GLOBAL INWARD REDUCTION")
    print(f"processed blocks        : {run.processed_blocks:,}")
    print(f"processed data          : {run.processed_bytes/1e9:.6f} GB")
    print(f"measured elapsed        : {run.elapsed_s:.6f} s")
    print(f"measured throughput     : {run.measured_gbps:.6f} GB/s")
    print(f"1000 GB projection      : {run.projected_1000gb_s:.3f} s at measured host rate")
    print(f"mean corrected MSE      : {run.mean_mse:.6e}")
    print(f"mean codec ratio        : {run.mean_compression_ratio:.3f}x")
    print(f"global latent |z|       : {run.global_norm:.6f}")
    print(f"global delta64          : {run.global_delta64:.6e}")
    print("\nNOTE: 1000 GB time is an extrapolation from measured host sample throughput.")
    print("      Full reconstruction information includes the residual stream; 64-D latents are summaries.")
    if ns.gui:
        try: GeometryViewer(cfg, engine.processed_addresses).run()
        except Exception as exc: print(f"GUI unavailable: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
