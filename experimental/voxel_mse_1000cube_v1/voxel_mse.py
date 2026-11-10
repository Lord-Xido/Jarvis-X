#!/usr/bin/env python3
"""JARVIS-X 3D voxel MSE objective: tiled, differentiable CPU reference.

The model is a small trainable spatial autoencoder. It processes *independent*
3D tiles, so tile-edge context differs from a full-volume convolution. Loss
normalization nonetheless counts every real voxel exactly once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch import nn


def validate_config(edge: int, tile: int) -> None:
    if not (1 <= edge <= 1_000_000):
        raise ValueError("edge must be in [1, 1000000]")
    if tile < 2 or tile % 2 or tile > 256:
        raise ValueError("tile must be an even integer in [2, 256]")
    if edge ** 3 > 2**63 - 1:
        raise ValueError("voxel count would overflow int64")


def iter_origins(edge: int, tile: int) -> Iterator[tuple[int, int, int]]:
    validate_config(edge, tile)
    for x in range(0, edge, tile):
        for y in range(0, edge, tile):
            for z in range(0, edge, tile):
                yield x, y, z


class UInt8Cube:
    """Read a 3D raw uint8 volume without allocating the complete cube.

    Without `raw`, procedurally synthesize spatially structured byte voxels.
    A raw input is required to match exactly edge^3 bytes; it is mmap backed.
    """

    def __init__(self, edge: int, raw: str | None = None) -> None:
        if not 1 <= edge <= 1_000_000:
            raise ValueError("invalid cube edge")
        self.edge = edge
        self.raw = None
        if raw is not None:
            path = Path(raw)
            if path.stat().st_size != edge**3:
                raise ValueError(f"raw file must contain exactly {edge**3} bytes")
            self.raw = np.memmap(path, dtype=np.uint8, mode="r", shape=(edge, edge, edge))

    def read(self, xyz: tuple[int, int, int], tile: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        x, y, z = xyz
        if not all(0 <= q < self.edge for q in xyz):
            raise ValueError("tile origin out of bounds")
        nx, ny, nz = (min(tile, self.edge - q) for q in xyz)
        block = np.zeros((tile, tile, tile), dtype=np.uint8)
        if self.raw is not None:
            block[:nx, :ny, :nz] = self.raw[x:x+nx, y:y+ny, z:z+nz]
        else:
            xx = np.arange(x, x+nx, dtype=np.int64)[:, None, None]
            yy = np.arange(y, y+ny, dtype=np.int64)[None, :, None]
            zz = np.arange(z, z+nz, dtype=np.int64)[None, None, :]
            # Deterministic 3D signal: gradients, cross terms, and stepped features.
            values = (13*xx + 7*yy + 3*zz + (xx//5)*19 + (yy//7)*11 + (zz//3)*5) % 256
            block[:nx, :ny, :nz] = values.astype(np.uint8)
        mask = np.zeros((tile, tile, tile), dtype=np.float32)
        mask[:nx, :ny, :nz] = 1.0
        # torch.from_numpy arrays are private contiguous copies, avoiding mmap writable warnings.
        return (torch.from_numpy(block.astype(np.float32).copy())[None, None] / 255.0,
                torch.from_numpy(mask)[None, None], nx * ny * nz)


class SpatialAutoencoder(nn.Module):
    """Trainable 3D encoder/decoder; a research model, NOT a codec bitstream."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv3d(1, 2, 3, stride=2, padding=1), nn.SiLU())
        self.decoder = nn.Sequential(nn.ConvTranspose3d(2, 1, 4, stride=2, padding=1), nn.Sigmoid())

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(volume))


def tiled_loss(model: nn.Module, cube: UInt8Cube, tile: int, backward: bool = False) -> tuple[float, int]:
    """Exact global MSE denominator, including edge tiles. No input cube allocation.

    If backward, add gradient of each tile's SSE / N to parameter .grad.
    Do not wrap backward=True in torch.no_grad().
    """
    validate_config(cube.edge, tile)
    total = cube.edge**3
    sse = 0.0
    n = 0
    for xyz in iter_origins(cube.edge, tile):
        x, mask, count = cube.read(xyz, tile)
        y = model(x)
        if y.shape != x.shape:
            raise ValueError("model reconstruction shape mismatch")
        sq = ((x - y).square() * mask).sum()
        sse += float(sq.detach().double())
        n += count
        if backward:
            (sq / total).backward()
    if n != total:
        raise AssertionError(f"voxel accounting mismatch: {n} != {total}")
    return sse / n, n


def model_digest(model: nn.Module) -> str:
    h = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        h.update(key.encode("utf-8") + b"\x00")
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def guarded_step(model: nn.Module, cube: UInt8Cube, tile: int,
                 lr: float, candidates: int = 4) -> dict:
    if not (math.isfinite(lr) and lr > 0) or candidates < 1:
        raise ValueError("lr must be positive finite and candidates >= 1")
    model.zero_grad(set_to_none=True)
    base, n = tiled_loss(model, cube, tile, backward=True)
    baseline = {k: p.detach().clone() for k, p in model.state_dict().items()}
    gradients = [p.grad.detach().clone() for p in model.parameters()]
    if not all(torch.isfinite(g).all().item() for g in gradients):
        raise ValueError("nonfinite gradient")
    accepted = False
    best = base
    used_lr = None
    for i in range(candidates):
        rate = lr * (0.5 ** i)
        with torch.no_grad():
            for param, grad in zip(model.parameters(), gradients):
                param.add_(grad, alpha=-rate)
        with torch.no_grad():
            candidate, check_n = tiled_loss(model, cube, tile)
        if check_n == n and math.isfinite(candidate) and candidate < base:
            accepted, best, used_lr = True, candidate, rate
            break
        model.load_state_dict(baseline)
    if not accepted:
        model.load_state_dict(baseline)
    model.zero_grad(set_to_none=True)
    return {"old_mse": base, "new_mse": best, "voxel_count": n,
            "status": "commit" if accepted else "rollback", "step_size": used_lr,
            "model_sha256": model_digest(model)}


def run() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--edge", type=int, default=16, help="cube edge, e.g. 1000")
    ap.add_argument("--tile", type=int, default=8, help="even local tile edge")
    ap.add_argument("--raw", type=str, help="optional edge^3-byte uint8 raw cube")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()
    validate_config(args.edge, args.tile)
    if not 0 <= args.steps <= 1000:
        ap.error("steps must be in [0, 1000]")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    model = SpatialAutoencoder()
    cube = UInt8Cube(args.edge, args.raw)
    with torch.no_grad():
        initial, n = tiled_loss(model, cube, args.tile)
    print(json.dumps({"event": "initial", "edge": args.edge, "tile": args.tile,
                      "voxels": n, "mse_normalized": initial,
                      "raw_input_bytes": n, "model_sha256": model_digest(model)}, sort_keys=True))
    for step in range(args.steps):
        print(json.dumps({"event": "step", "iteration": step + 1,
                          **guarded_step(model, cube, args.tile, args.lr)}, sort_keys=True))


if __name__ == "__main__":
    run()
