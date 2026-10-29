"""Opt-in trainable Phi_M 3D convolutional autoencoder (research-only).

This module wraps jarvisx.dr_moagi_phi_m_3d, which remains the mathematical
source of truth. Fits a *lossy* neural reconstruction on synthetic 3D fields.
No live system, codec, or runtime is automatically altered.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:
    raise ImportError("Optional trainable Φ_M module requires torch: pip install -e '.[phi_m_ae]'") from exc

from .dr_moagi_phi_m_3d import PhiMConfig, phi_m, numerical_self_test


@dataclass(frozen=True)
class TrainConfig:
    grid: int = 16
    depth: int = 1
    latent: int = 48
    train_samples: int = 48
    val_samples: int = 12
    batch: int = 8
    steps: int = 1000
    lr: float = 3e-3
    min_lr: float = 2e-5
    warmup: float = 0.08
    grad_clip: float = 1.0
    ema_decay: float = 0.99
    eval_every: int = 25
    seed: int = 2026
    device: str = "auto"

    def __post_init__(self) -> None:
        if type(self.grid) is not int or self.grid not in (8, 16):
            raise ValueError("grid must be 8 or 16")
        for name, low, high in (
            ("depth", 1, 16), ("latent", 2, 256), ("train_samples", 4, 2048),
            ("val_samples", 2, 512), ("batch", 1, 128), ("steps", 1, 10000),
            ("eval_every", 1, 1000), ("seed", 0, 2**32 - 1),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} out of range")
        for name, low_f, high_f in (
            ("lr", 1e-8, 1.0), ("min_lr", 1e-9, self.lr),
            ("warmup", 0.0, 1.0), ("grad_clip", 1e-6, 100.0),
            ("ema_decay", 0.0, 0.999999),
        ):
            float_value = getattr(self, name)
            if isinstance(float_value, bool) or not isinstance(float_value, (int, float)) or not math.isfinite(float_value) or not low_f <= float_value <= high_f:
                raise ValueError(f"{name} must be finite and bounded")
        if self.batch > self.train_samples:
            raise ValueError("batch exceeds training set")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu, cuda")


class FieldGenerator:
    def __init__(self, grid: int, depth: int, seed: int) -> None:
        if grid not in (8, 16) or not 1 <= depth <= 16:
            raise ValueError("invalid field geometry")
        self.grid, self.depth = grid, depth
        self.rng = np.random.default_rng(seed)
        a = np.linspace(-1.0, 1.0, grid)
        self.points = np.stack(np.meshgrid(a, a, a, indexing="ij"), axis=-1).reshape(-1, 3)

    def sample(self, count: int) -> np.ndarray:
        if type(count) is not int or not 1 <= count <= 2048:
            raise ValueError("invalid sample count")
        fields = np.empty((count, self.grid, self.grid, self.grid, 3), np.float32)
        for i in range(count):
            p = replace(PhiMConfig(),
                        amplitude=float(self.rng.uniform(0.47, 0.63)),
                        modulation=float(self.rng.uniform(0.30, 0.40)))
            t = float(self.rng.uniform(0, 3))
            state = self.points.copy()
            for _ in range(self.depth):
                state = phi_m(state, t, p)
            fields[i] = state.reshape(self.grid, self.grid, self.grid, 3)
        return fields


def field_tensor(data: np.ndarray) -> Any:
    a = np.asarray(data)
    if a.ndim != 5 or a.shape[-1] != 3 or a.shape[1] not in (8, 16) or a.shape[1:4] != (a.shape[1],)*3:
        raise ValueError("fields must be (N,G,G,G,3) with G=8 or 16")
    if not np.isfinite(a).all():
        raise ValueError("fields must be finite")
    return torch.from_numpy(np.ascontiguousarray(a.transpose(0, 4, 1, 2, 3), dtype=np.float32))


class PhiMAutoEncoder(nn.Module):
    def __init__(self, grid: int = 16, latent: int = 48) -> None:
        super().__init__()
        if grid not in (8, 16) or type(latent) is not int or not 2 <= latent <= 256:
            raise ValueError("invalid AE dimensions")
        self.grid, self.latent = grid, latent
        self.inner = grid // 8
        self.down = nn.Sequential(
            nn.Conv3d(3, 12, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv3d(12, 24, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv3d(24, 32, 3, stride=2, padding=1), nn.GELU(),
        )
        self.enc = nn.Linear(32 * self.inner**3, latent)
        self.dec = nn.Linear(latent, 32 * self.inner**3)
        self.up = nn.Sequential(
            nn.ConvTranspose3d(32, 24, 3, stride=2, padding=1, output_padding=1), nn.GELU(),
            nn.ConvTranspose3d(24, 12, 3, stride=2, padding=1, output_padding=1), nn.GELU(),
            nn.ConvTranspose3d(12, 3, 3, stride=2, padding=1, output_padding=1), nn.Tanh(),
        )

    def encode(self, x: Any) -> Any:
        if x.ndim != 5 or tuple(x.shape[1:]) != (3, self.grid, self.grid, self.grid):
            raise ValueError("input field does not match model grid")
        return self.enc(self.down(x).flatten(1))

    def decode(self, z: Any) -> Any:
        if z.ndim != 2 or z.shape[1] != self.latent:
            raise ValueError("invalid latent geometry")
        return self.up(self.dec(z).reshape(-1, 32, self.inner, self.inner, self.inner))

    def forward(self, x: Any) -> tuple[Any, Any]:
        z = self.encode(x)
        return self.decode(z), z


def _device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return ("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested


def _evaluate(model: PhiMAutoEncoder, data: Any, batch: int, device: str) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for chunk in data.split(batch):
            x = chunk.to(device)
            y, _ = model(x)
            total += float(F.mse_loss(y, x, reduction="sum").item())
    return float(total / int(data.numel()))


def _weights(model: Any) -> dict[str, Any]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def train(config: TrainConfig, out: Path) -> dict[str, Any]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    device = _device(config.device)
    x_train = field_tensor(FieldGenerator(config.grid, config.depth, config.seed).sample(config.train_samples))
    x_val = field_tensor(FieldGenerator(config.grid, config.depth, config.seed + 65537).sample(config.val_samples))
    model = PhiMAutoEncoder(config.grid, config.latent).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    ema = _weights(model)
    best = None
    best_val = float("inf")
    history = []
    warmup = max(1, round(config.steps * config.warmup))
    generator = torch.Generator().manual_seed(config.seed + 99)
    for step in range(config.steps):
        ids = torch.randint(config.train_samples, (config.batch,), generator=generator)
        x = x_train[ids].to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        y, _ = model(x)
        loss = F.mse_loss(y, x)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite training loss")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=True)
        if step < warmup:
            lr = config.lr * (step+1)/warmup
        else:
            phase = min(1.0, (step-warmup) / max(1, config.steps-warmup-1))
            lr = config.min_lr + (config.lr-config.min_lr)*0.5*(1+math.cos(math.pi*phase))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        with torch.no_grad():
            for k,v in model.state_dict().items():
                ema[k].mul_(config.ema_decay).add_(v.detach().cpu(), alpha=1-config.ema_decay)
        if (step+1) % config.eval_every == 0 or step == 0 or step+1 == config.steps:
            validation = _evaluate(model, x_val, config.batch, device)
            if not math.isfinite(validation):
                raise RuntimeError("nonfinite validation")
            if validation < best_val - 1e-8:
                best_val, best = validation, _weights(model)
            history.append(dict(step=step+1, train_mse=float(loss.detach()),
                                val_mse=validation, lr=lr, grad_norm=float(grad_norm.detach())))
    if best is None:
        raise RuntimeError("no validated checkpoint")
    model.load_state_dict(best)
    val_best = _evaluate(model, x_val, config.batch, device)
    model.load_state_dict(ema)
    val_ema = _evaluate(model, x_val, config.batch, device)
    choice = "ema" if val_ema < val_best else "best_validation"
    if choice != "ema":
        model.load_state_dict(best)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(_weights(model), out / "model.pt")
    (out / "config.json").write_text(json.dumps(dict(format="phi-m-ae-v1", grid=config.grid,
        latent=config.latent, depth=config.depth), indent=2))
    with torch.no_grad():
        y,z = model(x_val[:1].to(device))
        original = x_val[0].numpy().transpose(1,2,3,0)
        decoded = y[0].cpu().numpy().transpose(1,2,3,0)
        code = z[0].cpu().numpy()
    np.save(out / "demo_field.npy", original)
    np.save(out / "demo_recon.npy", decoded)
    np.save(out / "demo_code.npy", code)
    report = dict(status="shadow_only", config=asdict(config), updates=config.steps,
        parameters=sum(p.numel() for p in model.parameters()), selected=choice,
        validation_mse=min(val_best,val_ema), baseline_zero_mse=float(torch.mean(x_val**2)),
        demo_mse=float(np.mean((original-decoded)**2)), history=history)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report


def load_model(path: Path) -> PhiMAutoEncoder:
    info = json.loads((path.parent / "config.json").read_text())
    if info.get("format") != "phi-m-ae-v1" or type(info["grid"]) is not int or type(info["latent"]) is not int:
        raise ValueError("model format or dimensions unsupported")
    model = PhiMAutoEncoder(info["grid"], info["latent"])
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or any(not isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("expected tensor state_dict")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def encode_field(model: PhiMAutoEncoder, field: np.ndarray) -> Any:
    x = np.asarray(field)
    if x.shape != (model.grid,)*3 + (3,) or not np.isfinite(x).all():
        raise ValueError("field does not match model geometry")
    return np.asarray(model.encode(field_tensor(x[None]))[0].numpy(), dtype=np.float32)


@torch.no_grad()
def decode_code(model: PhiMAutoEncoder, code: np.ndarray) -> Any:
    z = np.asarray(code, dtype=np.float32)
    if z.shape != (model.latent,) or not np.isfinite(z).all():
        raise ValueError("invalid latent vector")
    out = model.decode(torch.from_numpy(np.ascontiguousarray(z[None])))
    return np.ascontiguousarray(out[0].numpy().transpose(1,2,3,0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train","audit","encode","decode","query"), default="train")
    parser.add_argument("--out", type=Path, default=Path("./phi_m_unified"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--field", type=Path)
    parser.add_argument("--code", type=Path)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--latent", type=int, default=48)
    parser.add_argument("--train-samples", type=int, default=48)
    parser.add_argument("--val-samples", type=int, default=12)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--points", default="0,0,0")
    parser.add_argument("--time", type=float, default=1.5)
    args = parser.parse_args()
    if args.mode == "train":
        report = train(TrainConfig(grid=args.grid, latent=args.latent, depth=args.depth,
            train_samples=args.train_samples, val_samples=args.val_samples,
            batch=args.batch, steps=args.iters), args.out)
    elif args.mode == "audit":
        report = numerical_self_test()
        if not report["passed"]:
            raise SystemExit(1)
    elif args.mode == "query":
        pts = np.array([[float(v) for v in p.split(",")] for p in args.points.split(";")])
        if pts.ndim != 2 or pts.shape[1] != 3 or not 1 <= args.depth <= 16:
            raise ValueError("invalid 3D query coordinates/depth")
        state = pts
        for _ in range(args.depth):
            state = phi_m(state, args.time)
        report = dict(mapped=state.tolist(), depth=args.depth, time=args.time)
    elif args.mode == "encode":
        if args.model is None or args.field is None:
            raise ValueError("encode requires --model and --field")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        code = encode_field(load_model(args.model), np.load(args.field, allow_pickle=False))
        np.save(args.out, code)
        report = dict(code_shape=list(code.shape), path=str(args.out))
    else:
        if args.model is None or args.code is None:
            raise ValueError("decode requires --model and --code")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        field = decode_code(load_model(args.model), np.load(args.code, allow_pickle=False))
        np.save(args.out, field)
        report = dict(field_shape=list(field.shape), path=str(args.out))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
