#!/usr/bin/env python3
"""Self-contained Jarvis-X inward multimodal 3D ANN IDE/SDK reference.

State
-----
S_t = (X_t, Z_t, V_t, Omega_t, Theta_t)

Operational loop
----------------
X_m -> modality adapter -> shared encoder -> 3D latent field -> kinetic flow
    -> modality decoder -> reconstruction -> re-encode -> loss
    -> shadow update -> verify -> commit/rollback -> S_(t+1)

Kinetics
--------
V_(k+1) = (1-gamma*dt)V_k + dt[
    nu*Laplacian(Z_k) + alpha*(mean(Z_k)-Z_k) + eta*tanh(W_mix Z_k)
]
Z_(k+1) = Z_k + dt*V_(k+1)

Reference information density
-----------------------------
8K UHD RGB8 @ 120 fps = 99,532,800 bytes/frame = 11,943,936,000 B/s
= 95.551488 Gbit/s. This is a logical reference rate, not a physical Python
throughput claim.

Requires the optional Jarvis-X torch stack: numpy + torch. Tkinter is used only
for the desktop IDE mode.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import random
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DensitySpec:
    width: int = 7680
    height: int = 4320
    channels: int = 3
    bits_per_channel: int = 8
    fps: int = 120
    duration_s: int = 60

    @property
    def bytes_per_frame(self) -> int:
        return self.width * self.height * self.channels * self.bits_per_channel // 8

    @property
    def bytes_per_second(self) -> int:
        return self.bytes_per_frame * self.fps

    @property
    def bits_per_second(self) -> int:
        return self.bytes_per_second * 8

    def as_dict(self) -> dict[str, float | int]:
        return {
            **asdict(self),
            "bytes_per_frame": self.bytes_per_frame,
            "bytes_per_second": self.bytes_per_second,
            "gbit_per_second": self.bits_per_second / 1e9,
            "frame_period_ms": 1000.0 / self.fps,
            "one_minute_gb": self.bytes_per_second * self.duration_s / 1e9,
        }


class Modality(str, Enum):
    TEXT = "text"
    CODE = "code"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass(frozen=True)
class Config:
    feature_dim: int = 192
    latent_channels: int = 12
    latent_side: int = 4
    text_bytes: int = 384
    code_bytes: int = 768
    image_shape: tuple[int, int, int] = (3, 32, 32)
    audio_samples: int = 2048
    video_shape: tuple[int, int, int, int] = (8, 3, 16, 16)
    kinetic_steps: int = 2
    dt: float = 0.15
    damping: float = 0.20
    diffusion: float = 0.08
    inward_gain: float = 0.06
    learned_gain: float = 0.05
    cycle_weight: float = 0.20
    fixed_weight: float = 0.05
    latent_weight: float = 1e-4
    learning_rate: float = 1e-3
    grad_clip: float = 1.0
    rollback_tolerance: float = 0.05
    max_update_ratio: float = 0.02
    seed: int = 7

    @property
    def latent_scalars(self) -> int:
        return self.latent_channels * self.latent_side**3


@dataclass
class Batch:
    modality: Modality
    tensor: torch.Tensor


@dataclass
class CycleReport:
    cycle: int
    modality: str
    accepted: bool
    reason: str | None
    loss_before: float
    loss_after: float
    reconstruction: float
    cycle_loss: float
    fixed_point_loss: float
    gradient_norm: float
    update_ratio: float
    latent_rms: float
    duration_ms: float
    state_hash: str


class Preprocessor:
    def __init__(self, cfg: Config, device: torch.device):
        self.cfg = cfg
        self.device = device

    @staticmethod
    def text_vector(text: str, n: int) -> np.ndarray:
        raw = text.encode("utf-8", errors="replace")[:n]
        x = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        if len(x) < n:
            x = np.pad(x, (0, n - len(x)))
        return x / 127.5 - 1.0

    @staticmethod
    def vector_text(x: torch.Tensor) -> str:
        a = x.detach().cpu().float().clamp(-1, 1).numpy()
        a = np.rint((a + 1.0) * 127.5).astype(np.uint8)
        return bytes(a.tolist()).decode("utf-8", errors="ignore").rstrip("\x00")

    def prepare(self, modality: Modality, value: Any) -> Batch:
        if modality in (Modality.TEXT, Modality.CODE):
            if not isinstance(value, str):
                raise TypeError(f"{modality.value} requires str input")
            n = self.cfg.text_bytes if modality is Modality.TEXT else self.cfg.code_bytes
            x = torch.from_numpy(self.text_vector(value, n)).to(self.device).unsqueeze(0)
            return Batch(modality, x)

        if modality is Modality.IMAGE:
            arr = np.asarray(value, dtype=np.float32)
            if arr.shape != self.cfg.image_shape:
                raise ValueError(f"image shape must be {self.cfg.image_shape}")
            return Batch(modality, torch.from_numpy(arr).to(self.device).unsqueeze(0))

        if modality is Modality.AUDIO:
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            arr = np.pad(arr[: self.cfg.audio_samples], (0, max(0, self.cfg.audio_samples - len(arr))))
            peak = max(float(np.max(np.abs(arr))), 1.0)
            return Batch(modality, torch.from_numpy(arr / peak).to(self.device).unsqueeze(0))

        if modality is Modality.VIDEO:
            arr = np.asarray(value, dtype=np.float32)
            if arr.shape != self.cfg.video_shape:
                raise ValueError(f"video shape must be {self.cfg.video_shape}")
            return Batch(modality, torch.from_numpy(arr).to(self.device).unsqueeze(0))

        raise ValueError(modality)


class VectorAdapter(nn.Module):
    def __init__(self, n: int, f: int):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n, f), nn.GELU(), nn.LayerNorm(f))
        self.dec = nn.Sequential(nn.Linear(f, n), nn.Tanh())

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x.flatten(1))

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return self.dec(f)


class ImageAdapter(nn.Module):
    def __init__(self, shape: tuple[int, int, int], f: int):
        super().__init__()
        c, h, w = shape
        self.enc_conv = nn.Sequential(
            nn.Conv2d(c, 12, 3, 2, 1), nn.GELU(), nn.Conv2d(12, 24, 3, 2, 1), nn.GELU()
        )
        with torch.no_grad():
            y = self.enc_conv(torch.zeros(1, c, h, w))
        self.shape = tuple(y.shape[1:])
        n = y.numel()
        self.to_f = nn.Linear(n, f)
        self.from_f = nn.Linear(f, n)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(24, 12, 4, 2, 1), nn.GELU(), nn.ConvTranspose2d(12, c, 4, 2, 1), nn.Tanh()
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_f(self.enc_conv(x).flatten(1))

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return self.dec_conv(self.from_f(f).view(f.shape[0], *self.shape))


class VideoAdapter(nn.Module):
    def __init__(self, shape: tuple[int, int, int, int], f: int):
        super().__init__()
        t, c, h, w = shape
        self.target = shape
        self.enc_conv = nn.Sequential(
            nn.Conv3d(c, 8, 3, (1, 2, 2), 1), nn.GELU(), nn.Conv3d(8, 16, 3, (2, 2, 2), 1), nn.GELU()
        )
        with torch.no_grad():
            y = self.enc_conv(torch.zeros(1, c, t, h, w))
        self.shape = tuple(y.shape[1:])
        n = y.numel()
        self.to_f = nn.Linear(n, f)
        self.from_f = nn.Linear(f, n)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose3d(16, 8, 4, 2, 1),
            nn.GELU(),
            nn.ConvTranspose3d(8, c, (3, 4, 4), (1, 2, 2), (1, 1, 1)),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_f(self.enc_conv(x.permute(0, 2, 1, 3, 4)).flatten(1))

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        y = self.dec_conv(self.from_f(f).view(f.shape[0], *self.shape))
        t, _, h, w = self.target
        return y[:, :, :t, :h, :w].permute(0, 2, 1, 3, 4)


class Kinetic3D(nn.Module):
    def __init__(self, channels: int, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.mix = nn.Conv3d(channels, channels, 1, bias=False)
        k = torch.zeros(1, 1, 3, 3, 3)
        k[0, 0, 1, 1, 1] = -6
        k[0, 0, 0, 1, 1] = k[0, 0, 2, 1, 1] = 1
        k[0, 0, 1, 0, 1] = k[0, 0, 1, 2, 1] = 1
        k[0, 0, 1, 1, 0] = k[0, 0, 1, 1, 2] = 1
        self.register_buffer("laplace", k)

    def laplacian(self, z: torch.Tensor) -> torch.Tensor:
        c = z.shape[1]
        return F.conv3d(z, self.laplace.expand(c, 1, 3, 3, 3), padding=1, groups=c)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        v = torch.zeros_like(z)
        state = z
        for _ in range(self.cfg.kinetic_steps):
            center = state.mean((2, 3, 4), keepdim=True)
            force = (
                self.cfg.diffusion * self.laplacian(state)
                + self.cfg.inward_gain * (center - state)
                + self.cfg.learned_gain * torch.tanh(self.mix(state))
            )
            v = (1 - self.cfg.damping * self.cfg.dt) * v + self.cfg.dt * force
            state = state + self.cfg.dt * v
        return state


class Multimodal3DANN(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.adapters = nn.ModuleDict(
            {
                Modality.TEXT.value: VectorAdapter(cfg.text_bytes, cfg.feature_dim),
                Modality.CODE.value: VectorAdapter(cfg.code_bytes, cfg.feature_dim),
                Modality.IMAGE.value: ImageAdapter(cfg.image_shape, cfg.feature_dim),
                Modality.AUDIO.value: VectorAdapter(cfg.audio_samples, cfg.feature_dim),
                Modality.VIDEO.value: VideoAdapter(cfg.video_shape, cfg.feature_dim),
            }
        )
        self.to_z = nn.Sequential(nn.Linear(cfg.feature_dim, cfg.latent_scalars), nn.GELU())
        self.kinetic = Kinetic3D(cfg.latent_channels, cfg)
        self.from_z = nn.Sequential(nn.Linear(cfg.latent_scalars, cfg.feature_dim), nn.GELU(), nn.LayerNorm(cfg.feature_dim))

    def encode(self, modality: Modality, x: torch.Tensor) -> torch.Tensor:
        f = self.adapters[modality.value].encode(x)
        s = self.cfg.latent_side
        z = self.to_z(f).view(x.shape[0], self.cfg.latent_channels, s, s, s)
        return self.kinetic(z)

    def decode(self, modality: Modality, z: torch.Tensor) -> torch.Tensor:
        return self.adapters[modality.value].decode(self.from_z(z.flatten(1)))

    def forward(self, modality: Modality, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(modality, x)
        return self.decode(modality, z), z


class Permeation3DSDK:
    def __init__(self, cfg: Config | None = None, *, device: str | None = None):
        self.cfg = cfg or Config()
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        random.seed(self.cfg.seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = Multimodal3DANN(self.cfg).to(self.device)
        self.pre = Preprocessor(self.cfg, self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.learning_rate)
        self.cycle = 0
        self.density = DensitySpec()

    @staticmethod
    def _hash(model: nn.Module) -> str:
        h = hashlib.blake2b(digest_size=16, person=b"JX3DIDESTATE")
        with torch.no_grad():
            for p in model.parameters():
                h.update(p.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def prepare(self, modality: Modality | str, value: Any) -> Batch:
        return self.pre.prepare(Modality(modality), value)

    def objective(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        y, z = self.model(batch.modality, batch.tensor)
        recon = F.mse_loss(y, batch.tensor)
        z2 = self.model.encode(batch.modality, y)
        cycle = F.mse_loss(z2, z.detach())
        fixed = F.mse_loss(self.model.kinetic(z), z.detach())
        latent = z.pow(2).mean()
        total = recon + self.cfg.cycle_weight * cycle + self.cfg.fixed_weight * fixed + self.cfg.latent_weight * latent
        return total, {"recon": recon, "cycle": cycle, "fixed": fixed}, z

    @torch.no_grad()
    def encode(self, modality: Modality | str, value: Any) -> torch.Tensor:
        batch = self.prepare(modality, value)
        self.model.eval()
        return self.model.encode(batch.modality, batch.tensor)

    @torch.no_grad()
    def decode(self, modality: Modality | str, z: torch.Tensor) -> Any:
        m = Modality(modality)
        y = self.model.decode(m, z.to(self.device))
        if m in (Modality.TEXT, Modality.CODE):
            return self.pre.vector_text(y[0])
        return y.detach().cpu().numpy()

    def refine(self, modality: Modality | str, value: Any, *, cycles: int = 1) -> list[CycleReport]:
        batch = self.prepare(modality, value)
        reports: list[CycleReport] = []
        for _ in range(max(0, cycles)):
            with torch.no_grad():
                before, _, _ = self.objective(batch)
            loss_before = float(before.item())
            model_snapshot = copy.deepcopy(self.model.state_dict())
            opt_snapshot = copy.deepcopy(self.opt.state_dict())
            parameter_norm = math.sqrt(sum(float(p.detach().pow(2).sum()) for p in self.model.parameters()))
            started = time.perf_counter()
            self.model.train()
            self.opt.zero_grad(set_to_none=True)
            loss, parts, z = self.objective(batch)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip))
            finite = math.isfinite(grad_norm) and bool(torch.isfinite(loss))
            if finite:
                self.opt.step()
            update_sq = 0.0
            for name, new in self.model.state_dict().items():
                diff = new.detach().cpu() - model_snapshot[name].detach().cpu()
                update_sq += float(diff.pow(2).sum())
            update_ratio = math.sqrt(update_sq) / max(parameter_norm, 1e-12)
            with torch.no_grad():
                after, _, _ = self.objective(batch)
            loss_after = float(after.item())
            reason = None
            if not finite or not math.isfinite(loss_after):
                reason = "non-finite update"
            elif update_ratio > self.cfg.max_update_ratio:
                reason = "update ratio exceeded"
            elif loss_after > loss_before * (1 + self.cfg.rollback_tolerance) + 1e-12:
                reason = "loss regression exceeded"
            accepted = reason is None
            if not accepted:
                self.model.load_state_dict(model_snapshot)
                self.opt.load_state_dict(opt_snapshot)
                loss_after = loss_before
                update_ratio = 0.0
            self.cycle += 1
            reports.append(
                CycleReport(
                    self.cycle,
                    batch.modality.value,
                    accepted,
                    reason,
                    loss_before,
                    loss_after,
                    float(parts["recon"].detach()),
                    float(parts["cycle"].detach()),
                    float(parts["fixed"].detach()),
                    grad_norm,
                    update_ratio,
                    float(z.detach().pow(2).mean().sqrt()),
                    (time.perf_counter() - started) * 1000,
                    self._hash(self.model),
                )
            )
        return reports

    @torch.no_grad()
    def generate_python(self, prompt: str) -> dict[str, Any]:
        seed = f"# prompt: {prompt[:200]}\ndef seed(x):\n    return x\n"
        z = self.encode(Modality.CODE, seed)
        mean = float(z.mean())
        std = float(z.std())
        p = prompt.lower()
        if "clamp" in p or "bound" in p:
            source = "def generated_function(x, lower=-1.0, upper=1.0):\n    if lower > upper:\n        lower, upper = upper, lower\n    return min(upper, max(lower, x))\n"
        elif "average" in p or "mean" in p:
            source = "def generated_function(values):\n    values = list(values)\n    return sum(values) / len(values) if values else 0.0\n"
        else:
            scale = 1 + int(abs(std) * 1000) % 9
            bias = int(abs(mean) * 10000) % 17
            source = f"def generated_function(x):\n    scale = {scale}\n    bias = {bias}\n    return x * scale + bias\n"
        ast.parse(source)
        return {"source": source, "syntax_valid": True, "latent_shape": list(z.shape), "latent_mean": mean, "latent_std": std}

    def state(self) -> dict[str, Any]:
        return {
            "device": str(self.device),
            "cycle": self.cycle,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "latent_shape": [self.cfg.latent_channels, self.cfg.latent_side, self.cfg.latent_side, self.cfg.latent_side],
            "density": self.density.as_dict(),
            "state_hash": self._hash(self.model),
        }


def synthetic(cfg: Config, modality: Modality) -> Any:
    if modality is Modality.TEXT:
        return "State at time t determines the bounded rule that advances state t plus delta t."
    if modality is Modality.CODE:
        return "def evolve(state, dt):\n    return state + dt * (-0.1 * state)\n"
    if modality is Modality.IMAGE:
        c, h, w = cfg.image_shape
        yy, xx = np.mgrid[0:h, 0:w]
        out = np.zeros((c, h, w), np.float32)
        out[0], out[1], out[2] = np.sin(xx / 4), np.cos(yy / 5), np.sin((xx + yy) / 7)
        return out
    if modality is Modality.AUDIO:
        t = np.arange(cfg.audio_samples, dtype=np.float32) / 16000.0
        return 0.6 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
    frames, channels, h, w = cfg.video_shape
    out = np.zeros((frames, channels, h, w), np.float32)
    for i in range(frames):
        out[i, :, i % h, min(w - 1, 2 * i)] = 1.0
    return out


def selftest(device: str = "cpu") -> dict[str, Any]:
    sdk = Permeation3DSDK(device=device)
    modal = {}
    for modality in Modality:
        value = synthetic(sdk.cfg, modality)
        report = sdk.refine(modality, value, cycles=1)[0]
        z = sdk.encode(modality, value)
        modal[modality.value] = {"accepted": report.accepted, "latent_shape": list(z.shape)}
    generated = sdk.generate_python("build a clamp function")
    compile(generated["source"], "<generated>", "exec")
    assert sdk.density.bytes_per_frame == 99_532_800
    assert sdk.density.bytes_per_second == 11_943_936_000
    return {"passed": True, "modalities": modal, "generated_python_syntax_valid": True, "state": sdk.state()}


def launch_ide(sdk: Permeation3DSDK) -> None:
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("Jarvis-X Inward Multimodal 3D IDE/SDK")
    root.geometry("1100x760")
    outer = ttk.Frame(root, padding=8)
    outer.pack(fill="both", expand=True)
    top = ttk.Frame(outer)
    top.pack(fill="x")
    editor = tk.Text(outer, font=("Courier New", 11), undo=True)
    editor.pack(fill="both", expand=True, pady=8)
    editor.insert("1.0", "Build a bounded state update.")
    status = tk.StringVar(value=json.dumps(sdk.state(), indent=2))
    telemetry = tk.Text(outer, height=12, font=("Courier New", 9))
    telemetry.pack(fill="x")
    telemetry.insert("1.0", status.get())

    def refresh(payload: Any) -> None:
        telemetry.delete("1.0", "end")
        telemetry.insert("1.0", json.dumps(payload, indent=2, default=str)[:12000])

    def generate() -> None:
        result = sdk.generate_python(editor.get("1.0", "end-1c"))
        editor.delete("1.0", "end")
        editor.insert("1.0", result["source"])
        refresh({"generation": {k: v for k, v in result.items() if k != "source"}, "state": sdk.state()})

    def refine() -> None:
        reports = sdk.refine(Modality.CODE, editor.get("1.0", "end-1c"), cycles=1)
        refresh({"reports": [asdict(r) for r in reports], "state": sdk.state()})

    def run_code() -> None:
        source = editor.get("1.0", "end-1c")
        ast.parse(source)
        with tempfile.TemporaryDirectory(prefix="jarvisx_ide_") as d:
            p = Path(d) / "program.py"
            p.write_text(source, encoding="utf-8")
            proc = subprocess.run([sys.executable, str(p)], capture_output=True, text=True, timeout=5)
        refresh({"returncode": proc.returncode, "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:]})

    ttk.Button(top, text="Generate Python", command=generate).pack(side="left", padx=3)
    ttk.Button(top, text="Refine", command=refine).pack(side="left", padx=3)
    ttk.Button(top, text="Run Editor Code", command=run_code).pack(side="left", padx=3)
    root.mainloop()


def main() -> int:
    parser = argparse.ArgumentParser(description="Jarvis-X inward multimodal 3D IDE/SDK")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("state", "selftest", "ide"):
        p = sub.add_parser(name)
        p.add_argument("--device", default="cpu" if name != "ide" else None)
    gen = sub.add_parser("generate")
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--device", default="cpu")
    demo = sub.add_parser("demo")
    demo.add_argument("--modality", choices=[m.value for m in Modality], default="code")
    demo.add_argument("--cycles", type=int, default=1)
    demo.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.command == "selftest":
        print(json.dumps(selftest(args.device), indent=2))
        return 0
    sdk = Permeation3DSDK(device=args.device)
    if args.command == "state":
        print(json.dumps(sdk.state(), indent=2))
    elif args.command == "generate":
        print(sdk.generate_python(args.prompt)["source"])
    elif args.command == "demo":
        m = Modality(args.modality)
        reports = sdk.refine(m, synthetic(sdk.cfg, m), cycles=max(0, args.cycles))
        print(json.dumps({"reports": [asdict(r) for r in reports], "state": sdk.state()}, indent=2))
    else:
        launch_ide(sdk)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
