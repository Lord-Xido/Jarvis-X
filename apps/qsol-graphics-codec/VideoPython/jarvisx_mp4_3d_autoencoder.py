#!/usr/bin/env python3
"""Jarvis-X 3D MP4 autoencoder reference pipeline.

MP4 -> RGB clips [N,3,T,H,W] -> Conv3D encoder -> latent NPZ
    -> Conv3D decoder -> reconstructed MP4 -> optional source-audio remux.

The latent archive is self-contained for the reference simulation: it stores the
float16 latent tensor, metadata, training history, and decoder/model weights.
In a deployed codec the decoder weights should normally be shared and excluded
from the per-video payload when computing compression ratio.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class VideoMeta:
    fps: float
    original_width: int
    original_height: int
    processed_width: int
    processed_height: int
    frame_count: int
    clip_len: int
    padded_frame_count: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_synthetic_mp4(
    path: str,
    width: int = 96,
    height: int = 96,
    frames: int = 64,
    fps: float = 24.0,
) -> None:
    """Create a deterministic moving-pattern MP4 for end-to-end self-test."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create synthetic MP4: {path}")

    yy, xx = np.mgrid[0:height, 0:width]
    for t in range(frames):
        phase = 2.0 * np.pi * t / max(frames, 1)
        cx = int(width * (0.5 + 0.28 * np.sin(phase)))
        cy = int(height * (0.5 + 0.25 * np.cos(phase * 1.3)))
        radius = max(4, int(min(width, height) * (0.11 + 0.03 * np.sin(2 * phase))))

        r = 0.5 + 0.5 * np.sin(0.08 * xx + phase)
        g = 0.5 + 0.5 * np.sin(0.07 * yy - 1.4 * phase)
        b = 0.5 + 0.5 * np.sin(0.05 * (xx + yy) + 0.6 * phase)
        rgb = np.stack([r, g, b], axis=-1)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
        rgb[mask, 0] = 1.0
        rgb[mask, 1] *= 0.2
        rgb[mask, 2] = 0.15

        frame = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def read_video_rgb(
    path: str,
    width: int,
    height: int,
    max_frames: int = 0,
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 24.0
    original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames: List[np.ndarray] = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if max_frames > 0 and len(frames) >= max_frames:
            break
        frame_bgr = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        raise RuntimeError(f"No decodable frames in {path}")
    return np.stack(frames, axis=0), fps, (original_w, original_h)


def write_video_rgb(path: str, frames: np.ndarray, fps: float) -> None:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected [T,H,W,3], got {frames.shape}")
    h, w = frames.shape[1:3]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output MP4: {path}")
    for rgb in frames:
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    writer.release()


def mux_original_audio(video_only: str, original: str, output: str) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video_only),
            "-i",
            str(original),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(output),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.returncode == 0


def pad_and_make_clips(frames_u8: np.ndarray, clip_len: int) -> Tuple[np.ndarray, int]:
    if clip_len % 4:
        raise ValueError("clip_len must be divisible by 4")
    frame_count = frames_u8.shape[0]
    pad_count = (-frame_count) % clip_len
    if pad_count:
        frames_u8 = np.concatenate(
            [frames_u8, np.repeat(frames_u8[-1:], pad_count, axis=0)], axis=0
        )

    x = frames_u8.astype(np.float32) / 255.0
    n = x.shape[0] // clip_len
    x = x.reshape(n, clip_len, x.shape[1], x.shape[2], 3)
    x = np.transpose(x, (0, 4, 1, 2, 3))
    return x.astype(np.float32), pad_count


def clips_to_frames(clips: np.ndarray, frame_count: int) -> np.ndarray:
    x = np.transpose(clips, (0, 2, 3, 4, 1))
    x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])[:frame_count]
    return np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)


class ClipDataset(Dataset):
    def __init__(self, clips: np.ndarray):
        self.clips = torch.from_numpy(clips)

    def __len__(self) -> int:
        return self.clips.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.clips[index]


class VideoAutoencoder3D(nn.Module):
    """X[B,3,T,H,W] -> Z[B,Cz,T/4,H/4,W/4] -> X_hat."""

    def __init__(self, latent_ch: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(16, 32, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv3d(32, latent_ch, 4, stride=2, padding=1),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(latent_ch, 32, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose3d(32, 16, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv3d(16, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z


def temporal_difference_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if x.shape[2] < 2:
        return torch.zeros((), device=x.device)
    dx = x[:, :, 1:] - x[:, :, :-1]
    dh = x_hat[:, :, 1:] - x_hat[:, :, :-1]
    return torch.mean((dh - dx) ** 2)


def train_autoencoder(
    model: VideoAutoencoder3D,
    clips: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    temporal_weight: float,
) -> List[float]:
    loader = DataLoader(
        ClipDataset(clips),
        batch_size=max(1, min(batch_size, len(clips))),
        shuffle=True,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    mse = nn.MSELoss()
    history: List[float] = []

    model.train()
    for epoch in range(epochs):
        running = 0.0
        count = 0
        for x in loader:
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            x_hat, _ = model(x)
            loss = mse(x_hat, x) + temporal_weight * temporal_difference_loss(x_hat, x)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach()) * x.shape[0]
            count += x.shape[0]
        epoch_loss = running / max(count, 1)
        history.append(epoch_loss)
        print(f"Epoch {epoch + 1:03d}/{epochs:03d} | loss={epoch_loss:.7f}")
    return history


@torch.no_grad()
def encode_clips(
    model: VideoAutoencoder3D,
    clips: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    chunks: List[np.ndarray] = []
    for start in range(0, len(clips), batch_size):
        x = torch.from_numpy(clips[start : start + batch_size]).to(device)
        chunks.append(model.encode(x).cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


@torch.no_grad()
def decode_latents(
    model: VideoAutoencoder3D,
    latents: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    chunks: List[np.ndarray] = []
    for start in range(0, len(latents), batch_size):
        z = torch.from_numpy(latents[start : start + batch_size]).to(device)
        chunks.append(model.decode(z).cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def save_latent_archive(
    path: str,
    latents: np.ndarray,
    meta: VideoMeta,
    model: VideoAutoencoder3D,
    history: List[float],
) -> None:
    payload: Dict[str, np.ndarray] = {
        "latent": latents.astype(np.float16),
        "meta_json": np.frombuffer(json.dumps(asdict(meta)).encode(), dtype=np.uint8),
        "train_history": np.asarray(history, dtype=np.float32),
    }
    for name, tensor in model.state_dict().items():
        payload["weight__" + name.replace(".", "__")] = (
            tensor.detach().cpu().numpy().astype(np.float32)
        )
    np.savez_compressed(path, **payload)


def load_latent_archive(path: str, latent_ch: int, device: torch.device):
    data = np.load(path, allow_pickle=False)
    latent = data["latent"].astype(np.float32)
    meta = VideoMeta(**json.loads(bytes(data["meta_json"].tolist()).decode()))
    model = VideoAutoencoder3D(latent_ch).to(device)
    state = {}
    for key in data.files:
        if key.startswith("weight__"):
            name = key[len("weight__") :].replace("__", ".")
            state[name] = torch.from_numpy(data[key])
    model.load_state_dict(state, strict=True)
    model.eval()
    return latent, meta, model


def mse_psnr(a_u8: np.ndarray, b_u8: np.ndarray) -> Tuple[float, float]:
    a = a_u8.astype(np.float32) / 255.0
    b = b_u8.astype(np.float32) / 255.0
    mse = float(np.mean((a - b) ** 2))
    return mse, float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse)


def run_pipeline(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    if args.width % 4 or args.height % 4 or args.clip_len % 4:
        raise ValueError("width, height and clip-len must be divisible by 4")

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    input_path = Path(args.input) if args.input else Path("synthetic_input.mp4")

    if args.synthetic:
        generate_synthetic_mp4(
            str(input_path),
            max(args.width, 64),
            max(args.height, 64),
            args.synthetic_frames,
            args.synthetic_fps,
        )
        print(f"Synthetic source created: {input_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input MP4 not found: {input_path}")

    frames, fps, original_size = read_video_rgb(
        str(input_path), args.width, args.height, args.max_frames
    )
    clips, pad_count = pad_and_make_clips(frames, args.clip_len)
    meta = VideoMeta(
        fps=fps,
        original_width=original_size[0],
        original_height=original_size[1],
        processed_width=args.width,
        processed_height=args.height,
        frame_count=len(frames),
        clip_len=args.clip_len,
        padded_frame_count=len(frames) + pad_count,
    )

    print(f"Device: {device}")
    print(f"Input X: {clips.shape}")
    model = VideoAutoencoder3D(args.latent_ch).to(device)
    history = train_autoencoder(
        model,
        clips,
        device,
        args.epochs,
        args.batch_size,
        args.lr,
        args.temporal_weight,
    )

    latent = encode_clips(model, clips, device, args.batch_size)
    save_latent_archive(args.latent, latent, meta, model, history)

    latent_loaded, loaded_meta, decoder = load_latent_archive(
        args.latent, args.latent_ch, device
    )
    reconstructed_clips = decode_latents(
        decoder, latent_loaded, device, args.batch_size
    )
    reconstructed_frames = clips_to_frames(
        reconstructed_clips, loaded_meta.frame_count
    )
    mse, psnr = mse_psnr(frames, reconstructed_frames)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.preserve_audio and not args.synthetic:
        with tempfile.TemporaryDirectory() as td:
            video_only = Path(td) / "reconstructed_video_only.mp4"
            write_video_rgb(str(video_only), reconstructed_frames, fps)
            if not mux_original_audio(str(video_only), str(input_path), str(output_path)):
                shutil.copyfile(video_only, output_path)
                print("Audio remux unavailable; wrote video-only reconstruction.")
    else:
        write_video_rgb(str(output_path), reconstructed_frames, fps)

    raw_bytes = frames.nbytes
    latent_f32_bytes = latent.nbytes
    latent_f16_bytes = latent_loaded.astype(np.float16).nbytes
    archive_bytes = os.path.getsize(args.latent)

    print("\n" + "=" * 72)
    print("JARVIS-X 3D MP4 AUTOENCODER RESULT")
    print("=" * 72)
    print(f"Input tensor            : {clips.shape}")
    print(f"Latent tensor           : {latent.shape}")
    print(f"Reconstructed frames    : {reconstructed_frames.shape}")
    print(f"MSE                     : {mse:.8f}")
    print(f"PSNR                    : {psnr:.3f} dB")
    print(f"Raw RGB bytes           : {raw_bytes:,}")
    print(f"Latent float32 bytes    : {latent_f32_bytes:,}")
    print(f"Latent float16 bytes    : {latent_f16_bytes:,}")
    print(f"Raw / latent-f32 ratio  : {raw_bytes / max(latent_f32_bytes, 1):.2f}x")
    print(f"Raw / latent-f16 ratio  : {raw_bytes / max(latent_f16_bytes, 1):.2f}x")
    print(f"Self-contained NPZ size : {archive_bytes:,} bytes")
    print(f"Output MP4              : {output_path}")
    print("=" * 72)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Jarvis-X 3D MP4 autoencoder")
    p.add_argument("--input", default=None)
    p.add_argument("--output", default="reconstructed.mp4")
    p.add_argument("--latent", default="encoded_latent.npz")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--synthetic-frames", type=int, default=64)
    p.add_argument("--synthetic-fps", type=float, default=24.0)
    p.add_argument("--width", type=int, default=96)
    p.add_argument("--height", type=int, default=96)
    p.add_argument("--clip-len", type=int, default=8)
    p.add_argument("--latent-ch", type=int, default=8)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--temporal-weight", type=float, default=0.15)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--preserve-audio",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return p


if __name__ == "__main__":
    run_pipeline(build_parser().parse_args())
