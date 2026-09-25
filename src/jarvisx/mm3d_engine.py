"""Trainable 3D multimodal autoencoding and generation backend for Jarvis-X.

This module is an executable reference architecture, not a claim of solved AGI.
It maps text, code, image, audio, and video into a shared cubic latent field,
refines that field with stable 3D operators and recurrent memory, and decodes
all supported modalities from the same internal state.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MM3DConfig:
    latent_channels: int = 64
    grid: int = 6
    sequence_length: int = 128
    vocab_size: int = 256
    image_size: int = 96
    audio_samples: int = 4096
    video_frames: int = 8
    video_size: int = 48
    environment_channels: int = 4
    action_dim: int = 32
    echo_weight: float = 0.5
    echo_depth: int = 4
    echo_mix: float = 0.25
    refinement_depth: int = 6
    attention_heads: int = 8
    diffusion_max: float = 0.12
    cycle_weight: float = 0.05
    latent_weight: float = 1.0e-4

    def validate(self) -> None:
        if self.grid < 2:
            raise ValueError("grid must be >= 2")
        if self.latent_channels % 8:
            raise ValueError("latent_channels must be divisible by 8")
        if self.latent_channels % self.attention_heads:
            raise ValueError("latent_channels must be divisible by attention_heads")
        if self.environment_channels < 1:
            raise ValueError("environment_channels must be >= 1")
        if self.action_dim < 2:
            raise ValueError("action_dim must be >= 2")
        if not 0.0 < self.echo_weight < 1.0:
            raise ValueError("echo_weight must satisfy 0 < echo_weight < 1")
        if self.echo_depth < 0:
            raise ValueError("echo_depth must be >= 0")
        if not 0.0 <= self.echo_mix <= 1.0:
            raise ValueError("echo_mix must satisfy 0 <= echo_mix <= 1")


def laplacian3d(x: torch.Tensor) -> torch.Tensor:
    """Six-neighbour discrete Laplacian with replicated boundaries."""
    p = F.pad(x, (1, 1, 1, 1, 1, 1), mode="replicate")
    c = p[:, :, 1:-1, 1:-1, 1:-1]
    neighbours = (
        p[:, :, 2:, 1:-1, 1:-1]
        + p[:, :, :-2, 1:-1, 1:-1]
        + p[:, :, 1:-1, 2:, 1:-1]
        + p[:, :, 1:-1, :-2, 1:-1]
        + p[:, :, 1:-1, 1:-1, 2:]
        + p[:, :, 1:-1, 1:-1, :-2]
    )
    return neighbours - 6.0 * c


def strings_to_tokens(
    strings: Sequence[str], cfg: MM3DConfig, device: torch.device
) -> torch.Tensor:
    tokens = torch.zeros(
        len(strings), cfg.sequence_length, dtype=torch.long, device=device
    )
    for i, text in enumerate(strings):
        raw = list(text.encode("utf-8")[: cfg.sequence_length])
        if raw:
            tokens[i, : len(raw)] = torch.tensor(raw, dtype=torch.long, device=device)
    return tokens


def decode_byte_logits(logits: torch.Tensor) -> List[str]:
    ids = logits.argmax(dim=-1).detach().cpu().tolist()
    decoded: List[str] = []
    for row in ids:
        data = bytes(int(value) for value in row if 0 < int(value) < 256)
        decoded.append(data.decode("utf-8", errors="ignore"))
    return decoded


class ByteEncoder(nn.Module):
    def __init__(self, cfg: MM3DConfig):
        super().__init__()
        c = cfg.latent_channels
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, c)
        self.position = nn.Parameter(torch.randn(cfg.sequence_length, c) * 0.01)
        layer = nn.TransformerEncoderLayer(
            d_model=c,
            nhead=cfg.attention_heads,
            dim_feedforward=4 * c,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)

    def forward(self, strings: Sequence[str], device: torch.device) -> torch.Tensor:
        tokens = strings_to_tokens(strings, self.cfg, device)
        h = self.transformer(self.embedding(tokens) + self.position.unsqueeze(0))
        h = F.adaptive_avg_pool1d(h.transpose(1, 2), self.cfg.grid**3)
        b = h.shape[0]
        g = self.cfg.grid
        return h.reshape(b, self.cfg.latent_channels, g, g, g)


class ImageEncoder(nn.Module):
    def __init__(self, cfg: MM3DConfig):
        super().__init__()
        c = cfg.latent_channels
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(48, c, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.depth_basis = nn.Parameter(torch.randn(1, c, cfg.grid, 1, 1) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.cfg.grid
        h = F.adaptive_avg_pool2d(self.net(x), (g, g)).unsqueeze(2)
        return h.repeat(1, 1, g, 1, 1) + self.depth_basis


class AudioEncoder(nn.Module):
    def __init__(self, cfg: MM3DConfig):
        super().__init__()
        c = cfg.latent_channels
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 15, stride=4, padding=7),
            nn.GELU(),
            nn.Conv1d(32, 48, 9, stride=4, padding=4),
            nn.GELU(),
            nn.Conv1d(48, c, 7, stride=4, padding=3),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.cfg.grid
        h = F.adaptive_avg_pool1d(self.net(x), g**3)
        return h.reshape(h.shape[0], self.cfg.latent_channels, g, g, g)


class VideoEncoder(nn.Module):
    def __init__(self, cfg: MM3DConfig):
        super().__init__()
        c = cfg.latent_channels
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Conv3d(3, 24, 3, stride=(1, 2, 2), padding=1),
            nn.GELU(),
            nn.Conv3d(24, 40, 3, stride=(2, 2, 2), padding=1),
            nn.GELU(),
            nn.Conv3d(40, c, 3, stride=2, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.cfg.grid
        return F.adaptive_avg_pool3d(self.net(x), (g, g, g))


class EnvironmentEncoder(nn.Module):
    """Encode an explicit 3D environment tensor into the shared latent lattice."""

    def __init__(self, cfg: MM3DConfig):
        super().__init__()
        c = cfg.latent_channels
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Conv3d(cfg.environment_channels, c // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(c // 2, c, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError("environment must have shape [B, C, D, H, W]")
        if x.shape[1] != self.cfg.environment_channels:
            raise ValueError(
                f"environment has {x.shape[1]} channels; "
                f"expected {self.cfg.environment_channels}"
            )
        g = self.cfg.grid
        return F.adaptive_avg_pool3d(self.net(x), (g, g, g))


class DecisionHead(nn.Module):
    """Map the latent field to logits used by the explicit softmax decision head."""

    def __init__(self, cfg: MM3DConfig):
        super().__init__()
        c = cfg.latent_channels
        self.net = nn.Sequential(
            nn.LayerNorm(c),
            nn.Linear(c, 2 * c),
            nn.GELU(),
            nn.Linear(2 * c, cfg.action_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        pooled = z.mean(dim=(2, 3, 4))
        return self.net(pooled)


class VolumeRenderer3D(nn.Module):
    """Differentiable orthographic volume renderer conditioned on action probabilities."""

    def __init__(self, cfg: MM3DConfig):
        super().__init__()
        c = cfg.latent_channels
        self.output_size = cfg.image_size
        self.decision_projection = nn.Linear(cfg.action_dim, c)
        self.rgba = nn.Conv3d(c, 4, 1)

    def forward(self, z: torch.Tensor, action_probs: torch.Tensor) -> torch.Tensor:
        b, c, depth, _, _ = z.shape
        condition = self.decision_projection(action_probs).view(b, c, 1, 1, 1)
        field = self.rgba(z + condition)
        rgb = torch.sigmoid(field[:, :3])
        density = F.softplus(field[:, 3:4])

        alpha = 1.0 - torch.exp(-density / float(max(1, depth)))
        prefix = torch.cat(
            (torch.ones_like(alpha[:, :, :1]), 1.0 - alpha + 1.0e-6), dim=2
        )
        transmittance = torch.cumprod(prefix, dim=2)[:, :, :-1]
        weights = alpha * transmittance
        rendered = (weights * rgb).sum(dim=2)
        return F.interpolate(
            rendered,
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )


class MultimodalFusion(nn.Module):
    def __init__(self, cfg: MM3DConfig, modalities: int = 6):
        super().__init__()
        c = cfg.latent_channels
        self.embedding = nn.Parameter(torch.randn(modalities, c) * 0.02)
        self.score = nn.Sequential(nn.Linear(c, c // 2), nn.GELU(), nn.Linear(c // 2, 1))
        self.post = nn.Sequential(nn.Conv3d(c, c, 1), nn.GroupNorm(8, c), nn.GELU())

    def forward(
        self, fields: Sequence[Tuple[int, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not fields:
            raise ValueError("at least one modality must be supplied")
        enriched: List[torch.Tensor] = []
        scores: List[torch.Tensor] = []
        for modality_id, field in fields:
            value = field + self.embedding[modality_id].view(1, -1, 1, 1, 1)
            enriched.append(value)
            scores.append(self.score(value.mean(dim=(2, 3, 4))))
        weights = torch.softmax(torch.cat(scores, dim=1), dim=1)
        fused = torch.zeros_like(enriched[0])
        for i, value in enumerate(enriched):
            fused = fused + value * weights[:, i].view(-1, 1, 1, 1, 1)
        return self.post(fused), weights


class GeometricResidualBlock(nn.Module):
    def __init__(self, cfg: MM3DConfig):
        super().__init__()
        c = cfg.latent_channels
        self.cfg = cfg
        self.diffusion_logit = nn.Parameter(torch.tensor(-1.0))
        self.norm = nn.GroupNorm(8, c)
        self.conv1 = nn.Conv3d(c, 2 * c, 3, padding=1)
        self.conv2 = nn.Conv3d(2 * c, c, 3, padding=1)
        self.gate = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Conv3d(c, c, 1), nn.Sigmoid())

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        alpha = self.cfg.diffusion_max * torch.sigmoid(self.diffusion_logit)
        diffused = z + alpha * laplacian3d(z)
        h = self.conv2(F.gelu(self.conv1(self.norm(diffused))))
        return diffused + self.gate(h) * h


class VoxelAttention(nn.Module):
    def __init__(self, cfg: MM3DConfig):
        super().__init__()
        c = cfg.latent_channels
        self.norm = nn.LayerNorm(c)
        self.attn = nn.MultiheadAttention(c, cfg.attention_heads, batch_first=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = z.shape
        tokens = z.flatten(2).transpose(1, 2)
        q = self.norm(tokens)
        attended, _ = self.attn(q, q, q, need_weights=False)
        return (tokens + attended).transpose(1, 2).reshape(b, c, d, h, w)


class EchoResolver3D(nn.Module):
    """Finite Neumann-series echo over a bounded 3D propagation operator.

    The local transition is a convex blend of the identity and replicated
    3x3x3 averaging. In the sup norm this operator is non-expansive, so the
    infinite weighted series converges for 0 < echo_weight < 1. Runtime
    execution uses the configured finite truncation and reports its tail
    factor explicitly rather than pretending to evaluate an infinite sum.
    """

    def __init__(self, cfg: MM3DConfig):
        super().__init__()
        self.weight = float(cfg.echo_weight)
        self.depth = int(cfg.echo_depth)
        self.mix = float(cfg.echo_mix)

    def transition(self, z: torch.Tensor) -> torch.Tensor:
        padded = F.pad(z, (1, 1, 1, 1, 1, 1), mode="replicate")
        averaged = F.avg_pool3d(padded, kernel_size=3, stride=1)
        return (1.0 - self.mix) * z + self.mix * averaged

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = z
        echoed = z
        coefficient = 1.0
        weight_sum = 1.0
        for _ in range(self.depth):
            state = self.transition(state)
            coefficient *= self.weight
            weight_sum += coefficient
            echoed = echoed + coefficient * state

        tail_factor = self.weight ** (self.depth + 1) / (1.0 - self.weight)
        return (
            echoed,
            z.new_tensor(weight_sum),
            z.new_tensor(tail_factor),
        )


class OmegaMemory(nn.Module):
    def __init__(self, cfg: MM3DConfig):
        super().__init__()
        c = cfg.latent_channels
        self.update = nn.Conv3d(2 * c, c, 1)
        self.candidate = nn.Conv3d(2 * c, c, 1)

    def forward(self, current: torch.Tensor, previous: torch.Tensor | None) -> torch.Tensor:
        if previous is None:
            return current
        joined = torch.cat((current, previous), dim=1)
        gate = torch.sigmoid(self.update(joined))
        candidate = torch.tanh(self.candidate(joined))
        return gate * candidate + (1.0 - gate) * previous


class ByteDecoder(nn.Module):
    def __init__(self, cfg: MM3DConfig):
        super().__init__()
        c = cfg.latent_channels
        self.position = nn.Parameter(torch.randn(cfg.sequence_length, c) * 0.01)
        self.out = nn.Linear(c, cfg.vocab_size)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        pooled = z.mean(dim=(2, 3, 4)).unsqueeze(1)
        return self.out(pooled + self.position.unsqueeze(0))


class ImageDecoder(nn.Module):
    def __init__(self, cfg: MM3DConfig):
        super().__init__()
        c = cfg.latent_channels
        self.size = cfg.image_size
        self.net = nn.Sequential(
            nn.Conv2d(c, 96, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(96, 48, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(48, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.net(z.mean(dim=2))
        return F.interpolate(h, size=(self.size, self.size), mode="bilinear", align_corners=False)


class AudioDecoder(nn.Module):
    def __init__(self, cfg: MM3DConfig):
        super().__init__()
        c = cfg.latent_channels
        self.net = nn.Sequential(
            nn.Linear(c, 4 * c), nn.GELU(), nn.Linear(4 * c, cfg.audio_samples), nn.Tanh()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z.mean(dim=(2, 3, 4))).unsqueeze(1)


class VideoDecoder(nn.Module):
    def __init__(self, cfg: MM3DConfig):
        super().__init__()
        c = cfg.latent_channels
        self.shape = (cfg.video_frames, cfg.video_size, cfg.video_size)
        self.net = nn.Sequential(
            nn.Conv3d(c, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(64, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.net(z)
        return F.interpolate(h, size=self.shape, mode="trilinear", align_corners=False)


class MM3DEngine(nn.Module):
    TEXT, CODE, IMAGE, AUDIO, VIDEO, ENVIRONMENT = range(6)

    def __init__(self, cfg: MM3DConfig = MM3DConfig()):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.text_encoder = ByteEncoder(cfg)
        self.code_encoder = ByteEncoder(cfg)
        self.image_encoder = ImageEncoder(cfg)
        self.audio_encoder = AudioEncoder(cfg)
        self.video_encoder = VideoEncoder(cfg)
        self.environment_encoder = EnvironmentEncoder(cfg)
        self.fusion = MultimodalFusion(cfg)
        self.refinement = nn.ModuleList(
            GeometricResidualBlock(cfg) for _ in range(cfg.refinement_depth)
        )
        self.attention = VoxelAttention(cfg)
        self.memory = OmegaMemory(cfg)
        self.echo = EchoResolver3D(cfg)
        self.decision = DecisionHead(cfg)
        self.renderer3d = VolumeRenderer3D(cfg)
        self.text_decoder = ByteDecoder(cfg)
        self.code_decoder = ByteDecoder(cfg)
        self.image_decoder = ImageDecoder(cfg)
        self.audio_decoder = AudioDecoder(cfg)
        self.video_decoder = VideoDecoder(cfg)
        self._omega: torch.Tensor | None = None

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def reset_memory(self) -> None:
        self._omega = None

    def forward(
        self,
        *,
        text: Sequence[str] | None = None,
        code: Sequence[str] | None = None,
        image: torch.Tensor | None = None,
        audio: torch.Tensor | None = None,
        video: torch.Tensor | None = None,
        environment: torch.Tensor | None = None,
        stateful: bool = True,
    ) -> Dict[str, torch.Tensor]:
        fields: List[Tuple[int, torch.Tensor]] = []
        if text is not None:
            fields.append((self.TEXT, self.text_encoder(text, self.device)))
        if code is not None:
            fields.append((self.CODE, self.code_encoder(code, self.device)))
        if image is not None:
            fields.append((self.IMAGE, self.image_encoder(image)))
        if audio is not None:
            fields.append((self.AUDIO, self.audio_encoder(audio)))
        if video is not None:
            fields.append((self.VIDEO, self.video_encoder(video)))
        if environment is not None:
            fields.append((self.ENVIRONMENT, self.environment_encoder(environment)))
        z, weights = self.fusion(fields)
        for block in self.refinement:
            z = block(z)
        z = self.attention(z)
        z = self.memory(z, self._omega if stateful else None)
        pre_echo = z
        z, echo_weight_sum, echo_tail_factor = self.echo(z)
        if stateful:
            self._omega = z.detach()
        action_logits = self.decision(z)
        action_probs = torch.softmax(action_logits, dim=-1)
        rendered_3d = self.renderer3d(z, action_probs)
        return {
            "latent": z,
            "pre_echo_latent": pre_echo,
            "echo_weight_sum": echo_weight_sum,
            "echo_tail_factor": echo_tail_factor,
            "modality_weights": weights,
            "action_logits": action_logits,
            "action_probs": action_probs,
            "rendered_3d": rendered_3d,
            "text_logits": self.text_decoder(z),
            "code_logits": self.code_decoder(z),
            "image": self.image_decoder(z),
            "audio": self.audio_decoder(z),
            "video": self.video_decoder(z),
        }

    def loss(
        self, outputs: Dict[str, torch.Tensor], **targets: object
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        losses: List[torch.Tensor] = []
        metrics: Dict[str, float] = {}
        if targets.get("text") is not None:
            tokens = strings_to_tokens(
                targets["text"], self.cfg, self.device  # type: ignore[arg-type]
            )
            value = F.cross_entropy(outputs["text_logits"].transpose(1, 2), tokens)
            losses.append(value)
            metrics["text_ce"] = float(value.detach())
        if targets.get("code") is not None:
            tokens = strings_to_tokens(
                targets["code"], self.cfg, self.device  # type: ignore[arg-type]
            )
            value = F.cross_entropy(outputs["code_logits"].transpose(1, 2), tokens)
            losses.append(value)
            metrics["code_ce"] = float(value.detach())
        for name in ("image", "audio", "video"):
            target = targets.get(name)
            if isinstance(target, torch.Tensor):
                value = F.l1_loss(outputs[name], target)
                losses.append(value)
                metrics[f"{name}_l1"] = float(value.detach())
        latent_reg = outputs["latent"].square().mean() * self.cfg.latent_weight
        losses.append(latent_reg)
        total = torch.stack(losses).sum()
        metrics["latent_reg"] = float(latent_reg.detach())
        metrics["total"] = float(total.detach())
        return total, metrics


def demo_batch(cfg: MM3DConfig, device: torch.device) -> Dict[str, object]:
    text = ["A luminous sphere crosses a geometric field while a tone rises."]
    code = ["state = refine(encode(frame), memory=state)"]
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, cfg.image_size, device=device),
        torch.linspace(-1, 1, cfg.image_size, device=device),
        indexing="ij",
    )
    r = torch.sqrt(xx.square() + yy.square())
    image = torch.stack(
        (torch.exp(-4 * r.square()), 0.5 + 0.5 * torch.sin(6 * xx), 0.5 + 0.5 * torch.cos(6 * yy))
    ).clamp(0, 1).unsqueeze(0)
    t = torch.linspace(0, 1, cfg.audio_samples, device=device)
    phase = 2 * math.pi * (180 * t + 420 * t.square())
    audio = (0.6 * torch.sin(phase) + 0.2 * torch.sin(2.01 * phase)).view(1, 1, -1)
    base = F.interpolate(
        image, size=(cfg.video_size, cfg.video_size), mode="bilinear", align_corners=False
    )[0]
    frames = []
    for i in range(cfg.video_frames):
        shift = int(round((i / max(1, cfg.video_frames - 1)) * 8 - 4))
        frames.append(torch.roll(base, shifts=(shift, -shift), dims=(1, 2)))
    video = torch.stack(frames, dim=1).unsqueeze(0)
    return {"text": text, "code": code, "image": image, "audio": audio, "video": video}


def save_outputs(outputs: Dict[str, torch.Tensor], out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = out_dir / "generated_image.png"
    audio_path = out_dir / "generated_audio.wav"
    video_path = out_dir / "generated_video.gif"
    text_path = out_dir / "generated_text.txt"
    code_path = out_dir / "generated_code.txt"
    rendered_path = out_dir / "rendered_3d.png"

    image = outputs["image"][0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    Image.fromarray((image * 255).astype(np.uint8), "RGB").save(image_path)

    audio = outputs["audio"][0, 0].detach().cpu().clamp(-1, 1).numpy()
    pcm = (audio * 32767).astype(np.int16)
    with wave.open(str(audio_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm.tobytes())

    video = outputs["video"][0].detach().cpu().clamp(0, 1)
    frames: List[Image.Image] = []
    for index in range(video.shape[1]):
        frame = video[:, index].permute(1, 2, 0).numpy()
        frames.append(Image.fromarray((frame * 255).astype(np.uint8), "RGB"))
    frames[0].save(video_path, save_all=True, append_images=frames[1:], duration=125, loop=0)

    text_path.write_text(decode_byte_logits(outputs["text_logits"])[0], encoding="utf-8")
    code_path.write_text(decode_byte_logits(outputs["code_logits"])[0], encoding="utf-8")

    rendered = (
        outputs["rendered_3d"][0]
        .detach()
        .cpu()
        .clamp(0, 1)
        .permute(1, 2, 0)
        .numpy()
    )
    Image.fromarray((rendered * 255).astype(np.uint8), "RGB").save(rendered_path)

    return {name: str(path) for name, path in {
        "image": image_path,
        "audio": audio_path,
        "video": video_path,
        "text": text_path,
        "code": code_path,
        "rendered_3d": rendered_path,
    }.items()}


def train_demo(engine: MM3DEngine, batch: Dict[str, object], steps: int, lr: float) -> None:
    optimizer = torch.optim.AdamW(engine.parameters(), lr=lr)
    engine.train()
    for step in range(1, steps + 1):
        engine.reset_memory()
        optimizer.zero_grad(set_to_none=True)
        outputs = engine(stateful=False, **batch)  # type: ignore[arg-type]
        loss, metrics = engine.loss(outputs, **batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(engine.parameters(), 1.0)
        optimizer.step()
        print(json.dumps({"step": step, **metrics}, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Jarvis-X trainable MM3D multimodal engine")
    parser.add_argument("--steps", type=int, default=0, help="bounded demo training steps")
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--out", type=Path, default=Path("mm3d-output"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = MM3DConfig()
    engine = MM3DEngine(cfg).to(device)
    batch = demo_batch(cfg, device)
    if args.steps > 0:
        train_demo(engine, batch, args.steps, args.lr)
    engine.eval()
    engine.reset_memory()
    with torch.no_grad():
        outputs = engine(stateful=False, **batch)  # type: ignore[arg-type]
    files = save_outputs(outputs, args.out)
    telemetry = {
        "config": asdict(cfg),
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in engine.parameters()),
        "latent_shape": list(outputs["latent"].shape),
        "latent_mean": float(outputs["latent"].mean()),
        "latent_std": float(outputs["latent"].std()),
        "echo_weight_sum": float(outputs["echo_weight_sum"]),
        "echo_tail_factor": float(outputs["echo_tail_factor"]),
        "modality_weights": outputs["modality_weights"].detach().cpu().tolist(),
        "outputs": files,
    }
    (args.out / "telemetry.json").write_text(json.dumps(telemetry, indent=2), encoding="utf-8")
    print(json.dumps(telemetry, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
