#!/usr/bin/env python3
"""Recursive inward-looped 3D multimodal/multimedia ANN engine.

The engine maps heterogeneous media into one 3D latent field, recursively
decodes and re-encodes its own generated outputs, and exposes modality-specific
generation heads for image, video, audio, text logits, and sensor/features.

This is a finite, executable fixed-point approximation:
    F_{r+1} = (1-g) F_r + g P(G(D(E(F_r))))
where E/D are the volumetric encoder/decoder, G are generative heads, and P
projects generated media back into the shared 3D field.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Inward3DConfig:
    base_res: int = 8
    field_channels: int = 16
    latent_channels: int = 64
    core_res: int = 2
    vector_hidden: int = 128
    audio_features: int = 128
    text_features: int = 128
    sensor_features: int = 64
    image_size: int = 32
    video_frames: int = 4
    video_size: int = 32
    audio_samples: int = 1024
    vocab_size: int = 512
    text_length: int = 16
    text_model_dim: int = 128
    recursive_steps: int = 3
    inner_steps: int = 2
    correction_gain: float = 0.50
    reentry_gain: float = 0.35
    memory_decay: float = 0.90
    convergence_tol: float = 1.0e-4

    def validate(self) -> None:
        if self.base_res < 4:
            raise ValueError("base_res must be >= 4")
        if self.core_res < 1 or self.core_res > self.base_res:
            raise ValueError("core_res must be in [1, base_res]")
        if self.field_channels <= 0 or self.latent_channels <= 0:
            raise ValueError("channel counts must be positive")
        if self.recursive_steps < 1 or self.inner_steps < 1:
            raise ValueError("recursive_steps and inner_steps must be >= 1")
        if not 0.0 < self.correction_gain <= 1.0:
            raise ValueError("correction_gain must be in (0, 1]")
        if not 0.0 < self.reentry_gain <= 1.0:
            raise ValueError("reentry_gain must be in (0, 1]")
        if not 0.0 <= self.memory_decay < 1.0:
            raise ValueError("memory_decay must be in [0, 1)")


@dataclass
class MultimodalBatch:
    image: Optional[torch.Tensor] = None          # [B,3,H,W]
    video: Optional[torch.Tensor] = None          # [B,3,T,H,W]
    audio_features: Optional[torch.Tensor] = None # [B,audio_features]
    text_features: Optional[torch.Tensor] = None  # [B,text_features]
    sensor: Optional[torch.Tensor] = None         # [B,sensor_features]
    text_tokens: Optional[torch.Tensor] = None    # [B,text_length], training target only

    def tensors(self) -> Iterable[torch.Tensor]:
        for value in (
            self.image,
            self.video,
            self.audio_features,
            self.text_features,
            self.sensor,
        ):
            if value is not None:
                yield value

    @property
    def batch_size(self) -> int:
        values = list(self.tensors())
        if not values:
            raise ValueError("at least one input modality is required")
        batch_sizes = {int(value.shape[0]) for value in values}
        if len(batch_sizes) != 1:
            raise ValueError("all modalities must have the same batch size")
        return batch_sizes.pop()


@dataclass
class EngineOutput:
    field: torch.Tensor
    core: torch.Tensor
    memory: torch.Tensor
    reconstruction: torch.Tensor
    reentry_field: torch.Tensor
    image: torch.Tensor
    video: torch.Tensor
    audio: torch.Tensor
    text_logits: torch.Tensor
    sensor: torch.Tensor
    convergence: torch.Tensor
    recursive_steps: int


class VectorToVolume(nn.Module):
    """Low-rank vector -> 3D volume projection to avoid a giant dense matrix."""

    def __init__(self, input_dim: int, channels: int, base_res: int, hidden: int) -> None:
        super().__init__()
        self.channels = channels
        self.base_res = base_res
        self.seed_res = min(4, base_res)
        seed_dim = channels * self.seed_res**3
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, seed_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 2:
            raise ValueError("vector modality must have shape [B,F]")
        batch = value.shape[0]
        seed = self.net(value).view(
            batch, self.channels, self.seed_res, self.seed_res, self.seed_res
        )
        if self.seed_res == self.base_res:
            return seed
        return F.interpolate(
            seed,
            size=(self.base_res, self.base_res, self.base_res),
            mode="trilinear",
            align_corners=False,
        )


class MultimodalProjector3D(nn.Module):
    """Map image/video/audio/text/sensor inputs into a common 3D field."""

    MODALITIES = ("image", "video", "audio", "text", "sensor")

    def __init__(self, cfg: Inward3DConfig) -> None:
        super().__init__()
        c, r = cfg.field_channels, cfg.base_res
        self.cfg = cfg
        self.image_net = nn.Sequential(
            nn.Conv2d(3, c * r, 3, padding=1),
            nn.SiLU(),
        )
        self.video_net = nn.Sequential(
            nn.Conv3d(3, c, 3, padding=1),
            nn.SiLU(),
        )
        self.audio_net = VectorToVolume(cfg.audio_features, c, r, cfg.vector_hidden)
        self.text_net = VectorToVolume(cfg.text_features, c, r, cfg.vector_hidden)
        self.sensor_net = VectorToVolume(cfg.sensor_features, c, r, cfg.vector_hidden)
        self.logits = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(())) for name in self.MODALITIES}
        )

    def project_image(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape [B,3,H,W]")
        r, c = self.cfg.base_res, self.cfg.field_channels
        x = F.interpolate(image, size=(r, r), mode="bilinear", align_corners=False)
        return self.image_net(x).view(image.shape[0], c, r, r, r)

    def project_video(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError("video must have shape [B,3,T,H,W]")
        r = self.cfg.base_res
        x = F.interpolate(
            video, size=(r, r, r), mode="trilinear", align_corners=False
        )
        return self.video_net(x)

    def project_batch(self, batch: MultimodalBatch) -> Dict[str, torch.Tensor]:
        _ = batch.batch_size
        projected: Dict[str, torch.Tensor] = {}
        if batch.image is not None:
            projected["image"] = self.project_image(batch.image)
        if batch.video is not None:
            projected["video"] = self.project_video(batch.video)
        if batch.audio_features is not None:
            projected["audio"] = self.audio_net(batch.audio_features)
        if batch.text_features is not None:
            projected["text"] = self.text_net(batch.text_features)
        if batch.sensor is not None:
            projected["sensor"] = self.sensor_net(batch.sensor)
        return projected

    def fuse(self, volumes: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not volumes:
            raise ValueError("cannot fuse an empty modality set")
        names = list(volumes)
        weights = torch.softmax(torch.stack([self.logits[name] for name in names]), dim=0)
        result = torch.zeros_like(volumes[names[0]])
        for weight, name in zip(weights, names):
            result = result + weight * volumes[name]
        return result


class Encoder3D(nn.Module):
    def __init__(self, cfg: Inward3DConfig) -> None:
        super().__init__()
        c = cfg.field_channels
        h = max(32, c * 2)
        self.core_res = cfg.core_res
        self.net = nn.Sequential(
            nn.Conv3d(c, h, 3, padding=1),
            nn.GroupNorm(4, h),
            nn.SiLU(),
            nn.Conv3d(h, cfg.latent_channels, 3, padding=1),
            nn.GroupNorm(8, cfg.latent_channels),
            nn.SiLU(),
        )

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        x = self.net(field)
        return F.adaptive_avg_pool3d(
            x, (self.core_res, self.core_res, self.core_res)
        )


class Decoder3D(nn.Module):
    def __init__(self, cfg: Inward3DConfig) -> None:
        super().__init__()
        self.base_res = cfg.base_res
        h = max(32, cfg.field_channels * 2)
        self.net = nn.Sequential(
            nn.Conv3d(cfg.latent_channels, h, 3, padding=1),
            nn.GroupNorm(4, h),
            nn.SiLU(),
            nn.Conv3d(h, cfg.field_channels, 3, padding=1),
        )

    def forward(self, core: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            core,
            size=(self.base_res, self.base_res, self.base_res),
            mode="trilinear",
            align_corners=False,
        )
        return self.net(x)


class RecursiveCore3D(nn.Module):
    """Finite self-referential fixed-point dynamics with an EMA memory field."""

    def __init__(self, cfg: Inward3DConfig) -> None:
        super().__init__()
        c = cfg.latent_channels
        self.decay = cfg.memory_decay
        self.proposal = nn.Sequential(
            nn.Conv3d(c * 2, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.Tanh(),
        )
        self.gate = nn.Conv3d(c * 2, c, 1)

    def forward(
        self, core: torch.Tensor, memory: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        joined = torch.cat([core, memory], dim=1)
        proposal = self.proposal(joined)
        gate = torch.sigmoid(self.gate(joined))
        next_core = core + gate * (proposal - core)
        next_memory = self.decay * memory + (1.0 - self.decay) * next_core
        return next_core, next_memory


class AudioGeneratorHead(nn.Module):
    def __init__(self, cfg: Inward3DConfig) -> None:
        super().__init__()
        self.samples = cfg.audio_samples
        self.seed_length = 16
        self.fc = nn.Linear(cfg.latent_channels, 64 * self.seed_length)
        self.net = nn.Sequential(
            nn.ConvTranspose1d(64, 32, 4, stride=4),
            nn.SiLU(),
            nn.ConvTranspose1d(32, 16, 4, stride=4),
            nn.SiLU(),
            nn.ConvTranspose1d(16, 1, 4, stride=4),
            nn.Tanh(),
        )

    def forward(self, pooled_core: torch.Tensor) -> torch.Tensor:
        x = self.fc(pooled_core).view(-1, 64, self.seed_length)
        x = self.net(x)
        if x.shape[-1] != self.samples:
            x = F.interpolate(x, size=self.samples, mode="linear", align_corners=False)
        return x.squeeze(1)


class TextGeneratorHead(nn.Module):
    def __init__(self, cfg: Inward3DConfig) -> None:
        super().__init__()
        self.length = cfg.text_length
        self.vocab_size = cfg.vocab_size
        self.seed = nn.Linear(cfg.latent_channels, cfg.text_model_dim)
        self.position = nn.Parameter(torch.randn(cfg.text_length, cfg.text_model_dim) * 0.02)
        self.norm = nn.LayerNorm(cfg.text_model_dim)
        self.to_logits = nn.Linear(cfg.text_model_dim, cfg.vocab_size)
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.text_features)

    def forward(self, pooled_core: torch.Tensor) -> torch.Tensor:
        state = self.seed(pooled_core)[:, None, :] + self.position[None, :, :]
        return self.to_logits(torch.tanh(self.norm(state)))

    def expected_feature(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        token_features = probs @ self.token_embedding.weight
        return token_features.mean(dim=1)


class MultimediaGenerator3D(nn.Module):
    """Decode shared volumetric state into several synchronized media modalities."""

    def __init__(self, cfg: Inward3DConfig) -> None:
        super().__init__()
        c = cfg.field_channels
        self.cfg = cfg
        self.image_net = nn.Sequential(
            nn.Conv2d(c, max(16, c), 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(max(16, c), 3, 1),
            nn.Tanh(),
        )
        self.video_net = nn.Sequential(
            nn.Conv3d(c, max(16, c), 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(max(16, c), 3, 1),
            nn.Tanh(),
        )
        self.audio_head = AudioGeneratorHead(cfg)
        self.text_head = TextGeneratorHead(cfg)
        self.sensor_head = nn.Sequential(
            nn.Linear(cfg.latent_channels, cfg.vector_hidden),
            nn.SiLU(),
            nn.Linear(cfg.vector_hidden, cfg.sensor_features),
        )

    def forward(self, field: torch.Tensor, core: torch.Tensor) -> Dict[str, torch.Tensor]:
        pooled = F.adaptive_avg_pool3d(core, 1).flatten(1)

        image = self.image_net(field.mean(dim=2))
        image = F.interpolate(
            image,
            size=(self.cfg.image_size, self.cfg.image_size),
            mode="bilinear",
            align_corners=False,
        )

        video = self.video_net(field)
        video = F.interpolate(
            video,
            size=(self.cfg.video_frames, self.cfg.video_size, self.cfg.video_size),
            mode="trilinear",
            align_corners=False,
        )

        return {
            "image": image,
            "video": video,
            "audio": self.audio_head(pooled),
            "text_logits": self.text_head(pooled),
            "sensor": self.sensor_head(pooled),
        }


class InwardMultimedia3DANN(nn.Module):
    """3D multimodal ANN whose generated outputs are projected back into itself."""

    def __init__(self, cfg: Inward3DConfig = Inward3DConfig()) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.projector = MultimodalProjector3D(cfg)
        self.encoder = Encoder3D(cfg)
        self.decoder = Decoder3D(cfg)
        self.core_dynamics = RecursiveCore3D(cfg)
        self.generator = MultimediaGenerator3D(cfg)
        self.core_feedback = nn.Linear(cfg.latent_channels, cfg.field_channels)
        self.last_metrics: Dict[str, float | int] = {}

    def _generated_reentry(
        self, generated: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        audio_features = F.interpolate(
            generated["audio"].unsqueeze(1),
            size=self.cfg.audio_features,
            mode="linear",
            align_corners=False,
        ).squeeze(1)
        text_features = self.generator.text_head.expected_feature(
            generated["text_logits"]
        )
        volumes = {
            "image": self.projector.project_image(generated["image"]),
            "video": self.projector.project_video(generated["video"]),
            "audio": self.projector.audio_net(audio_features),
            "text": self.projector.text_net(text_features),
            "sensor": self.projector.sensor_net(generated["sensor"]),
        }
        return self.projector.fuse(volumes)

    def _run_field(
        self,
        field: torch.Tensor,
        *,
        recursive_steps: Optional[int] = None,
        convergence_tol: Optional[float] = None,
    ) -> EngineOutput:
        cfg = self.cfg
        steps = cfg.recursive_steps if recursive_steps is None else int(recursive_steps)
        tol = cfg.convergence_tol if convergence_tol is None else float(convergence_tol)
        if steps < 1:
            raise ValueError("recursive_steps must be >= 1")

        memory: Optional[torch.Tensor] = None
        convergence = torch.full(
            (field.shape[0],), float("inf"), dtype=field.dtype, device=field.device
        )
        generated: Dict[str, torch.Tensor] = {}
        reconstruction = field
        reentry = field
        core = self.encoder(field)
        executed = 0

        for index in range(steps):
            core = self.encoder(field)
            if memory is None:
                memory = torch.zeros_like(core)
            for _ in range(cfg.inner_steps):
                core, memory = self.core_dynamics(core, memory)

            reconstruction = self.decoder(core)
            residual_core = self.encoder(field - reconstruction)
            core = core + cfg.correction_gain * residual_core
            reconstruction = self.decoder(core)

            pooled = F.adaptive_avg_pool3d(core, 1).flatten(1)
            channel_bias = self.core_feedback(pooled)[:, :, None, None, None]
            reconstruction = torch.tanh(reconstruction + channel_bias)

            generated = self.generator(reconstruction, core)
            reentry = self._generated_reentry(generated)
            target_field = 0.5 * reconstruction + 0.5 * reentry
            next_field = field + cfg.reentry_gain * (target_field - field)

            convergence = (next_field - field).flatten(1).pow(2).mean(dim=1).sqrt()
            field = next_field
            executed = index + 1
            if tol > 0.0 and bool(torch.all(convergence <= tol)):
                break

        assert memory is not None
        generated = self.generator(field, core)
        self.last_metrics = {
            "recursive_steps": executed,
            "mean_fixed_point_delta": float(convergence.detach().mean()),
            "max_fixed_point_delta": float(convergence.detach().max()),
        }
        return EngineOutput(
            field=field,
            core=core,
            memory=memory,
            reconstruction=reconstruction,
            reentry_field=reentry,
            image=generated["image"],
            video=generated["video"],
            audio=generated["audio"],
            text_logits=generated["text_logits"],
            sensor=generated["sensor"],
            convergence=convergence,
            recursive_steps=executed,
        )

    def forward(
        self,
        batch: MultimodalBatch,
        *,
        recursive_steps: Optional[int] = None,
        convergence_tol: Optional[float] = None,
    ) -> EngineOutput:
        volumes = self.projector.project_batch(batch)
        field = self.projector.fuse(volumes)
        return self._run_field(
            field,
            recursive_steps=recursive_steps,
            convergence_tol=convergence_tol,
        )

    def generate(
        self,
        batch_size: int = 1,
        *,
        device: Optional[torch.device | str] = None,
        temperature: float = 1.0,
        recursive_steps: Optional[int] = None,
    ) -> EngineOutput:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        parameter = next(self.parameters())
        target_device = parameter.device if device is None else torch.device(device)
        field = torch.randn(
            batch_size,
            self.cfg.field_channels,
            self.cfg.base_res,
            self.cfg.base_res,
            self.cfg.base_res,
            device=target_device,
            dtype=parameter.dtype,
        ) * temperature
        return self._run_field(field, recursive_steps=recursive_steps)

    def loss(
        self,
        batch: MultimodalBatch,
        output: EngineOutput,
        *,
        cycle_weight: float = 0.25,
        fixed_point_weight: float = 0.05,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        terms: Dict[str, torch.Tensor] = {}

        if batch.image is not None:
            target = F.interpolate(
                batch.image,
                size=(self.cfg.image_size, self.cfg.image_size),
                mode="bilinear",
                align_corners=False,
            )
            terms["image"] = F.mse_loss(output.image, target)

        if batch.video is not None:
            target = F.interpolate(
                batch.video,
                size=(self.cfg.video_frames, self.cfg.video_size, self.cfg.video_size),
                mode="trilinear",
                align_corners=False,
            )
            terms["video"] = F.mse_loss(output.video, target)

        if batch.audio_features is not None:
            target = F.interpolate(
                batch.audio_features.unsqueeze(1),
                size=self.cfg.audio_samples,
                mode="linear",
                align_corners=False,
            ).squeeze(1)
            terms["audio"] = F.mse_loss(output.audio, target)

        if batch.sensor is not None:
            terms["sensor"] = F.mse_loss(output.sensor, batch.sensor)

        if batch.text_tokens is not None:
            if batch.text_tokens.ndim != 2:
                raise ValueError("text_tokens must have shape [B,L]")
            if batch.text_tokens.shape[1] != self.cfg.text_length:
                raise ValueError(
                    f"text_tokens length must equal configured text_length={self.cfg.text_length}"
                )
            terms["text"] = F.cross_entropy(
                output.text_logits.reshape(-1, self.cfg.vocab_size),
                batch.text_tokens.reshape(-1),
            )

        terms["cycle"] = F.mse_loss(output.field, output.reentry_field)
        terms["fixed_point"] = output.convergence.mean()

        reconstruction_terms = [
            value for name, value in terms.items()
            if name not in {"cycle", "fixed_point"}
        ]
        if reconstruction_terms:
            total = torch.stack(reconstruction_terms).sum()
        else:
            total = output.field.new_zeros(())
        total = total + cycle_weight * terms["cycle"] + fixed_point_weight * terms["fixed_point"]
        return total, terms


def demo() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Inward3DConfig(base_res=8, recursive_steps=2, inner_steps=2)
    engine = InwardMultimedia3DANN(cfg).to(device)

    batch = MultimodalBatch(
        image=torch.randn(1, 3, 32, 32, device=device),
        video=torch.randn(1, 3, 4, 32, 32, device=device),
        audio_features=torch.randn(1, cfg.audio_features, device=device),
        text_features=torch.randn(1, cfg.text_features, device=device),
        sensor=torch.randn(1, cfg.sensor_features, device=device),
        text_tokens=torch.randint(
            0, cfg.vocab_size, (1, cfg.text_length), device=device
        ),
    )

    output = engine(batch)
    loss, terms = engine.loss(batch, output)
    print(f"device={device}")
    print(f"field={tuple(output.field.shape)} core={tuple(output.core.shape)}")
    print(
        "generated:",
        f"image={tuple(output.image.shape)}",
        f"video={tuple(output.video.shape)}",
        f"audio={tuple(output.audio.shape)}",
        f"text_logits={tuple(output.text_logits.shape)}",
        f"sensor={tuple(output.sensor.shape)}",
    )
    print(f"loss={float(loss.detach()):.6f}")
    print({name: float(value.detach()) for name, value in terms.items()})
    print(engine.last_metrics)


if __name__ == "__main__":
    demo()
