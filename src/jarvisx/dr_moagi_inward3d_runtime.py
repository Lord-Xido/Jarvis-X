"""Bounded inward-recursive 3D autoencoding laboratory for Jarvis-X.

The runtime is deliberately explicit about its limits:

* the advertised large matrix is a virtual streamed address space, not a literal
  contiguous allocation;
* byte payloads are represented as eight binary bit-planes in a finite 3D tile;
* the learned path uses ordinary PyTorch floating-point convolutions; a separate
  XNOR/POPCOUNT primitive is provided and tested, but is not claimed as the
  neural execution kernel;
* recursion is true latent-state recursion: ``z_{r+1}`` inherits ``z_r`` rather
  than re-encoding the same input on every fold;
* acceptance is guarded by an immutable composite validation score and exact
  bit/Hamming telemetry is reported separately from differentiable loss.

This module requires the optional ``jarvisx[torch]`` dependency.
"""

from __future__ import annotations

import copy
import hashlib
import math
import random
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Inward3DConfig:
    """Configuration for the bounded 3D reference runtime."""

    tile_edge: int = 16
    base_channels: int = 8
    latent_channels: int = 16
    recursion_depth: int = 3
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-5
    recon_weight: float = 1.0
    cycle_weight: float = 0.15
    gradient_weight: float = 0.05
    self_weight: float = 0.03
    latent_weight: float = 1.0e-5
    grad_clip: float = 1.0
    seed: int = 7

    def validate(self) -> None:
        if self.tile_edge < 16 or self.tile_edge % 8:
            raise ValueError("tile_edge must be >= 16 and divisible by 8")
        if self.base_channels <= 0 or self.latent_channels <= 0:
            raise ValueError("channel counts must be positive")
        if self.recursion_depth <= 0:
            raise ValueError("recursion_depth must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")


@dataclass(frozen=True)
class BitIdentity:
    """Exact byte/bit reconstruction telemetry."""

    total_bits: int
    differing_bits: int
    bit_accuracy: float
    byte_accuracy: float
    sha256_original: str
    sha256_reconstructed: str

    @property
    def exact(self) -> bool:
        return self.differing_bits == 0 and self.sha256_original == self.sha256_reconstructed


@dataclass(frozen=True)
class FoldTelemetry:
    """One inward recursion step."""

    depth: int
    gate_mean: float
    correction_rms: float
    error_rms: float
    latent_delta_rms: float


@dataclass(frozen=True)
class ScoreTelemetry:
    """Differentiable validation terms plus the immutable composite score."""

    score: float
    reconstruction: float
    cycle: float
    gradient: float
    self_consistency: float
    latent_energy: float
    error_roughness: float


class VirtualBitVolume:
    """Map byte streams into streamed 3D bit-plane tiles and back."""

    def __init__(self, edge: int) -> None:
        if edge < 1:
            raise ValueError("edge must be positive")
        self.edge = edge
        self.tile_bytes = edge**3

    @staticmethod
    def _byte_bits(values: np.ndarray) -> np.ndarray:
        shifts = np.arange(8, dtype=np.uint8)
        return ((values[:, None] >> shifts[None, :]) & 1).astype(np.float32)

    def encode(self, payload: bytes) -> list[np.ndarray]:
        if not payload:
            return []
        tiles: list[np.ndarray] = []
        for start in range(0, len(payload), self.tile_bytes):
            chunk = np.frombuffer(payload[start : start + self.tile_bytes], dtype=np.uint8)
            padded = np.zeros(self.tile_bytes, dtype=np.uint8)
            padded[: chunk.size] = chunk
            bits = self._byte_bits(padded)
            volume = bits.T.reshape(8, self.edge, self.edge, self.edge)
            tiles.append(volume)
        return tiles

    def decode(self, tiles: Iterable[np.ndarray], original_length: int) -> bytes:
        if original_length < 0:
            raise ValueError("original_length must be non-negative")
        values: list[np.ndarray] = []
        weights = (1 << np.arange(8, dtype=np.uint16)).reshape(8, 1)
        for tile in tiles:
            if tile.shape != (8, self.edge, self.edge, self.edge):
                raise ValueError("tile shape does not match configured bit volume")
            bit_matrix = (tile.reshape(8, -1) >= 0.5).astype(np.uint16)
            byte_values = np.sum(bit_matrix * weights, axis=0).astype(np.uint8)
            values.append(byte_values)
        if not values:
            return b""
        return np.concatenate(values)[:original_length].tobytes()


def xnor_popcount_dot(a: np.ndarray, b: np.ndarray) -> int:
    """Return the bipolar dot product using the XNOR/POPCOUNT identity."""

    aa = np.asarray(a, dtype=np.uint8).reshape(-1)
    bb = np.asarray(b, dtype=np.uint8).reshape(-1)
    if aa.shape != bb.shape:
        raise ValueError("bit vectors must have equal length")
    if not np.all((aa == 0) | (aa == 1)) or not np.all((bb == 0) | (bb == 1)):
        raise ValueError("xnor_popcount_dot accepts binary values only")
    matches = int(np.count_nonzero(aa == bb))
    return 2 * matches - int(aa.size)


def bit_identity(original: bytes, reconstructed: bytes) -> BitIdentity:
    """Measure exact Hamming and digest identity for equal-length byte strings."""

    if len(original) != len(reconstructed):
        raise ValueError("bit identity requires equal-length byte strings")
    a = np.frombuffer(original, dtype=np.uint8)
    b = np.frombuffer(reconstructed, dtype=np.uint8)
    xor = np.bitwise_xor(a, b)
    differing = int(sum(int(value).bit_count() for value in xor.tolist()))
    total = len(original) * 8
    bit_accuracy = 1.0 if total == 0 else 1.0 - differing / total
    byte_accuracy = 1.0 if len(original) == 0 else float(np.mean(a == b))
    return BitIdentity(
        total_bits=total,
        differing_bits=differing,
        bit_accuracy=bit_accuracy,
        byte_accuracy=byte_accuracy,
        sha256_original=hashlib.sha256(original).hexdigest(),
        sha256_reconstructed=hashlib.sha256(reconstructed).hexdigest(),
    )


def _gradient3d(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dx = x[..., :, :, 1:] - x[..., :, :, :-1]
    dy = x[..., :, 1:, :] - x[..., :, :-1, :]
    dz = x[..., 1:, :, :] - x[..., :-1, :, :]
    return dx, dy, dz


def _rms(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(x.square()) + 1.0e-12)


def _roughness(x: torch.Tensor) -> torch.Tensor:
    return sum(component.abs().mean() for component in _gradient3d(x)) / 3.0


class _Block3D(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, stride: int = 1) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(channels_in, channels_out, 3, stride=stride, padding=1),
            nn.SiLU(),
            nn.Conv3d(channels_out, channels_out, 3, padding=1),
            nn.SiLU(),
        )
        self.skip: nn.Module
        if channels_in == channels_out and stride == 1:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv3d(channels_in, channels_out, 1, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x) + self.skip(x)


class _Encoder3D(nn.Module):
    def __init__(self, base: int, latent: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _Block3D(8, base),
            _Block3D(base, base * 2, 2),
            _Block3D(base * 2, base * 4, 2),
            _Block3D(base * 4, latent, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Decoder3D(nn.Module):
    def __init__(self, base: int, latent: int) -> None:
        super().__init__()
        self.pre = _Block3D(latent, base * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose3d(base * 4, base * 2, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose3d(base * 2, base, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose3d(base, base, 4, 2, 1),
            nn.SiLU(),
            nn.Conv3d(base, 8, 3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(self.pre(z))


class _LatentSelfModel(nn.Module):
    def __init__(self, latent: int) -> None:
        super().__init__()
        hidden = max(16, latent)
        self.error_embed = nn.Sequential(nn.Conv3d(8, latent, 1), nn.SiLU())
        self.core = nn.Sequential(
            _Block3D(latent * 2, hidden),
            _Block3D(hidden, hidden),
        )
        self.delta = nn.Conv3d(hidden, latent, 1)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(hidden, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor, error: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = F.adaptive_avg_pool3d(error, z.shape[-3:])
        embedded = self.error_embed(pooled)
        state = self.core(torch.cat([z, embedded], dim=1))
        return torch.tanh(self.delta(state)), self.gate(state)


class InwardRecursive3DModel(nn.Module):
    """Encoder, true latent recurrence, self-model and decoder."""

    def __init__(self, config: Inward3DConfig) -> None:
        super().__init__()
        config.validate()
        self.encoder = _Encoder3D(config.base_channels, config.latent_channels)
        self.decoder = _Decoder3D(config.base_channels, config.latent_channels)
        self.self_model = _LatentSelfModel(config.latent_channels)

    def forward_recursive(
        self,
        x: torch.Tensor,
        depth: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[FoldTelemetry]]:
        if depth <= 0:
            raise ValueError("depth must be positive")
        z = self.encoder(x)
        error = torch.zeros_like(x)
        telemetry: list[FoldTelemetry] = []
        logits = torch.zeros_like(x)

        for index in range(depth):
            previous_z = z
            delta, gate = self.self_model(z, error)
            z = z + gate * delta
            logits = self.decoder(z)
            if logits.shape[-3:] != x.shape[-3:]:
                logits = F.interpolate(logits, size=x.shape[-3:], mode="trilinear", align_corners=False)
            reconstruction = torch.sigmoid(logits)
            error = x - reconstruction
            telemetry.append(
                FoldTelemetry(
                    depth=index,
                    gate_mean=float(gate.detach().mean().cpu()),
                    correction_rms=float(_rms(delta).detach().cpu()),
                    error_rms=float(_rms(error).detach().cpu()),
                    latent_delta_rms=float(_rms(z - previous_z).detach().cpu()),
                )
            )

        return z, logits, telemetry


class InwardRecursive3DRuntime:
    """Guarded training and reconstruction wrapper around the 3D model."""

    def __init__(self, config: Inward3DConfig | None = None, device: str = "cpu") -> None:
        self.config = config or Inward3DConfig()
        self.config.validate()
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        self.device = torch.device(device)
        self.volume = VirtualBitVolume(self.config.tile_edge)
        self.model = InwardRecursive3DModel(self.config).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _tensor(self, tile: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(tile).unsqueeze(0).to(self.device)

    def _score(self, x: torch.Tensor) -> tuple[torch.Tensor, ScoreTelemetry]:
        z, logits, _ = self.model.forward_recursive(x, self.config.recursion_depth)
        reconstruction = torch.sigmoid(logits)
        recon = F.binary_cross_entropy_with_logits(logits, x)
        z_cycle = self.model.encoder(reconstruction)
        cycle = F.mse_loss(z_cycle, z.detach())
        grad = sum(
            F.l1_loss(a, b) for a, b in zip(_gradient3d(x), _gradient3d(reconstruction))
        ) / 3.0
        error = x - reconstruction
        correction, _ = self.model.self_model(z_cycle, error)
        self_consistency = correction.square().mean()
        latent = z.square().mean()
        rough = _roughness(error)
        total = (
            self.config.recon_weight * recon
            + self.config.cycle_weight * cycle
            + self.config.gradient_weight * grad
            + self.config.self_weight * self_consistency
            + self.config.latent_weight * latent
        )
        immutable_score = recon + 0.25 * cycle + 0.10 * grad + 0.05 * self_consistency + 0.02 * rough
        return total, ScoreTelemetry(
            score=float(immutable_score.detach().cpu()),
            reconstruction=float(recon.detach().cpu()),
            cycle=float(cycle.detach().cpu()),
            gradient=float(grad.detach().cpu()),
            self_consistency=float(self_consistency.detach().cpu()),
            latent_energy=float(latent.detach().cpu()),
            error_roughness=float(rough.detach().cpu()),
        )

    @torch.no_grad()
    def evaluate_tile(self, tile: np.ndarray) -> ScoreTelemetry:
        self.model.eval()
        _, score = self._score(self._tensor(tile))
        return score

    def train_tile(self, tile: np.ndarray, steps: int = 1) -> ScoreTelemetry:
        if steps <= 0:
            raise ValueError("steps must be positive")
        x = self._tensor(tile)
        self.model.train()
        telemetry: ScoreTelemetry | None = None
        for _ in range(steps):
            self.optimizer.zero_grad(set_to_none=True)
            loss, telemetry = self._score(x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
        assert telemetry is not None
        return telemetry

    def guarded_train_tile(self, tile: np.ndarray, steps: int = 1) -> tuple[bool, float, float]:
        """Accept only if the immutable validation score improves; otherwise rollback."""

        before = self.evaluate_tile(tile).score
        model_snapshot = copy.deepcopy(self.model.state_dict())
        optimizer_snapshot = copy.deepcopy(self.optimizer.state_dict())
        self.train_tile(tile, steps=steps)
        after = self.evaluate_tile(tile).score
        if math.isfinite(after) and after < before:
            return True, before, after
        self.model.load_state_dict(model_snapshot)
        self.optimizer.load_state_dict(optimizer_snapshot)
        return False, before, self.evaluate_tile(tile).score

    @torch.no_grad()
    def reconstruct_tile(self, tile: np.ndarray) -> tuple[np.ndarray, list[FoldTelemetry]]:
        self.model.eval()
        x = self._tensor(tile)
        _, logits, telemetry = self.model.forward_recursive(x, self.config.recursion_depth)
        return torch.sigmoid(logits).squeeze(0).cpu().numpy(), telemetry

    def process_bytes(
        self,
        payload: bytes,
        *,
        training_steps: int = 0,
    ) -> tuple[bytes, BitIdentity, list[FoldTelemetry]]:
        if training_steps < 0:
            raise ValueError("training_steps must be non-negative")
        tiles = self.volume.encode(payload)
        outputs: list[np.ndarray] = []
        folds: list[FoldTelemetry] = []
        for tile in tiles:
            if training_steps:
                self.guarded_train_tile(tile, steps=training_steps)
            output, tile_folds = self.reconstruct_tile(tile)
            outputs.append(output)
            folds.extend(tile_folds)
        reconstructed = self.volume.decode(outputs, len(payload))
        return reconstructed, bit_identity(payload, reconstructed), folds
