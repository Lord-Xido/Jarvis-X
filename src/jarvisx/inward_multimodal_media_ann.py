"""Inward 3D multimodal media ANN research runtime.

This module keeps media generation and host-process packaging separate. It
implements a trainable NumPy autoencoder, shared 3D coordination, inward
E(D(z)) refinement, graph coupling, memory, deterministic frame/audio synthesis,
and bounded configuration search. It does not launch external processes.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, asdict, replace
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np

VIRTUAL_SIDE = 8192
VIRTUAL_CELLS = VIRTUAL_SIDE ** 3


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    GEOMETRY = "geometry"
    CODE = "code"
    DATA = "data"


@dataclass
class Particle:
    modality: Modality
    position: np.ndarray
    feature: np.ndarray
    source_id: str


@dataclass(frozen=True)
class MediaANNConfig:
    feature_dim: int = 32
    latent_dim: int = 16
    hidden_dim: int = 48
    dt: float = 0.18
    task_gain: float = 0.16
    swarm_gain: float = 0.14
    inward_gain: float = 0.38
    memory_gain: float = 0.08
    memory_retention: float = 0.86
    q_phi: float = 0.45
    geometry_gain: float = 0.22
    feature_gain: float = 1.15
    inner_steps: int = 18
    train_epochs: int = 180
    learning_rate: float = 0.035

    def validate(self) -> None:
        if min(self.feature_dim, self.latent_dim, self.hidden_dim) <= 0:
            raise ValueError("network dimensions must be positive")
        if not 0.0 < self.dt <= 1.0:
            raise ValueError("dt must lie in (0, 1]")
        if not 0.0 < self.q_phi < 1.0:
            raise ValueError("q_phi must lie in (0, 1)")
        if not 0.0 <= self.memory_retention < 1.0:
            raise ValueError("memory_retention must lie in [0, 1)")
        total = self.task_gain + self.swarm_gain + self.inward_gain + self.memory_gain
        if total >= 1.0:
            raise ValueError("sum of recurrent gains must remain below 1")


@dataclass(frozen=True)
class MediaANNMetrics:
    step: int
    fixed_point_residual: float
    consensus_spread: float
    feature_spread: float
    reconstruction_mse: float
    elapsed_ms: float


@dataclass(frozen=True)
class MediaANNResult:
    particles: tuple[Particle, ...]
    metrics: tuple[MediaANNMetrics, ...]
    training_loss: tuple[float, ...]
    score: float


def _normalize(value: np.ndarray) -> np.ndarray:
    return value / (np.linalg.norm(value) + 1e-12)


def _softmax(value: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = value - np.max(value, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / (np.sum(exp, axis=axis, keepdims=True) + 1e-12)


def _to_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, np.ndarray):
        header = json.dumps(
            {"shape": list(value.shape), "dtype": str(value.dtype)},
            sort_keys=True,
        ).encode("utf-8")
        return header + b"\n" + value.tobytes()
    return json.dumps(value, sort_keys=True, default=str).encode("utf-8")


def byte_features(blob: bytes, width: int) -> np.ndarray:
    if not blob:
        blob = b"\x00"
    values = np.frombuffer(blob, dtype=np.uint8).astype(np.float64) / 255.0
    groups = max(1, width // 4)
    chunks = np.array_split(values, groups)
    means = np.array([chunk.mean() if len(chunk) else 0.0 for chunk in chunks])
    stds = np.array([chunk.std() if len(chunk) else 0.0 for chunk in chunks])
    rms = np.array([
        math.sqrt(float(np.mean(chunk * chunk))) if len(chunk) else 0.0
        for chunk in chunks
    ])
    maxima = np.array([chunk.max() if len(chunk) else 0.0 for chunk in chunks])
    feature = np.concatenate([means, stds, rms, maxima])[:width]
    if len(feature) < width:
        feature = np.pad(feature, (0, width - len(feature)))
    return _normalize(feature)


def raw_coordinate(blob: bytes) -> np.ndarray:
    digest = hashlib.sha256(blob if blob else b"\x00").digest()
    values = np.frombuffer(digest[:6], dtype=np.uint16).astype(np.float64)
    return 2.0 * values[:3] / 65535.0 - 1.0


def virtual_xyz(position: np.ndarray) -> tuple[int, int, int]:
    xyz = np.clip(
        np.rint((position + 1.0) * 0.5 * (VIRTUAL_SIDE - 1)),
        0,
        VIRTUAL_SIDE - 1,
    ).astype(int)
    return int(xyz[0]), int(xyz[1]), int(xyz[2])


class TinyAutoencoder:
    """Small MLP autoencoder with explicit NumPy backpropagation."""

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, 0.16, (input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.w2 = rng.normal(0, 0.16, (hidden_dim, latent_dim))
        self.b2 = np.zeros(latent_dim)
        self.w3 = rng.normal(0, 0.16, (latent_dim, hidden_dim))
        self.b3 = np.zeros(hidden_dim)
        self.w4 = rng.normal(0, 0.16, (hidden_dim, input_dim))
        self.b4 = np.zeros(input_dim)

    def encode(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hidden = np.tanh(x @ self.w1 + self.b1)
        latent = np.tanh(hidden @ self.w2 + self.b2)
        return hidden, latent

    def decode(self, latent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hidden = np.tanh(latent @ self.w3 + self.b3)
        output = np.tanh(hidden @ self.w4 + self.b4)
        return hidden, output

    def forward(self, x: np.ndarray):
        h1, latent = self.encode(x)
        h3, output = self.decode(latent)
        return h1, latent, h3, output

    def train(self, x: np.ndarray, epochs: int, learning_rate: float) -> tuple[float, ...]:
        history: list[float] = []
        n = x.shape[0]
        for _ in range(epochs):
            h1, latent, h3, output = self.forward(x)
            error = output - x
            history.append(float(np.mean(error * error)))
            d_output = 2.0 * error / (n * x.shape[1])
            da4 = d_output * (1.0 - output * output)
            gw4, gb4 = h3.T @ da4, da4.sum(0)
            dh3 = da4 @ self.w4.T
            da3 = dh3 * (1.0 - h3 * h3)
            gw3, gb3 = latent.T @ da3, da3.sum(0)
            d_latent = da3 @ self.w3.T
            da2 = d_latent * (1.0 - latent * latent)
            gw2, gb2 = h1.T @ da2, da2.sum(0)
            dh1 = da2 @ self.w2.T
            da1 = dh1 * (1.0 - h1 * h1)
            gw1, gb1 = x.T @ da1, da1.sum(0)
            for name, gradient in (
                ("w4", gw4), ("b4", gb4), ("w3", gw3), ("b3", gb3),
                ("w2", gw2), ("b2", gb2), ("w1", gw1), ("b1", gb1),
            ):
                setattr(
                    self,
                    name,
                    getattr(self, name) - learning_rate * np.clip(gradient, -1.0, 1.0),
                )
        return tuple(history)


class MultimodalEncoder:
    def __init__(self, config: MediaANNConfig):
        self.config = config

    def encode(self, modality: Modality, value: Any, source_id: str) -> Particle:
        blob = _to_bytes(value)
        feature = byte_features(blob, self.config.feature_dim)
        raw = raw_coordinate(blob)
        index = list(Modality).index(modality)
        phase = (index + 1) / len(Modality)
        bias = np.array([
            math.sin(math.pi * phase),
            math.cos(math.pi * phase),
            2.0 * phase - 1.0,
        ])
        position = np.clip(0.30 * raw + 0.70 * bias, -1.0, 1.0)
        return Particle(modality, position, feature, source_id)


def attention(particles: Sequence[Particle], config: MediaANNConfig) -> np.ndarray:
    features = np.stack([particle.feature for particle in particles])
    positions = np.stack([particle.position for particle in particles])
    normalized = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)
    semantic = normalized @ normalized.T
    distance2 = ((positions[:, None, :] - positions[None, :, :]) ** 2).sum(-1)
    logits = config.feature_gain * semantic - config.geometry_gain * distance2
    if len(particles) > 1:
        np.fill_diagonal(logits, -1e9)
    return _softmax(logits, axis=1)


class InwardMultimodalMediaANN:
    def __init__(self, config: MediaANNConfig | None = None):
        self.config = config or MediaANNConfig()
        self.config.validate()
        self.encoder = MultimodalEncoder(self.config)
        self.autoencoder = TinyAutoencoder(
            self.config.feature_dim,
            self.config.latent_dim,
            self.config.hidden_dim,
        )

    def encode_inputs(self, inputs: Mapping[Modality, Sequence[Any]]) -> list[Particle]:
        particles = [
            self.encoder.encode(modality, value, f"{modality.value}:{index}")
            for modality, values in inputs.items()
            for index, value in enumerate(values)
        ]
        if not particles:
            raise ValueError("at least one input is required")
        return particles

    def inward_target(self, particle: Particle) -> tuple[np.ndarray, np.ndarray]:
        _, latent = self.autoencoder.encode(particle.feature[None, :])
        _, reconstructed = self.autoencoder.decode(latent)
        reconstructed_feature = _normalize(reconstructed[0])
        latent_position = np.zeros(3)
        width = min(3, latent.shape[1])
        latent_position[:width] = latent[0, :width]
        latent_position = np.tanh(latent_position)
        q = self.config.q_phi
        position = q * particle.position + (1.0 - q) * latent_position
        feature = _normalize(q * particle.feature + (1.0 - q) * reconstructed_feature)
        return position, feature

    def run(self, inputs: Mapping[Modality, Sequence[Any]], task: Any) -> MediaANNResult:
        particles = self.encode_inputs(inputs)
        features = np.stack([particle.feature for particle in particles])
        training_loss = self.autoencoder.train(
            features,
            self.config.train_epochs,
            self.config.learning_rate,
        )
        task_particle = self.encoder.encode(Modality.TEXT, task, "task")
        memory_position = np.mean(np.stack([p.position for p in particles]), axis=0)
        memory_feature = _normalize(np.mean(np.stack([p.feature for p in particles]), axis=0))
        metrics: list[MediaANNMetrics] = []
        start = time.perf_counter()

        for step in range(self.config.inner_steps + 1):
            positions = np.stack([p.position for p in particles])
            features = np.stack([p.feature for p in particles])
            graph = attention(particles, self.config)
            residuals: list[float] = []
            reconstruction_errors: list[float] = []
            for particle in particles:
                target_position, target_feature = self.inward_target(particle)
                residuals.append(math.sqrt(
                    float(np.sum((target_position - particle.position) ** 2))
                    + float(np.sum((target_feature - particle.feature) ** 2))
                ))
                _, latent = self.autoencoder.encode(particle.feature[None, :])
                _, reconstructed = self.autoencoder.decode(latent)
                reconstruction_errors.append(float(np.mean((reconstructed[0] - particle.feature) ** 2)))
            position_center = positions.mean(0)
            feature_center = _normalize(features.mean(0))
            metrics.append(MediaANNMetrics(
                step=step,
                fixed_point_residual=float(np.mean(residuals)),
                consensus_spread=float(np.sqrt(np.mean(np.sum((positions - position_center) ** 2, axis=1)))),
                feature_spread=float(np.sqrt(np.mean(np.sum((features - feature_center) ** 2, axis=1)))),
                reconstruction_mse=float(np.mean(reconstruction_errors)),
                elapsed_ms=(time.perf_counter() - start) * 1000.0,
            ))
            if step == self.config.inner_steps:
                break
            self_gain = 1.0 - (
                self.config.task_gain + self.config.swarm_gain
                + self.config.inward_gain + self.config.memory_gain
            )
            updated: list[Particle] = []
            for index, particle in enumerate(particles):
                target_position, target_feature = self.inward_target(particle)
                graph_position = graph[index] @ positions
                graph_feature = _normalize(graph[index] @ features)
                candidate_position = (
                    self_gain * particle.position
                    + self.config.task_gain * task_particle.position
                    + self.config.swarm_gain * graph_position
                    + self.config.inward_gain * target_position
                    + self.config.memory_gain * memory_position
                )
                candidate_feature = _normalize(
                    self_gain * particle.feature
                    + self.config.task_gain * task_particle.feature
                    + self.config.swarm_gain * graph_feature
                    + self.config.inward_gain * target_feature
                    + self.config.memory_gain * memory_feature
                )
                position = (1.0 - self.config.dt) * particle.position + self.config.dt * candidate_position
                feature = _normalize((1.0 - self.config.dt) * particle.feature + self.config.dt * candidate_feature)
                updated.append(Particle(
                    particle.modality,
                    np.clip(position, -1.0, 1.0),
                    feature,
                    particle.source_id,
                ))
            particles = updated
            positions = np.stack([p.position for p in particles])
            features = np.stack([p.feature for p in particles])
            rho = self.config.memory_retention
            memory_position = rho * memory_position + (1.0 - rho) * positions.mean(0)
            memory_feature = _normalize(rho * memory_feature + (1.0 - rho) * features.mean(0))

        score = self.score(metrics[-1])
        return MediaANNResult(tuple(particles), tuple(metrics), training_loss, score)

    @staticmethod
    def score(metrics: MediaANNMetrics) -> float:
        penalty = (
            4.0 * metrics.fixed_point_residual
            + 2.0 * metrics.consensus_spread
            + 1.5 * metrics.feature_spread
            + 3.0 * metrics.reconstruction_mse
        )
        return 1.0 / (1.0 + penalty)


def consensus_latent(result: MediaANNResult) -> np.ndarray:
    return np.mean(np.stack([particle.position for particle in result.particles]), axis=0)


def generate_rgb_frame(latent: np.ndarray, phase: float, size: int = 256) -> np.ndarray:
    """Generate one RGB frame from the converged 3D latent state."""
    yy, xx = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    radius = np.sqrt(xx * xx + yy * yy)
    angle = np.arctan2(yy, xx)
    z0, z1, z2 = [float(value) for value in latent]
    field = 0.5 + 0.5 * np.sin(16 * radius - phase - z0 * math.pi) * np.exp(-2.2 * radius)
    field += 0.18 * np.cos(8 * angle + 2 * phase + z1 * math.pi)
    field += 0.12 * np.sin(5 * xx * math.cos(phase) - 5 * yy * math.sin(phase) + z2 * math.pi)
    field = (field - field.min()) / (np.ptp(field) + 1e-12)
    red = np.clip(field, 0.0, 1.0)
    green = np.clip(np.roll(field, int(10 + 8 * z1), axis=0), 0.0, 1.0)
    blue = np.clip(np.roll(field, int(10 + 8 * z2), axis=1), 0.0, 1.0)
    return np.uint8(np.stack([red, green, blue], axis=-1) * 255)


def generate_pcm_audio(
    latent: np.ndarray,
    duration_seconds: float,
    sample_rate: int = 16_000,
) -> np.ndarray:
    """Generate signed 16-bit PCM samples from the same 3D latent state."""
    timeline = np.arange(int(duration_seconds * sample_rate)) / sample_rate
    f0 = 180 + 100 * (latent[0] + 1)
    f1 = 300 + 140 * (latent[1] + 1)
    f2 = 420 + 160 * (latent[2] + 1)
    signal = (
        0.50 * np.sin(2 * np.pi * f0 * timeline)
        + 0.23 * np.sin(2 * np.pi * f1 * timeline)
        + 0.12 * np.sin(2 * np.pi * f2 * timeline)
    )
    signal /= np.max(np.abs(signal)) + 1e-12
    return np.int16(signal * 30_000)


def mutate_config(config: MediaANNConfig, rng: np.random.Generator) -> MediaANNConfig:
    def perturb(value: float, low: float, high: float, sigma: float) -> float:
        return float(np.clip(value + rng.normal(0.0, sigma), low, high))
    candidate = replace(
        config,
        task_gain=perturb(config.task_gain, 0.05, 0.30, 0.02),
        swarm_gain=perturb(config.swarm_gain, 0.03, 0.28, 0.02),
        inward_gain=perturb(config.inward_gain, 0.18, 0.58, 0.03),
        memory_gain=perturb(config.memory_gain, 0.01, 0.14, 0.012),
        q_phi=perturb(config.q_phi, 0.20, 0.75, 0.035),
        geometry_gain=perturb(config.geometry_gain, 0.05, 0.60, 0.04),
        memory_retention=perturb(config.memory_retention, 0.65, 0.96, 0.025),
    )
    total = candidate.task_gain + candidate.swarm_gain + candidate.inward_gain + candidate.memory_gain
    if total >= 0.95:
        scale = 0.94 / total
        candidate = replace(
            candidate,
            task_gain=candidate.task_gain * scale,
            swarm_gain=candidate.swarm_gain * scale,
            inward_gain=candidate.inward_gain * scale,
            memory_gain=candidate.memory_gain * scale,
        )
    candidate.validate()
    return candidate


def bounded_optimize(
    inputs: Mapping[Modality, Sequence[Any]],
    task: Any,
    initial: MediaANNConfig | None = None,
    *,
    generations: int = 4,
    population: int = 8,
    seed: int = 42,
) -> tuple[MediaANNConfig, MediaANNResult, tuple[dict[str, Any], ...]]:
    rng = np.random.default_rng(seed)
    incumbent = initial or MediaANNConfig()
    incumbent_result = InwardMultimodalMediaANN(incumbent).run(inputs, task)
    audit: list[dict[str, Any]] = []
    for generation in range(generations):
        candidates = [incumbent] + [mutate_config(incumbent, rng) for _ in range(population - 1)]
        best = None
        for index, config in enumerate(candidates):
            result = InwardMultimodalMediaANN(config).run(inputs, task)
            fixed_point_ok = (
                result.metrics[-1].fixed_point_residual
                <= incumbent_result.metrics[-1].fixed_point_residual + 1e-12
            )
            improvement_ok = result.score > incumbent_result.score * 1.0025
            promoted = bool(fixed_point_ok and improvement_ok and np.isfinite(result.score))
            audit.append({
                "generation": generation,
                "candidate": index,
                "score": result.score,
                "fixed_point_residual": result.metrics[-1].fixed_point_residual,
                "promoted": promoted,
                "config": asdict(config),
            })
            if promoted and (best is None or result.score > best[1].score):
                best = (config, result)
        if best is not None:
            incumbent, incumbent_result = best
    return incumbent, incumbent_result, tuple(audit)


__all__ = [
    "MediaANNConfig",
    "MediaANNMetrics",
    "MediaANNResult",
    "Modality",
    "Particle",
    "TinyAutoencoder",
    "InwardMultimodalMediaANN",
    "VIRTUAL_CELLS",
    "VIRTUAL_SIDE",
    "bounded_optimize",
    "byte_features",
    "consensus_latent",
    "generate_pcm_audio",
    "generate_rgb_frame",
    "mutate_config",
    "virtual_xyz",
]
