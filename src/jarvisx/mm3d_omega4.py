"""Bounded operational MM3D-AED-BCE-Ω⁴ reference kernel.

Preserves the intended Ξ→encode→explore→decode→ΠΛ→Ω→Θ cycle while
separating the 50T conceptual target from the executable allocation profile.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import struct
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

MODULUS = 8
VOXEL_BITS = 384
VOXEL_BYTES = 48
TOKEN_BYTES, VISUAL_BYTES, AUDIO_BYTES, MOTION_BYTES = 16, 12, 8, 6
VOXEL_DTYPE = np.dtype(
    [
        ("token_embedding", np.int8, (TOKEN_BYTES,)),
        ("visual_feature", np.int8, (VISUAL_BYTES,)),
        ("audio_spectral", np.int8, (AUDIO_BYTES,)),
        ("motion_vector", np.int8, (MOTION_BYTES,)),
        ("attention_weight", "<f4"),
        ("ethical_flag", np.uint8),
        ("reserved", np.uint8),
    ],
    align=False,
)
assert VOXEL_DTYPE.itemsize == VOXEL_BYTES


class Modality(Enum):
    TEXT = auto()
    IMAGE = auto()
    AUDIO = auto()
    VIDEO = auto()
    MOTION = auto()


@dataclass(frozen=True)
class MM3DConfig:
    xi_size: int = 32
    manifold_dim: int = 4096
    latent_size: int = 8
    codebook_size: int = 1024
    codebook_dim: int = 16
    projection_rank: int = 32
    metric_rank: int = 16
    exploration_depth: int = 6
    render_image_size: int = 32
    render_video_frames: int = 4
    render_audio_samples: int = 2048
    omega_capacity: int = 10000
    target_cycle_ms: float = 13.7
    seed: int = 0x4D4D3344
    max_reference_bytes: int = 512 * 1024 * 1024
    conceptual_parameters_total: int = 50_000_000_000_000
    conceptual_active_fraction: float = 0.005

    @property
    def latent_tokens(self) -> int:
        return self.latent_size**3

    @property
    def conceptual_parameters_active(self) -> int:
        return int(self.conceptual_parameters_total * self.conceptual_active_fraction)

    def estimated_reference_bytes(self) -> int:
        latent_flat = self.latent_tokens * self.codebook_dim
        scalars = (
            self.manifold_dim * self.metric_rank
            + self.metric_rank
            + self.projection_rank * self.manifold_dim
            + 2 * latent_flat * self.projection_rank
            + self.manifold_dim * self.projection_rank
            + self.codebook_size * self.codebook_dim
        )
        return int(4 * scalars + self.xi_size**3 * (VOXEL_BYTES + 1))

    def validate(self) -> None:
        dimensions = (
            self.xi_size,
            self.manifold_dim,
            self.latent_size,
            self.codebook_size,
            self.codebook_dim,
            self.projection_rank,
            self.metric_rank,
            self.exploration_depth,
            self.render_image_size,
            self.render_video_frames,
            self.render_audio_samples,
            self.omega_capacity,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("MM3D dimensions and capacities must be positive")
        if not 0.0 < self.conceptual_active_fraction <= 1.0:
            raise ValueError("conceptual_active_fraction must be in (0, 1]")
        if self.estimated_reference_bytes() > self.max_reference_bytes:
            raise ValueError("reference configuration exceeds allocation guard")


@dataclass
class Voxel:
    """Exact 384-bit voxel; original feature dimensions are interpreted as bits."""

    token_embedding: np.ndarray = field(
        default_factory=lambda: np.zeros(TOKEN_BYTES, dtype=np.int8)
    )
    visual_feature: np.ndarray = field(
        default_factory=lambda: np.zeros(VISUAL_BYTES, dtype=np.int8)
    )
    audio_spectral: np.ndarray = field(
        default_factory=lambda: np.zeros(AUDIO_BYTES, dtype=np.int8)
    )
    motion_vector: np.ndarray = field(
        default_factory=lambda: np.zeros(MOTION_BYTES, dtype=np.int8)
    )
    attention_weight: np.float32 = np.float32(0.0)
    ethical_flag: np.uint8 = np.uint8(0)
    reserved: np.uint8 = np.uint8(0)

    def __post_init__(self) -> None:
        for name, width in (
            ("token_embedding", TOKEN_BYTES),
            ("visual_feature", VISUAL_BYTES),
            ("audio_spectral", AUDIO_BYTES),
            ("motion_vector", MOTION_BYTES),
        ):
            value = np.asarray(getattr(self, name), dtype=np.int8)
            if value.shape != (width,):
                raise ValueError(f"{name} must have shape ({width},)")
            setattr(self, name, value.copy())
        self.attention_weight = np.float32(self.attention_weight)
        self.ethical_flag = np.uint8(self.ethical_flag)
        self.reserved = np.uint8(self.reserved)

    def to_bytes(self) -> bytes:
        payload = b"".join(
            (
                self.token_embedding.tobytes(),
                self.visual_feature.tobytes(),
                self.audio_spectral.tobytes(),
                self.motion_vector.tobytes(),
                struct.pack(
                    "<fBB",
                    float(self.attention_weight),
                    int(self.ethical_flag),
                    int(self.reserved),
                ),
            )
        )
        if len(payload) != VOXEL_BYTES:
            raise RuntimeError("voxel serialization violated the 48-byte contract")
        return payload

    @classmethod
    def from_bytes(cls, data: bytes) -> "Voxel":
        if len(data) != VOXEL_BYTES:
            raise ValueError("voxel payload must be exactly 48 bytes")
        boundaries = (0, 16, 28, 36, 42)
        attention, ethical, reserved = struct.unpack("<fBB", data[42:48])
        return cls(
            np.frombuffer(data[boundaries[0] : boundaries[1]], dtype=np.int8).copy(),
            np.frombuffer(data[boundaries[1] : boundaries[2]], dtype=np.int8).copy(),
            np.frombuffer(data[boundaries[2] : boundaries[3]], dtype=np.int8).copy(),
            np.frombuffer(data[boundaries[3] : boundaries[4]], dtype=np.int8).copy(),
            attention,
            ethical,
            reserved,
        )


class Z8QCA:
    def __init__(self, size: int, seed: int = 0) -> None:
        if size <= 0:
            raise ValueError("QCA size must be positive")
        self.size = size
        self.state = np.random.default_rng(seed).integers(
            0, MODULUS, size=(size, size, size), dtype=np.uint8
        )

    def laplacian(self) -> np.ndarray:
        state = self.state.astype(np.int16)
        neighbors = sum(
            np.roll(state, shift, axis=axis)
            for axis in range(3)
            for shift in (-1, 1)
        )
        return neighbors - 6 * state

    def update(self, diffusion_gain: int = 1, reaction_gain: int = 1) -> None:
        if min(diffusion_gain, reaction_gain) < 0:
            raise ValueError("QCA gains must be non-negative")
        state = self.state.astype(np.int16)
        reaction = state * (MODULUS - state) // 2
        self.state = np.mod(
            state + diffusion_gain * self.laplacian() + reaction_gain * reaction,
            MODULUS,
        ).astype(np.uint8)

    def get_region(
        self, x: int, y: int, z: int, w: int, h: int, d: int
    ) -> np.ndarray:
        if min(w, h, d) <= 0:
            raise ValueError("region dimensions must be positive")
        axes = [
            np.mod(np.arange(start, start + width), self.size)
            for start, width in ((x, w), (y, h), (z, d))
        ]
        return self.state[np.ix_(*axes)].copy()


class XiCube:
    """Contiguous 48-byte voxel lattice; no millions of Python objects."""

    def __init__(self, size: int, seed: int = 0) -> None:
        self.size = size
        self.data = np.zeros((size, size, size), dtype=VOXEL_DTYPE)
        self.qca = Z8QCA(size, seed)

    @property
    def allocated_bytes(self) -> int:
        return int(self.data.nbytes + self.qca.state.nbytes)

    def _index(self, x: int, y: int, z: int) -> Tuple[int, int, int]:
        return x % self.size, y % self.size, z % self.size

    def load(self, x: int, y: int, z: int) -> Voxel:
        item = self.data[self._index(x, y, z)]
        return Voxel(*(item[name] for name in VOXEL_DTYPE.names or ()))

    def store(self, x: int, y: int, z: int, voxel: Voxel) -> None:
        item = self.data[self._index(x, y, z)]
        for name in VOXEL_DTYPE.names or ():
            item[name] = getattr(voxel, name)

    def encode_region(
        self, region: Tuple[int, int, int, int, int, int]
    ) -> np.ndarray:
        values = self.qca.get_region(*region)
        return values.reshape(-1).astype(np.float32) / np.float32(MODULUS - 1)

    def get_ethical_mask(self) -> np.ndarray:
        return self.data["ethical_flag"] > 0


@dataclass(frozen=True)
class LatentCode:
    codes: np.ndarray
    continuous: Optional[np.ndarray] = None

    def validate(self, config: MM3DConfig) -> None:
        expected = (config.latent_size,) * 3
        if self.codes.shape != expected or self.codes.dtype.kind not in "iu":
            raise ValueError(f"latent codes must be integer {expected}")
        if np.any(self.codes < 0) or np.any(self.codes >= config.codebook_size):
            raise ValueError("latent code outside codebook")
        if self.continuous is not None and self.continuous.shape != (
            config.latent_tokens,
            config.codebook_dim,
        ):
            raise ValueError("continuous latent shape mismatch")


class FactorizedMetric:
    def __init__(self, dimension: int, rank: int, rng: np.random.Generator) -> None:
        self.diagonal = np.ones(dimension, dtype=np.float32)
        self.low_rank = rng.standard_normal(
            (dimension, rank), dtype=np.float32
        ) / np.float32(math.sqrt(dimension))

    def inverse_apply(self, vector: np.ndarray) -> np.ndarray:
        inv_diag = 1.0 / self.diagonal
        dinv_l = inv_diag[:, None] * self.low_rank
        middle = (
            np.eye(self.low_rank.shape[1], dtype=np.float32)
            + self.low_rank.T @ dinv_l
        )
        return (
            inv_diag * vector
            - dinv_l
            @ np.linalg.solve(middle, self.low_rank.T @ (inv_diag * vector))
        ).astype(np.float32)

    @property
    def parameter_count(self) -> int:
        return int(self.diagonal.size + self.low_rank.size)


class GeometricAutoEncoder:
    """Low-rank metric projection plus vector quantization."""

    def __init__(self, config: MM3DConfig) -> None:
        config.validate()
        self.config = config
        rng = np.random.default_rng(config.seed ^ 0xE0C0DE)
        latent_flat = config.latent_tokens * config.codebook_dim

        def matrix(shape: Tuple[int, int], fan_in: int) -> np.ndarray:
            return rng.standard_normal(shape, dtype=np.float32) / np.float32(
                math.sqrt(fan_in)
            )

        self.metric = FactorizedMetric(config.manifold_dim, config.metric_rank, rng)
        self.encode_right = matrix(
            (config.projection_rank, config.manifold_dim), config.manifold_dim
        )
        self.encode_left = matrix(
            (latent_flat, config.projection_rank), config.projection_rank
        )
        self.decode_right = matrix(
            (config.projection_rank, latent_flat), latent_flat
        )
        self.decode_left = matrix(
            (config.manifold_dim, config.projection_rank), config.projection_rank
        )
        self.codebook = rng.standard_normal(
            (config.codebook_size, config.codebook_dim), dtype=np.float32
        ) * np.float32(0.1)
        self.codebook_norm = np.sum(self.codebook * self.codebook, axis=1)

    @property
    def parameter_count(self) -> int:
        arrays = (
            self.encode_right,
            self.encode_left,
            self.decode_right,
            self.decode_left,
            self.codebook,
        )
        return self.metric.parameter_count + sum(int(array.size) for array in arrays)

    @property
    def allocated_bytes(self) -> int:
        arrays = (
            self.metric.diagonal,
            self.metric.low_rank,
            self.encode_right,
            self.encode_left,
            self.decode_right,
            self.decode_left,
            self.codebook,
            self.codebook_norm,
        )
        return sum(int(array.nbytes) for array in arrays)

    def prepare_input(self, values: np.ndarray) -> np.ndarray:
        flat = np.nan_to_num(np.asarray(values, dtype=np.float32).reshape(-1))
        output = np.zeros(self.config.manifold_dim, dtype=np.float32)
        output[: min(flat.size, output.size)] = flat[: output.size]
        return output

    def metric_prepare(self, values: np.ndarray) -> np.ndarray:
        return self.metric.inverse_apply(self.prepare_input(values))

    def project_partition(
        self, values: np.ndarray, start: int, stop: int
    ) -> np.ndarray:
        if not 0 <= start <= stop <= self.config.manifold_dim:
            raise ValueError("invalid manifold partition")
        return self.encode_right[:, start:stop] @ values[start:stop]

    def quantize(self, vectors: np.ndarray, chunk_size: int = 256) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.config.codebook_dim:
            raise ValueError("quantization width mismatch")
        codes = np.empty(vectors.shape[0], dtype=np.int32)
        for start in range(0, len(vectors), chunk_size):
            chunk = vectors[start : start + chunk_size]
            distances = (
                np.sum(chunk * chunk, axis=1)[:, None]
                + self.codebook_norm[None, :]
            )
            distances -= 2.0 * chunk @ self.codebook.T
            codes[start : start + len(chunk)] = np.argmin(distances, axis=1)
        return codes

    def encode_from_hidden(self, hidden: np.ndarray) -> LatentCode:
        continuous = np.tanh(self.encode_left @ np.tanh(hidden)).reshape(
            self.config.latent_tokens, self.config.codebook_dim
        )
        codes = self.quantize(continuous).reshape((self.config.latent_size,) * 3)
        latent = LatentCode(codes.astype(np.int32), continuous)
        latent.validate(self.config)
        return latent

    def encode(self, values: np.ndarray) -> LatentCode:
        metric_values = self.metric_prepare(values)
        return self.encode_from_hidden(self.encode_right @ metric_values)

    def dequantize(self, latent: LatentCode) -> np.ndarray:
        latent.validate(self.config)
        return self.codebook[latent.codes.reshape(-1)]

    def decode(self, latent: LatentCode) -> np.ndarray:
        hidden = np.tanh(self.decode_right @ self.dequantize(latent).reshape(-1))
        reconstructed = self.decode_left @ hidden
        return (reconstructed + 0.01 * np.tanh(reconstructed)).astype(np.float32)


class PhiQAS:
    """Classical deterministic latent candidate exploration, not quantum hardware."""

    def __init__(self, autoencoder: GeometricAutoEncoder) -> None:
        self.autoencoder = autoencoder

    def _score(self, vectors: np.ndarray) -> float:
        codes = self.autoencoder.quantize(vectors)
        quantized = self.autoencoder.codebook[codes]
        error = float(np.mean((vectors - quantized) ** 2))
        grid = vectors.reshape(
            (self.autoencoder.config.latent_size,) * 3 + (-1,)
        )
        smoothness = sum(
            float(np.mean((grid - np.roll(grid, 1, axis=axis)) ** 2))
            for axis in range(3)
        )
        phase = math.cos(float(np.sum(vectors)) / math.sqrt(vectors.size))
        return error + 0.01 * smoothness - 1e-4 * phase

    def explore(self, latent: LatentCode, depth: int) -> LatentCode:
        if depth <= 0:
            raise ValueError("exploration depth must be positive")
        base = (
            latent.continuous.copy()
            if latent.continuous is not None
            else self.autoencoder.dequantize(latent)
        )
        digest = hashlib.sha3_256(
            latent.codes.tobytes() + struct.pack("<I", depth)
        ).digest()
        rng = np.random.default_rng(
            int.from_bytes(digest[:8], "little") ^ self.autoencoder.config.seed
        )
        candidates = [base]
        for index in range(depth):
            noise = rng.standard_normal(base.shape, dtype=np.float32)
            candidates.append(
                np.clip(
                    base + noise * np.float32(0.02 / math.sqrt(index + 1)),
                    -1.0,
                    1.0,
                )
            )
        best = min(
            enumerate(candidates),
            key=lambda item: (self._score(item[1]), item[0]),
        )[1]
        result = LatentCode(
            self.autoencoder.quantize(best)
            .reshape(latent.codes.shape)
            .astype(np.int32),
            best,
        )
        result.validate(self.autoencoder.config)
        return result


class LambdaConstraint:
    def __init__(
        self, policies: Optional[Sequence[Callable[[Any], bool]]] = None
    ) -> None:
        self.policies = list(policies or [])
        self.violation_count = 0

    def require(self, operation: Any) -> None:
        if any(not bool(policy(operation)) for policy in self.policies):
            self.violation_count += 1
            raise RuntimeError("Lambda constraint rejected operation")

    def project(self, state: np.ndarray, mask: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if state.shape != mask.shape:
            raise ValueError("Lambda mask must match state")
        safe = state.copy()
        safe[mask] = 0.0
        return safe


@dataclass(frozen=True)
class OmegaEntry:
    sequence: int
    instruction_hash: str
    voxel_region: Tuple[int, int, int, int, int, int]
    ethical_flag: int
    result_hash: str
    prev_hash: str = ""

    def compute_hash(self) -> str:
        payload = json.dumps(
            {
                "sequence": self.sequence,
                "instruction_hash": self.instruction_hash,
                "voxel_region": self.voxel_region,
                "ethical_flag": self.ethical_flag,
                "result_hash": self.result_hash,
                "prev_hash": self.prev_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha3_256(payload.encode()).hexdigest()


class OmegaMemory:
    GENESIS = "0" * 64

    def __init__(self, capacity: int = 10000) -> None:
        self.capacity = capacity
        self.chain: Deque[Tuple[str, OmegaEntry]] = deque()
        self.anchor_hash = self.GENESIS
        self.head_hash = self.GENESIS
        self.next_sequence = 0

    def append(self, entry: OmegaEntry) -> str:
        if entry.sequence != self.next_sequence:
            raise ValueError("Omega sequence is not contiguous")
        committed = replace(entry, prev_hash=self.head_hash)
        entry_hash = committed.compute_hash()
        if len(self.chain) >= self.capacity:
            self.anchor_hash = self.chain.popleft()[0]
        self.chain.append((entry_hash, committed))
        self.head_hash = entry_hash
        self.next_sequence += 1
        return entry_hash

    def verify(self) -> bool:
        previous = self.anchor_hash
        for entry_hash, entry in self.chain:
            if entry.prev_hash != previous or entry.compute_hash() != entry_hash:
                return False
            previous = entry_hash
        return previous == self.head_hash


class ThetaProjection:
    def __init__(self, config: MM3DConfig) -> None:
        self.config = config

    @staticmethod
    def _repeat(state: np.ndarray, count: int) -> np.ndarray:
        flat = np.asarray(state, dtype=np.float32).reshape(-1)
        if not flat.size:
            raise ValueError("cannot render empty state")
        return np.resize(flat, count)

    def render_text(self, state: np.ndarray, vocab_size: int = 4096) -> str:
        indices = np.argsort(
            -self._repeat(state, vocab_size), kind="stable"
        )[:32]
        return " ".join(str(int(index)) for index in indices)

    def render_image(self, state: np.ndarray) -> np.ndarray:
        size = self.config.render_image_size
        values = self._repeat(state, size * size * 3)
        return (
            np.clip(np.tanh(values) * 127.5 + 127.5, 0, 255)
            .reshape(size, size, 3)
            .astype(np.uint8)
        )

    def render_audio(self, state: np.ndarray) -> np.ndarray:
        values = self._repeat(state, 8)
        frequencies = 110.0 + 770.0 / (1.0 + np.exp(-values[:4]))
        amplitudes = 0.1 * np.tanh(values[4:])
        timeline = (
            np.arange(self.config.render_audio_samples, dtype=np.float32) / 24000.0
        )
        return np.sum(
            [
                amplitude * np.sin(2 * math.pi * frequency * timeline)
                for frequency, amplitude in zip(frequencies, amplitudes)
            ],
            axis=0,
            dtype=np.float32,
        )

    def render_video(self, state: np.ndarray) -> np.ndarray:
        image = self.render_image(state)
        return np.stack(
            [
                np.roll(image, frame, axis=1)
                for frame in range(self.config.render_video_frames)
            ]
        )


@dataclass(frozen=True)
class MM3DCycleResult:
    latent_code: np.ndarray
    safe_state: np.ndarray
    text: str
    image: np.ndarray
    audio: np.ndarray
    video: np.ndarray
    cycle_time_ms: float
    cycle_number: int
    omega_head: str
    xi_state_hash: str
    target_cycle_ms: float
    actual_parameter_count: int
    conceptual_parameter_count: int
    conceptual_active_parameter_count: int
    allocated_bytes: int

    def summary(self) -> Dict[str, Any]:
        return {
            "cycle_number": self.cycle_number,
            "cycle_time_ms": self.cycle_time_ms,
            "target_cycle_ms": self.target_cycle_ms,
            "target_met": self.cycle_time_ms <= self.target_cycle_ms,
            "omega_head": self.omega_head,
            "xi_state_hash": self.xi_state_hash,
            "latent_shape": list(self.latent_code.shape),
            "image_shape": list(self.image.shape),
            "audio_shape": list(self.audio.shape),
            "video_shape": list(self.video.shape),
            "text_preview": self.text[:80],
            "actual_parameter_count": self.actual_parameter_count,
            "conceptual_parameter_count": self.conceptual_parameter_count,
            "conceptual_active_parameter_count": self.conceptual_active_parameter_count,
            "allocated_bytes": self.allocated_bytes,
        }


class CloudNode:
    def __init__(self, node_id: int) -> None:
        self.node_id = node_id
        self.claimed_ranges: List[Tuple[int, int]] = []
        self.executor = ThreadPoolExecutor(max_workers=1)

    async def dispatch(self, func: Callable[..., Any], *args: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, func, *args)

    def claim_range(self, start: int, stop: int) -> None:
        self.claimed_ranges.append((start, stop))

    def close(self) -> None:
        self.executor.shutdown(wait=True)


class MM3DEngine:
    """Bounded operational Ξ→Φ→Λ→Ω→Θ cycle."""

    def __init__(self, config: Optional[MM3DConfig] = None) -> None:
        self.config = config or MM3DConfig()
        self.config.validate()
        self.xi = XiCube(self.config.xi_size, self.config.seed)
        self.encoder = GeometricAutoEncoder(self.config)
        self.qas = PhiQAS(self.encoder)
        self.lam = LambdaConstraint(
            [
                lambda operation: isinstance(operation, dict),
                lambda operation: operation.get("action") == "MM3D.CYCLE",
                lambda operation: int(operation.get("input_elements", 0)) > 0,
            ]
        )
        self.omega = OmegaMemory(self.config.omega_capacity)
        self.theta = ThetaProjection(self.config)
        self.nodes: List[CloudNode] = []
        self.cycle_count = 0

    @property
    def allocated_bytes(self) -> int:
        return self.xi.allocated_bytes + self.encoder.allocated_bytes

    def add_cloud_node(self, node: CloudNode) -> None:
        if any(existing.node_id == node.node_id for existing in self.nodes):
            raise ValueError("duplicate node id")
        self.nodes.append(node)

    def _ethical_mask(self) -> np.ndarray:
        raw = self.xi.get_ethical_mask().reshape(-1)
        mask = np.zeros(self.config.manifold_dim, dtype=bool)
        mask[: min(mask.size, raw.size)] = raw[: mask.size]
        return mask

    def _finish(
        self, latent: LatentCode, operation: Dict[str, Any], started: float
    ) -> MM3DCycleResult:
        optimized = self.qas.explore(latent, self.config.exploration_depth)
        safe = self.lam.project(
            self.encoder.decode(optimized), self._ethical_mask()
        )
        state_hash = hashlib.sha3_256(safe.tobytes()).hexdigest()
        instruction_hash = hashlib.sha3_256(
            json.dumps(operation, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        head = self.omega.append(
            OmegaEntry(
                self.omega.next_sequence,
                instruction_hash,
                (
                    0,
                    0,
                    0,
                    self.config.xi_size,
                    self.config.xi_size,
                    self.config.xi_size,
                ),
                int(np.any(self._ethical_mask())),
                state_hash,
            )
        )
        self.cycle_count += 1
        return MM3DCycleResult(
            optimized.codes.copy(),
            safe,
            self.theta.render_text(safe),
            self.theta.render_image(safe),
            self.theta.render_audio(safe),
            self.theta.render_video(safe),
            (time.perf_counter() - started) * 1000.0,
            self.cycle_count,
            head,
            state_hash[:16],
            self.config.target_cycle_ms,
            self.encoder.parameter_count,
            self.config.conceptual_parameters_total,
            self.config.conceptual_parameters_active,
            self.allocated_bytes,
        )

    def cycle(self, psi_input: np.ndarray, sop: Any = None) -> MM3DCycleResult:
        started = time.perf_counter()
        values = np.asarray(psi_input)
        operation = {
            "action": "MM3D.CYCLE",
            "input_elements": int(values.size),
            "sop_type": type(sop).__name__ if sop is not None else None,
        }
        self.lam.require(operation)
        self.xi.qca.update()
        return self._finish(self.encoder.encode(values), operation, started)

    async def distributed_cycle(
        self, psi_input: np.ndarray, sop: Any = None
    ) -> MM3DCycleResult:
        if not self.nodes:
            return self.cycle(psi_input, sop)
        started = time.perf_counter()
        values = np.asarray(psi_input)
        operation = {
            "action": "MM3D.CYCLE",
            "input_elements": int(values.size),
            "sop_type": type(sop).__name__ if sop is not None else None,
            "distributed_nodes": len(self.nodes),
        }
        self.lam.require(operation)
        metric_values = self.encoder.metric_prepare(values)
        boundaries = np.linspace(
            0,
            self.config.manifold_dim,
            len(self.nodes) + 1,
            dtype=np.int64,
        )
        futures = []
        for index, node in enumerate(self.nodes):
            start, stop = int(boundaries[index]), int(boundaries[index + 1])
            node.claim_range(start, stop)
            futures.append(
                node.dispatch(
                    self.encoder.project_partition,
                    metric_values,
                    start,
                    stop,
                )
            )
        partials = await asyncio.gather(*futures)
        hidden = np.sum(np.stack(partials), axis=0)
        latent = self.encoder.encode_from_hidden(hidden)
        self.xi.qca.update()
        return self._finish(latent, operation, started)

    def close(self) -> None:
        for node in self.nodes:
            node.close()
        self.nodes.clear()
