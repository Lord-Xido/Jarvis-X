from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field

import numpy as np


_MODALITY_VIDEO = 0
_MODALITY_AUDIO = 1
_MODALITY_GRAPH = 2
_MODALITY_CONTEXT = 3


def _array(value: object, *, name: str, ndim: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {result.shape}")
    if result.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    if np.min(result) < 0.0 or np.max(result) > 1.0:
        raise ValueError(f"{name} must be normalized to [0, 1]")
    return result.copy()


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(np.clip(shifted, -60.0, 60.0))
    return ex / np.maximum(np.sum(ex, axis=axis, keepdims=True), 1e-8)


def _layer_norm(x: np.ndarray) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(variance + 1e-5)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = np.asarray(a, dtype=np.float64).reshape(-1)
    b_flat = np.asarray(b, dtype=np.float64).reshape(-1)
    denominator = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if denominator <= 1e-12:
        return 0.0 if np.linalg.norm(a_flat - b_flat) <= 1e-12 else 1.0
    similarity = float(np.dot(a_flat, b_flat) / denominator)
    return float(np.clip(1.0 - similarity, 0.0, 2.0))


def _ssim_scalar(a: np.ndarray, b: np.ndarray) -> float:
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    mean_a = float(np.mean(a64))
    mean_b = float(np.mean(b64))
    var_a = float(np.var(a64))
    var_b = float(np.var(b64))
    covariance = float(np.mean((a64 - mean_a) * (b64 - mean_b)))
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2.0 * mean_a * mean_b + c1) * (2.0 * covariance + c2)
    denominator = (mean_a**2 + mean_b**2 + c1) * (var_a + var_b + c2)
    return float(np.clip(numerator / max(denominator, 1e-12), -1.0, 1.0))


def morton4d_encode(t: int, x: int, y: int, z: int) -> int:
    """Interleave four unsigned 16-bit coordinates into one 64-bit key."""

    coordinates = (t, x, y, z)
    if min(coordinates) < 0 or max(coordinates) >= 2**16:
        raise ValueError("4D Morton coordinates must lie in [0, 2**16)")
    code = 0
    for bit in range(16):
        for axis, value in enumerate(coordinates):
            code |= ((value >> bit) & 1) << (4 * bit + axis)
    return code


def morton4d_decode(code: int) -> tuple[int, int, int, int]:
    if code < 0 or code >= 2**64:
        raise ValueError("4D Morton code must be an unsigned 64-bit integer")
    values = [0, 0, 0, 0]
    for bit in range(16):
        for axis in range(4):
            values[axis] |= ((code >> (4 * bit + axis)) & 1) << bit
    return values[0], values[1], values[2], values[3]


@dataclass(slots=True)
class GraphTensor:
    node_features: np.ndarray
    adjacency: np.ndarray

    def __post_init__(self) -> None:
        self.node_features = _array(self.node_features, name="graph.node_features", ndim=2)
        self.adjacency = _array(self.adjacency, name="graph.adjacency", ndim=2)
        node_count = self.node_features.shape[0]
        if self.adjacency.shape != (node_count, node_count):
            raise ValueError("graph adjacency must be square and match the node count")


@dataclass(slots=True)
class AetherInput:
    video: np.ndarray
    audio: np.ndarray
    graph: GraphTensor
    context: np.ndarray

    def __post_init__(self) -> None:
        self.video = _array(self.video, name="video", ndim=4)
        self.audio = _array(self.audio, name="audio", ndim=2)
        self.context = _array(self.context, name="context", ndim=2)
        if self.video.shape[-1] < 1:
            raise ValueError("video must contain at least one channel")


@dataclass(slots=True)
class AetherOutput:
    video: np.ndarray
    audio: np.ndarray
    graph: GraphTensor
    context: np.ndarray

    def shape_summary(self) -> dict[str, object]:
        return {
            "video": list(self.video.shape),
            "audio": list(self.audio.shape),
            "graph_nodes": list(self.graph.node_features.shape),
            "graph_adjacency": list(self.graph.adjacency.shape),
            "context": list(self.context.shape),
        }


@dataclass(frozen=True, slots=True)
class Sparse4DField:
    coordinates: np.ndarray
    morton_keys: np.ndarray
    modalities: np.ndarray
    features: np.ndarray

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.coordinates, dtype=np.int64)
        keys = np.asarray(self.morton_keys, dtype=np.uint64)
        modalities = np.asarray(self.modalities, dtype=np.int8)
        features = np.asarray(self.features, dtype=np.float32)
        if coordinates.ndim != 2 or coordinates.shape[1] != 4:
            raise ValueError("coordinates must have shape (tokens, 4)")
        token_count = coordinates.shape[0]
        if keys.shape != (token_count,) or modalities.shape != (token_count,):
            raise ValueError("field metadata must match the token count")
        if features.ndim != 2 or features.shape[0] != token_count:
            raise ValueError("field features must have shape (tokens, hidden_dim)")
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "morton_keys", keys)
        object.__setattr__(self, "modalities", modalities)
        object.__setattr__(self, "features", features)


@dataclass(frozen=True, slots=True)
class AetherPolicy:
    evolution: str = "ssm"
    recurrent_steps: int = 2
    cross_modal_gain: float = 0.25

    def __post_init__(self) -> None:
        if self.evolution not in {"ssm", "euler"}:
            raise ValueError("evolution must be 'ssm' or 'euler'")
        if not 1 <= self.recurrent_steps <= 4:
            raise ValueError("recurrent_steps must lie in [1, 4]")
        if not 0.0 <= self.cross_modal_gain <= 1.0:
            raise ValueError("cross_modal_gain must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class AetherConfig:
    hidden_dim: int = 24
    latent_dim: int = 12
    max_tokens: int = 1024
    learning_rate: float = 0.05
    max_update_norm: float = 0.25
    semantic_tolerance: float = 0.75
    min_improvement: float = 1e-8
    seed: int = 7

    def __post_init__(self) -> None:
        if self.hidden_dim < 4 or self.latent_dim < 2:
            raise ValueError("hidden_dim and latent_dim are too small")
        if self.latent_dim >= self.hidden_dim:
            raise ValueError("latent_dim must be smaller than hidden_dim")
        if self.max_tokens < 4:
            raise ValueError("max_tokens must be at least four")
        if self.learning_rate <= 0.0 or self.max_update_norm <= 0.0:
            raise ValueError("learning_rate and max_update_norm must be positive")


@dataclass(frozen=True, slots=True)
class AetherLossWeights:
    reconstruction: float = 0.60
    perceptual: float = 0.15
    semantic: float = 0.15
    efficiency: float = 0.08
    novelty: float = 0.02

    def normalized(self) -> dict[str, float]:
        values = asdict(self)
        total = sum(values.values())
        if total <= 0.0:
            raise ValueError("loss weights must sum to a positive value")
        return {key: value / total for key, value in values.items()}


@dataclass(frozen=True, slots=True)
class AetherLoss:
    total: float
    reconstruction: float
    perceptual: float
    semantic: float
    efficiency: float
    novelty: float
    components: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["components"] = dict(self.components)
        return result


@dataclass(slots=True)
class AetherResult:
    output: AetherOutput
    latent: np.ndarray
    field: Sparse4DField
    loss: AetherLoss
    policy: AetherPolicy
    adapted: bool
    optimized: bool
    base_digest: str
    state_digest: str
    journal: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, object]:
        result: dict[str, object] = {
            "output_shapes": self.output.shape_summary(),
            "latent_shape": list(self.latent.shape),
            "active_4d_tokens": int(self.field.features.shape[0]),
            "loss": self.loss.to_dict(),
            "policy": asdict(self.policy),
            "adapted": self.adapted,
            "optimized": self.optimized,
            "base_digest": self.base_digest,
            "state_digest": self.state_digest,
            "journal": list(self.journal),
        }
        if include_arrays:
            result["output"] = {
                "video": self.output.video.tolist(),
                "audio": self.output.audio.tolist(),
                "graph": {
                    "node_features": self.output.graph.node_features.tolist(),
                    "adjacency": self.output.graph.adjacency.tolist(),
                },
                "context": self.output.context.tolist(),
            }
        return result


@dataclass(frozen=True, slots=True)
class _Descriptor:
    modality: int
    index: tuple[int, ...]


class AetherEngine:
    """Sparse 4D multimodal auto-encoding reference processor.

    The base parameter bank is sealed after first input binding. Online changes
    are stored in a bounded overlay and are activated only after shadow
    execution improves the constrained objective.
    """

    def __init__(
        self,
        config: AetherConfig | None = None,
        *,
        policy: AetherPolicy | None = None,
        loss_weights: AetherLossWeights | None = None,
    ) -> None:
        self.config = config or AetherConfig()
        self.policy = policy or AetherPolicy()
        self.loss_weights = loss_weights or AetherLossWeights()
        self._rng = np.random.default_rng(self.config.seed)
        self._base: dict[str, np.ndarray] = {}
        self._overlay: dict[str, np.ndarray] = {}
        self._signature: tuple[int, int, int, int] | None = None
        self._base_digest = "uninitialized"
        self.journal: list[dict[str, object]] = []

    @property
    def base_digest(self) -> str:
        return self._base_digest

    @property
    def state_digest(self) -> str:
        h = hashlib.sha256()
        h.update(self._base_digest.encode("ascii"))
        for name in sorted(self._overlay):
            h.update(name.encode("utf-8"))
            h.update(np.ascontiguousarray(self._overlay[name]).tobytes())
        h.update(json.dumps(asdict(self.policy), sort_keys=True).encode("utf-8"))
        return h.hexdigest()

    def run(
        self,
        data: AetherInput,
        *,
        adapt: bool = False,
        optimize: bool = False,
    ) -> AetherResult:
        self._ensure_initialized(data)
        adapted = self._adapt(data) if adapt else False
        optimized = self._optimize_policy(data) if optimize else False
        output, field, latent = self._forward(data, self.policy, self._overlay)
        loss = self._loss(data, output, field, latent, self.policy, self._overlay)
        return AetherResult(
            output=output,
            latent=latent,
            field=field,
            loss=loss,
            policy=self.policy,
            adapted=adapted,
            optimized=optimized,
            base_digest=self.base_digest,
            state_digest=self.state_digest,
            journal=list(self.journal),
        )

    def _ensure_initialized(self, data: AetherInput) -> None:
        signature = (
            data.video.shape[-1],
            data.audio.shape[-1],
            data.graph.node_features.shape[-1],
            data.context.shape[-1],
        )
        if self._signature is not None:
            if signature != self._signature:
                raise ValueError(f"input feature signature changed from {self._signature} to {signature}")
            return
        self._signature = signature
        hidden = self.config.hidden_dim
        latent = self.config.latent_dim

        def weight(name: str, shape: tuple[int, ...], scale: float | None = None) -> None:
            sigma = scale if scale is not None else math.sqrt(2.0 / max(shape[0], 1))
            value = self._rng.normal(0.0, sigma, shape).astype(np.float32)
            self._base[name] = value
            self._overlay[name] = np.zeros_like(value)

        for modality, input_dim in enumerate(signature):
            weight(f"proj_{modality}", (input_dim, hidden))
            weight(f"out_{modality}", (hidden, input_dim))
            self._base[f"bias_{modality}"] = np.zeros(input_dim, dtype=np.float32)
            self._overlay[f"bias_{modality}"] = np.zeros(input_dim, dtype=np.float32)

        weight("modality_embedding", (4, hidden), 0.1)
        weight("ssm_in", (hidden, hidden))
        self._base["ssm_decay"] = self._rng.uniform(0.55, 0.90, hidden).astype(np.float32)
        self._overlay["ssm_decay"] = np.zeros(hidden, dtype=np.float32)
        weight("kan", (hidden * 3, hidden))
        weight("liquid_in", (hidden, hidden))
        weight("liquid_state", (hidden, hidden), 0.05)
        weight("gate", (hidden, hidden))
        for name in ("q", "k", "v", "attn_out"):
            weight(name, (hidden, hidden))
        weight("to_latent", (hidden, latent))
        weight("from_latent", (latent, hidden))
        weight("latent_in", (latent, latent))
        self._base["latent_decay"] = self._rng.uniform(0.60, 0.92, latent).astype(np.float32)
        self._overlay["latent_decay"] = np.zeros(latent, dtype=np.float32)
        weight("flow", (latent, latent), 0.08)
        weight("flow_condition", (latent, latent), 0.08)
        weight("edge", (latent, latent), 0.08)
        self._base_digest = self._digest_weights(self._base)
        self._journal("SEAL_BASE", {"signature": list(signature), "digest": self._base_digest})

    @staticmethod
    def _digest_weights(weights: dict[str, np.ndarray]) -> str:
        h = hashlib.sha256()
        for name in sorted(weights):
            h.update(name.encode("utf-8"))
            value = np.ascontiguousarray(weights[name])
            h.update(str(value.dtype).encode("ascii"))
            h.update(str(value.shape).encode("ascii"))
            h.update(value.tobytes())
        return h.hexdigest()

    def _w(self, name: str, overlay: dict[str, np.ndarray]) -> np.ndarray:
        return self._base[name] + overlay[name]

    def _tokenize(
        self,
        data: AetherInput,
        overlay: dict[str, np.ndarray],
    ) -> tuple[Sparse4DField, list[_Descriptor]]:
        tokens: list[np.ndarray] = []
        coordinates: list[tuple[int, int, int, int]] = []
        modalities: list[int] = []
        descriptors: list[_Descriptor] = []
        hidden = self.config.hidden_dim

        def append(raw: np.ndarray, coordinate: tuple[int, int, int, int], descriptor: _Descriptor) -> None:
            modality = descriptor.modality
            projected = raw @ self._w(f"proj_{modality}", overlay)
            projected = projected + self._w("modality_embedding", overlay)[modality]
            tokens.append(np.asarray(projected, dtype=np.float32).reshape(hidden))
            coordinates.append(coordinate)
            modalities.append(modality)
            descriptors.append(descriptor)

        time_count, height, width, _ = data.video.shape
        patch_h = max(1, math.ceil(height / 4))
        patch_w = max(1, math.ceil(width / 4))
        for t in range(time_count):
            patch_x = 0
            for h0 in range(0, height, patch_h):
                patch_y = 0
                for w0 in range(0, width, patch_w):
                    h1 = min(height, h0 + patch_h)
                    w1 = min(width, w0 + patch_w)
                    raw = np.mean(data.video[t, h0:h1, w0:w1], axis=(0, 1))
                    append(
                        raw,
                        (t, patch_x, patch_y, _MODALITY_VIDEO),
                        _Descriptor(_MODALITY_VIDEO, (t, h0, h1, w0, w1)),
                    )
                    patch_y += 1
                patch_x += 1

        for t, raw in enumerate(data.audio):
            append(raw, (t, 0, 0, _MODALITY_AUDIO), _Descriptor(_MODALITY_AUDIO, (t,)))

        degrees = np.sum(data.graph.adjacency > 0.0, axis=1).astype(np.int64)
        for node, raw in enumerate(data.graph.node_features):
            degree = int(min(degrees[node], 2**16 - 1))
            append(raw, (node, degree, 0, _MODALITY_GRAPH), _Descriptor(_MODALITY_GRAPH, (node,)))

        for t, raw in enumerate(data.context):
            append(raw, (t, 0, 0, _MODALITY_CONTEXT), _Descriptor(_MODALITY_CONTEXT, (t,)))

        if len(tokens) > self.config.max_tokens:
            indices = np.linspace(0, len(tokens) - 1, self.config.max_tokens, dtype=np.int64)
            tokens = [tokens[index] for index in indices]
            coordinates = [coordinates[index] for index in indices]
            modalities = [modalities[index] for index in indices]
            descriptors = [descriptors[index] for index in indices]

        keys = np.asarray([morton4d_encode(*coordinate) for coordinate in coordinates], dtype=np.uint64)
        order = np.argsort(keys, kind="stable")
        coordinate_array = np.asarray(coordinates, dtype=np.int64)[order]
        modality_array = np.asarray(modalities, dtype=np.int8)[order]
        feature_array = np.asarray(tokens, dtype=np.float32)[order]
        descriptor_ordered = [descriptors[int(index)] for index in order]
        field = Sparse4DField(coordinate_array, keys[order], modality_array, feature_array)
        return field, descriptor_ordered

    def _encode(
        self,
        field: Sparse4DField,
        policy: AetherPolicy,
        overlay: dict[str, np.ndarray],
    ) -> np.ndarray:
        x = field.features
        token_count, hidden = x.shape
        ssm = np.zeros_like(x)
        state = np.zeros(hidden, dtype=np.float32)
        decay = np.clip(self._w("ssm_decay", overlay), 0.0, 0.999)
        ssm_in = self._w("ssm_in", overlay)
        for index in range(token_count):
            state = np.tanh(decay * state + x[index] @ ssm_in)
            ssm[index] = state

        basis = np.concatenate((x, x * x, np.sin(x)), axis=1)
        kan = np.tanh(basis @ self._w("kan", overlay))
        liquid = np.zeros_like(x)
        liquid_state = np.zeros(hidden, dtype=np.float32)
        for index in range(token_count):
            derivative = np.tanh(
                x[index] @ self._w("liquid_in", overlay)
                + liquid_state @ self._w("liquid_state", overlay)
            )
            liquid_state = liquid_state + 0.1 * derivative
            liquid[index] = liquid_state

        gate = _sigmoid(x @ self._w("gate", overlay))
        hybrid = _layer_norm(x + gate * ssm + (1.0 - gate) * kan + 0.1 * liquid)

        q = hybrid @ self._w("q", overlay)
        k = hybrid @ self._w("k", overlay)
        v = hybrid @ self._w("v", overlay)
        logits = q @ k.T / math.sqrt(hidden)
        cross_modal = field.modalities[:, None] != field.modalities[None, :]
        logits = logits + policy.cross_modal_gain * cross_modal.astype(np.float32)
        attention = _softmax(logits, axis=1) @ v
        fused = _layer_norm(hybrid + attention @ self._w("attn_out", overlay))
        return np.tanh(fused @ self._w("to_latent", overlay))

    def _evolve(
        self,
        latent: np.ndarray,
        policy: AetherPolicy,
        overlay: dict[str, np.ndarray],
    ) -> np.ndarray:
        z = latent.copy()
        if policy.evolution == "ssm":
            decay = np.clip(self._w("latent_decay", overlay), 0.0, 0.999)
            transition = self._w("latent_in", overlay)
            for _ in range(policy.recurrent_steps):
                state = np.zeros(z.shape[1], dtype=np.float32)
                next_z = np.zeros_like(z)
                for index in range(z.shape[0]):
                    state = np.tanh(decay * state + z[index] @ transition)
                    next_z[index] = 0.5 * z[index] + 0.5 * state
                z = next_z
            return z

        for _ in range(policy.recurrent_steps):
            condition = np.mean(z, axis=0, keepdims=True)
            derivative = np.tanh(
                z @ self._w("flow", overlay)
                + condition @ self._w("flow_condition", overlay)
            )
            z = np.clip(z + 0.1 * derivative, -1.0, 1.0)
        return z

    def _decode(
        self,
        data: AetherInput,
        latent: np.ndarray,
        descriptors: list[_Descriptor],
        overlay: dict[str, np.ndarray],
    ) -> AetherOutput:
        hidden = np.tanh(latent @ self._w("from_latent", overlay))
        video = np.zeros_like(data.video)
        video_counts = np.zeros(data.video.shape[:-1] + (1,), dtype=np.float32)
        audio = np.zeros_like(data.audio)
        audio_counts = np.zeros((data.audio.shape[0], 1), dtype=np.float32)
        nodes = np.zeros_like(data.graph.node_features)
        node_counts = np.zeros((data.graph.node_features.shape[0], 1), dtype=np.float32)
        context = np.zeros_like(data.context)
        context_counts = np.zeros((data.context.shape[0], 1), dtype=np.float32)
        graph_latents: list[tuple[int, np.ndarray]] = []

        for index, descriptor in enumerate(descriptors):
            modality = descriptor.modality
            predicted = _sigmoid(
                hidden[index] @ self._w(f"out_{modality}", overlay)
                + self._w(f"bias_{modality}", overlay)
            ).astype(np.float32)
            if modality == _MODALITY_VIDEO:
                t, h0, h1, w0, w1 = descriptor.index
                video[t, h0:h1, w0:w1] += predicted
                video_counts[t, h0:h1, w0:w1] += 1.0
            elif modality == _MODALITY_AUDIO:
                (t,) = descriptor.index
                audio[t] += predicted
                audio_counts[t] += 1.0
            elif modality == _MODALITY_GRAPH:
                (node,) = descriptor.index
                nodes[node] += predicted
                node_counts[node] += 1.0
                graph_latents.append((node, latent[index]))
            else:
                (t,) = descriptor.index
                context[t] += predicted
                context_counts[t] += 1.0

        video /= np.maximum(video_counts, 1.0)
        audio /= np.maximum(audio_counts, 1.0)
        nodes /= np.maximum(node_counts, 1.0)
        context /= np.maximum(context_counts, 1.0)

        node_count = data.graph.node_features.shape[0]
        adjacency = np.zeros((node_count, node_count), dtype=np.float32)
        if graph_latents:
            graph_latents.sort(key=lambda item: item[0])
            indices = [item[0] for item in graph_latents]
            z_graph = np.asarray([item[1] for item in graph_latents], dtype=np.float32)
            edge_logits = z_graph @ self._w("edge", overlay) @ z_graph.T / math.sqrt(z_graph.shape[1])
            predicted_edges = _sigmoid(edge_logits).astype(np.float32)
            predicted_edges = 0.5 * (predicted_edges + predicted_edges.T)
            for row, node_i in enumerate(indices):
                for column, node_j in enumerate(indices):
                    adjacency[node_i, node_j] = predicted_edges[row, column]
        np.fill_diagonal(adjacency, 0.0)
        return AetherOutput(video, audio, GraphTensor(nodes, adjacency), context)

    def _forward(
        self,
        data: AetherInput,
        policy: AetherPolicy,
        overlay: dict[str, np.ndarray],
    ) -> tuple[AetherOutput, Sparse4DField, np.ndarray]:
        field, descriptors = self._tokenize(data, overlay)
        latent = self._evolve(self._encode(field, policy, overlay), policy, overlay)
        output = self._decode(data, latent, descriptors, overlay)
        return output, field, latent

    def _loss(
        self,
        data: AetherInput,
        output: AetherOutput,
        field: Sparse4DField,
        latent: np.ndarray,
        policy: AetherPolicy,
        overlay: dict[str, np.ndarray],
    ) -> AetherLoss:
        video_mse = float(np.mean((data.video - output.video) ** 2))
        audio_mse = float(np.mean((data.audio - output.audio) ** 2))
        graph_feature_mse = float(np.mean((data.graph.node_features - output.graph.node_features) ** 2))
        graph_adjacency_mse = float(np.mean((data.graph.adjacency - output.graph.adjacency) ** 2))
        context_mse = float(np.mean((data.context - output.context) ** 2))
        reconstruction = video_mse + audio_mse + graph_feature_mse + graph_adjacency_mse + context_mse

        video_perceptual = 1.0 - _ssim_scalar(data.video, output.video)
        if data.audio.shape[0] > 1:
            original_spectrum = np.abs(np.fft.rfft(data.audio, axis=0))
            output_spectrum = np.abs(np.fft.rfft(output.audio, axis=0))
            audio_spectral = float(np.mean(np.abs(original_spectrum - output_spectrum)))
        else:
            audio_spectral = audio_mse
        perceptual = float(video_perceptual + audio_spectral)

        reconstructed_field, _ = self._tokenize(
            AetherInput(output.video, output.audio, output.graph, output.context),
            overlay,
        )
        semantic = _cosine_distance(np.mean(field.features, axis=0), np.mean(reconstructed_field.features, axis=0))

        token_count = field.features.shape[0]
        hidden = self.config.hidden_dim
        latent_dim = self.config.latent_dim
        estimated_flops = token_count * (10 * hidden * hidden + 4 * hidden * latent_dim)
        estimated_flops += token_count * token_count * hidden * 2
        memory_bytes = field.features.nbytes + latent.nbytes + sum(value.nbytes for value in self._base.values())
        efficiency = (
            min(estimated_flops / 100_000_000.0, 1.0)
            + min(memory_bytes / (64.0 * 1024.0 * 1024.0), 1.0)
            + min(token_count * policy.recurrent_steps / 100_000.0, 1.0)
        ) / 3.0

        flattened = np.concatenate(
            (
                output.video.reshape(-1),
                output.audio.reshape(-1),
                output.graph.node_features.reshape(-1),
                output.context.reshape(-1),
            )
        )
        diversity = float(np.std(flattened))
        semantic_penalty = max(0.0, semantic - self.config.semantic_tolerance) ** 2
        novelty = -min(diversity, 1.0) + 10.0 * semantic_penalty

        normalized_weights = self.loss_weights.normalized()
        total = (
            normalized_weights["reconstruction"] * reconstruction
            + normalized_weights["perceptual"] * perceptual
            + normalized_weights["semantic"] * semantic
            + normalized_weights["efficiency"] * efficiency
            + normalized_weights["novelty"] * novelty
        )
        components = {
            "video_mse": video_mse,
            "audio_mse": audio_mse,
            "graph_feature_mse": graph_feature_mse,
            "graph_adjacency_mse": graph_adjacency_mse,
            "context_mse": context_mse,
            "video_ssim": _ssim_scalar(data.video, output.video),
            "audio_spectral": audio_spectral,
            "diversity": diversity,
            "estimated_flops": float(estimated_flops),
            "memory_bytes": float(memory_bytes),
        }
        return AetherLoss(
            total=float(total),
            reconstruction=float(reconstruction),
            perceptual=float(perceptual),
            semantic=float(semantic),
            efficiency=float(efficiency),
            novelty=float(novelty),
            components=components,
        )

    def _evaluate(
        self,
        data: AetherInput,
        policy: AetherPolicy,
        overlay: dict[str, np.ndarray],
    ) -> tuple[AetherLoss, AetherOutput, Sparse4DField, np.ndarray]:
        output, field, latent = self._forward(data, policy, overlay)
        return self._loss(data, output, field, latent, policy, overlay), output, field, latent

    def _adapt(self, data: AetherInput) -> bool:
        baseline_loss, baseline_output, _, _ = self._evaluate(data, self.policy, self._overlay)
        candidate = {name: value.copy() for name, value in self._overlay.items()}
        residuals = (
            np.mean(data.video - baseline_output.video, axis=(0, 1, 2)),
            np.mean(data.audio - baseline_output.audio, axis=0),
            np.mean(data.graph.node_features - baseline_output.graph.node_features, axis=0),
            np.mean(data.context - baseline_output.context, axis=0),
        )
        raw_updates: dict[str, np.ndarray] = {}
        for modality, residual in enumerate(residuals):
            raw_updates[f"bias_{modality}"] = self.config.learning_rate * residual.astype(np.float32)
        update_norm = math.sqrt(sum(float(np.sum(update * update)) for update in raw_updates.values()))
        scale = min(1.0, self.config.max_update_norm / max(update_norm, 1e-12))
        for name, update in raw_updates.items():
            candidate[name] = candidate[name] + scale * update

        candidate_loss, _, _, _ = self._evaluate(data, self.policy, candidate)
        valid = (
            np.isfinite(candidate_loss.total)
            and candidate_loss.semantic <= self.config.semantic_tolerance
            and candidate_loss.total < baseline_loss.total - self.config.min_improvement
            and update_norm * scale <= self.config.max_update_norm + 1e-8
        )
        event = {
            "baseline_loss": baseline_loss.total,
            "candidate_loss": candidate_loss.total,
            "semantic": candidate_loss.semantic,
            "update_norm": update_norm * scale,
        }
        if valid:
            self._overlay = candidate
            self._journal("ADAPT_COMMIT", event)
            return True
        self._journal("ADAPT_ROLLBACK", event)
        return False

    def _optimize_policy(self, data: AetherInput) -> bool:
        baseline, _, _, _ = self._evaluate(data, self.policy, self._overlay)
        candidates: set[AetherPolicy] = {self.policy}
        alternate = "euler" if self.policy.evolution == "ssm" else "ssm"
        candidates.add(AetherPolicy(alternate, self.policy.recurrent_steps, self.policy.cross_modal_gain))
        for steps in {max(1, self.policy.recurrent_steps - 1), min(4, self.policy.recurrent_steps + 1)}:
            candidates.add(AetherPolicy(self.policy.evolution, steps, self.policy.cross_modal_gain))
        for gain in (0.10, 0.25, 0.40):
            candidates.add(AetherPolicy(self.policy.evolution, self.policy.recurrent_steps, gain))

        best_policy = self.policy
        best_loss = baseline
        evaluated: list[dict[str, object]] = []
        for candidate in sorted(candidates, key=lambda item: (item.evolution, item.recurrent_steps, item.cross_modal_gain)):
            loss, _, _, _ = self._evaluate(data, candidate, self._overlay)
            evaluated.append({"policy": asdict(candidate), "loss": loss.total})
            if loss.semantic <= self.config.semantic_tolerance and loss.total < best_loss.total:
                best_policy = candidate
                best_loss = loss
        if best_policy != self.policy and best_loss.total < baseline.total - self.config.min_improvement:
            previous = self.policy
            self.policy = best_policy
            self._journal(
                "POLICY_COMMIT",
                {
                    "previous": asdict(previous),
                    "candidate": asdict(best_policy),
                    "baseline_loss": baseline.total,
                    "candidate_loss": best_loss.total,
                    "evaluated": evaluated,
                },
            )
            return True
        self._journal(
            "POLICY_ROLLBACK",
            {"active": asdict(self.policy), "baseline_loss": baseline.total, "evaluated": evaluated},
        )
        return False

    def _journal(self, event: str, payload: dict[str, object]) -> None:
        previous_hash = self.journal[-1]["hash"] if self.journal else ""
        record = {
            "sequence": len(self.journal),
            "event": event,
            "payload": payload,
            "previous_hash": previous_hash,
        }
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        record["hash"] = hashlib.sha256(encoded).hexdigest()
        self.journal.append(record)


def synthetic_aether_input(seed: int = 7) -> AetherInput:
    rng = np.random.default_rng(seed)
    video = rng.random((3, 8, 8, 3), dtype=np.float32)
    audio = rng.random((6, 8), dtype=np.float32)
    node_features = rng.random((5, 6), dtype=np.float32)
    adjacency = np.asarray(
        [
            [0, 1, 0, 0, 1],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1],
            [1, 0, 0, 1, 0],
        ],
        dtype=np.float32,
    )
    context = rng.random((4, 10), dtype=np.float32)
    return AetherInput(video, audio, GraphTensor(node_features, adjacency), context)
