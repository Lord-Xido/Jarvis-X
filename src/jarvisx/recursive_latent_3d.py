"""Deterministic reference mechanics for a recursive 3D latent processor.

The implementation operationalises a bounded dense tensor fixture, a factorised
3D encoder, pairwise spatial attention, fractional-stride trilinear decoding,
structural objective telemetry and transactional recursive commits. It is a
correctness-oriented CPU reference for small grids, not a production tensor
runtime or a trainable neural-network framework.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, cast

Shape3D = tuple[int, int, int]
Vector = tuple[float, ...]
Matrix = tuple[Vector, ...]
ActivationName = Literal["identity", "tanh"]

DEFAULT_FEATURES = (
    "instruction_weight",
    "execution_gradient",
    "latency",
    "attention",
    "error",
)
MAX_REFERENCE_SCALARS = 1_000_000


def _product(shape: Shape3D) -> int:
    return shape[0] * shape[1] * shape[2]


def _validate_shape(shape: Shape3D) -> None:
    if any(axis < 1 for axis in shape):
        raise ValueError("all spatial dimensions must be positive")


def _index(x: int, y: int, z: int, shape: Shape3D) -> int:
    nx, ny, _ = shape
    return x + nx * (y + ny * z)


def _finite_vector(values: Vector, name: str) -> None:
    if not values:
        raise ValueError(f"{name} must not be empty")
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain finite values")


def _validate_matrix(matrix: Matrix, rows: int, columns: int, name: str) -> None:
    if len(matrix) != rows or any(len(row) != columns for row in matrix):
        raise ValueError(f"{name} must have shape ({rows}, {columns})")
    if not all(math.isfinite(value) for row in matrix for value in row):
        raise ValueError(f"{name} must contain finite values")


def _matvec(matrix: Matrix, vector: Vector) -> Vector:
    return tuple(
        sum(weight * value for weight, value in zip(row, vector)) for row in matrix
    )


def _activate(value: float, activation: ActivationName) -> float:
    if activation == "identity":
        return value
    if activation == "tanh":
        return math.tanh(value)
    raise ValueError("unsupported activation")


def _softmax(values: Vector) -> Vector:
    _finite_vector(values, "softmax input")
    maximum = max(values)
    exponentials = tuple(math.exp(value - maximum) for value in values)
    denominator = sum(exponentials)
    return tuple(value / denominator for value in exponentials)


def _rms(values: Vector) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(value * value for value in values) / len(values))


def _identity_matrix(size: int) -> Matrix:
    return tuple(
        tuple(1.0 if row == column else 0.0 for column in range(size))
        for row in range(size)
    )


@dataclass(frozen=True)
class LogicalTensorSpec:
    """Logical tensor metadata that does not allocate the represented volume."""

    shape: Shape3D
    feature_names: tuple[str, ...] = DEFAULT_FEATURES

    def __post_init__(self) -> None:
        _validate_shape(self.shape)
        if not self.feature_names or any(not name for name in self.feature_names):
            raise ValueError("feature names must be non-empty")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature names must be unique")

    @property
    def cell_count(self) -> int:
        return _product(self.shape)

    @property
    def scalar_count(self) -> int:
        return self.cell_count * len(self.feature_names)


@dataclass(frozen=True)
class DenseTensor3D:
    """Small immutable dense tensor with x as the fastest coordinate."""

    shape: Shape3D
    feature_count: int
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        _validate_shape(self.shape)
        if self.feature_count < 1:
            raise ValueError("feature count must be positive")
        expected = _product(self.shape) * self.feature_count
        if expected > MAX_REFERENCE_SCALARS:
            raise ValueError(
                "dense reference allocation exceeds MAX_REFERENCE_SCALARS; "
                "use LogicalTensorSpec or a sparse/accelerated backend"
            )
        if len(self.values) != expected:
            raise ValueError("tensor value count does not match shape and features")
        if not all(math.isfinite(value) for value in self.values):
            raise ValueError("tensor values must be finite")

    @classmethod
    def constant(
        cls, shape: Shape3D, feature_count: int, value: float
    ) -> "DenseTensor3D":
        if not math.isfinite(value):
            raise ValueError("constant value must be finite")
        count = _product(shape) * feature_count
        return cls(shape, feature_count, (float(value),) * count)

    @classmethod
    def from_cells(
        cls,
        shape: Shape3D,
        cells: tuple[Vector, ...],
    ) -> "DenseTensor3D":
        if not cells:
            raise ValueError("cells must not be empty")
        feature_count = len(cells[0])
        if any(len(cell) != feature_count for cell in cells):
            raise ValueError("every cell must have the same feature count")
        if len(cells) != _product(shape):
            raise ValueError("cell count does not match tensor shape")
        values = tuple(value for cell in cells for value in cell)
        return cls(shape, feature_count, values)

    def _offset(self, x: int, y: int, z: int, feature: int) -> int:
        nx, ny, nz = self.shape
        if not (0 <= x < nx and 0 <= y < ny and 0 <= z < nz):
            raise IndexError("tensor coordinate out of range")
        if not 0 <= feature < self.feature_count:
            raise IndexError("feature index out of range")
        return _index(x, y, z, self.shape) * self.feature_count + feature

    def at(self, x: int, y: int, z: int, feature: int) -> float:
        return self.values[self._offset(x, y, z, feature)]

    def cell(self, x: int, y: int, z: int) -> Vector:
        start = self._offset(x, y, z, 0)
        return self.values[start : start + self.feature_count]

    def feature_mean(self, feature: int) -> float:
        if not 0 <= feature < self.feature_count:
            raise IndexError("feature index out of range")
        return sum(self.values[feature :: self.feature_count]) / _product(self.shape)

    def blend(self, other: "DenseTensor3D", weight: float) -> "DenseTensor3D":
        if self.shape != other.shape or self.feature_count != other.feature_count:
            raise ValueError("tensor shapes and feature counts must match")
        if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
            raise ValueError("blend weight must be finite and in [0, 1]")
        values = tuple(
            (1.0 - weight) * left + weight * right
            for left, right in zip(self.values, other.values)
        )
        return DenseTensor3D(self.shape, self.feature_count, values)

    def difference(self, other: "DenseTensor3D") -> Vector:
        if self.shape != other.shape or self.feature_count != other.feature_count:
            raise ValueError("tensor shapes and feature counts must match")
        return tuple(right - left for left, right in zip(self.values, other.values))


@dataclass(frozen=True)
class LatentField3D:
    shape: Shape3D
    channels: int
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        _validate_shape(self.shape)
        if self.channels < 1:
            raise ValueError("latent channel count must be positive")
        expected = _product(self.shape) * self.channels
        if len(self.values) != expected:
            raise ValueError("latent value count does not match shape and channels")
        if not all(math.isfinite(value) for value in self.values):
            raise ValueError("latent values must be finite")

    @property
    def token_count(self) -> int:
        return _product(self.shape)

    def cell(self, x: int, y: int, z: int) -> Vector:
        nx, ny, nz = self.shape
        if not (0 <= x < nx and 0 <= y < ny and 0 <= z < nz):
            raise IndexError("latent coordinate out of range")
        start = _index(x, y, z, self.shape) * self.channels
        return self.values[start : start + self.channels]

    def token(self, token_index: int) -> Vector:
        if not 0 <= token_index < self.token_count:
            raise IndexError("latent token index out of range")
        start = token_index * self.channels
        return self.values[start : start + self.channels]


@dataclass(frozen=True)
class LatentPosterior3D:
    mean: LatentField3D
    log_variance: LatentField3D

    def __post_init__(self) -> None:
        if self.mean.shape != self.log_variance.shape:
            raise ValueError("posterior fields must have matching shapes")
        if self.mean.channels != self.log_variance.channels:
            raise ValueError("posterior fields must have matching channels")
        if any(abs(value) > 40.0 for value in self.log_variance.values):
            raise ValueError("posterior log variance must remain in [-40, 40]")

    @property
    def kl_standard_normal(self) -> float:
        terms = tuple(
            math.exp(log_variance) + mean * mean - 1.0 - log_variance
            for mean, log_variance in zip(self.mean.values, self.log_variance.values)
        )
        return 0.5 * sum(terms) / len(terms)


@dataclass(frozen=True)
class FactorizedEncoder3D:
    """Spatial kernel followed by a feature-to-latent projection."""

    input_features: int
    latent_channels: int
    kernel_shape: Shape3D
    spatial_weights: Vector
    projection: Matrix
    bias: Vector
    log_variance: Vector
    stride: Shape3D = (2, 2, 2)
    activation: ActivationName = "tanh"

    def __post_init__(self) -> None:
        if self.input_features < 1 or self.latent_channels < 1:
            raise ValueError("encoder feature counts must be positive")
        _validate_shape(self.kernel_shape)
        if any(axis % 2 == 0 for axis in self.kernel_shape):
            raise ValueError("encoder kernel dimensions must be odd")
        if len(self.spatial_weights) != _product(self.kernel_shape):
            raise ValueError("spatial weight count does not match kernel shape")
        _finite_vector(self.spatial_weights, "encoder spatial weights")
        if abs(sum(self.spatial_weights)) < 1.0e-12:
            raise ValueError("encoder spatial weights must have a non-zero sum")
        _validate_matrix(
            self.projection,
            self.latent_channels,
            self.input_features,
            "encoder projection",
        )
        if len(self.bias) != self.latent_channels:
            raise ValueError("encoder bias length does not match latent channels")
        if len(self.log_variance) != self.latent_channels:
            raise ValueError("encoder log-variance length does not match latent channels")
        _finite_vector(self.bias, "encoder bias")
        _finite_vector(self.log_variance, "encoder log variance")
        _validate_shape(self.stride)
        if self.activation not in ("identity", "tanh"):
            raise ValueError("unsupported encoder activation")

    def output_shape(self, input_shape: Shape3D) -> Shape3D:
        return cast(
            Shape3D,
            tuple(
                (axis + stride - 1) // stride
                for axis, stride in zip(input_shape, self.stride)
            ),
        )

    def encode(self, field: DenseTensor3D) -> LatentPosterior3D:
        if field.feature_count != self.input_features:
            raise ValueError("field feature count does not match encoder input")
        output_shape = self.output_shape(field.shape)
        nx, ny, nz = field.shape
        kx_size, ky_size, kz_size = self.kernel_shape
        half_x, half_y, half_z = kx_size // 2, ky_size // 2, kz_size // 2
        normalizer = sum(self.spatial_weights)
        mean_values: list[float] = []
        log_variance_values: list[float] = []

        for oz in range(output_shape[2]):
            for oy in range(output_shape[1]):
                for ox in range(output_shape[0]):
                    center_x = ox * self.stride[0]
                    center_y = oy * self.stride[1]
                    center_z = oz * self.stride[2]
                    pooled = [0.0] * self.input_features
                    weight_index = 0
                    for dz in range(-half_z, half_z + 1):
                        for dy in range(-half_y, half_y + 1):
                            for dx in range(-half_x, half_x + 1):
                                weight = self.spatial_weights[weight_index]
                                weight_index += 1
                                x = (center_x + dx) % nx
                                y = (center_y + dy) % ny
                                z = (center_z + dz) % nz
                                for feature in range(self.input_features):
                                    pooled[feature] += weight * field.at(
                                        x, y, z, feature
                                    )
                    pooled_vector = tuple(value / normalizer for value in pooled)
                    projected = _matvec(self.projection, pooled_vector)
                    mean_values.extend(
                        _activate(value + bias, self.activation)
                        for value, bias in zip(projected, self.bias)
                    )
                    log_variance_values.extend(self.log_variance)

        mean = LatentField3D(output_shape, self.latent_channels, tuple(mean_values))
        log_variance = LatentField3D(
            output_shape,
            self.latent_channels,
            tuple(log_variance_values),
        )
        return LatentPosterior3D(mean, log_variance)


@dataclass(frozen=True)
class AttentionTelemetry:
    mean_entropy: float
    maximum_weight: float
    maximum_row_sum_error: float


@dataclass(frozen=True)
class AttentionResult:
    field: LatentField3D
    telemetry: AttentionTelemetry


@dataclass(frozen=True)
class SpatialSelfAttention3D:
    channels: int
    query_projection: Matrix
    key_projection: Matrix
    value_projection: Matrix
    output_projection: Matrix
    residual_weight: float = 0.5
    max_tokens: int = 512

    def __post_init__(self) -> None:
        if self.channels < 1:
            raise ValueError("attention channel count must be positive")
        for matrix, name in (
            (self.query_projection, "query projection"),
            (self.key_projection, "key projection"),
            (self.value_projection, "value projection"),
            (self.output_projection, "output projection"),
        ):
            _validate_matrix(matrix, self.channels, self.channels, name)
        if (
            not math.isfinite(self.residual_weight)
            or not 0.0 <= self.residual_weight <= 1.0
        ):
            raise ValueError("attention residual weight must be in [0, 1]")
        if self.max_tokens < 1:
            raise ValueError("attention max_tokens must be positive")

    def apply(self, field: LatentField3D) -> AttentionResult:
        if field.channels != self.channels:
            raise ValueError("latent channels do not match attention channels")
        if field.token_count > self.max_tokens:
            raise ValueError(
                "global attention token count exceeds max_tokens; "
                "use windowed or sparse attention"
            )

        tokens = tuple(field.token(index) for index in range(field.token_count))
        queries = tuple(_matvec(self.query_projection, token) for token in tokens)
        keys = tuple(_matvec(self.key_projection, token) for token in tokens)
        values = tuple(_matvec(self.value_projection, token) for token in tokens)
        scale = math.sqrt(self.channels)
        outputs: list[float] = []
        entropies: list[float] = []
        maximum_weight = 0.0
        maximum_row_sum_error = 0.0

        for query, source in zip(queries, tokens):
            scores = tuple(
                sum(left * right for left, right in zip(query, key)) / scale
                for key in keys
            )
            weights = _softmax(scores)
            maximum_weight = max(maximum_weight, max(weights))
            maximum_row_sum_error = max(
                maximum_row_sum_error, abs(sum(weights) - 1.0)
            )
            entropies.append(
                -sum(
                    weight * math.log(max(weight, 1.0e-300))
                    for weight in weights
                )
            )
            attended = tuple(
                sum(weight * value[channel] for weight, value in zip(weights, values))
                for channel in range(self.channels)
            )
            projected = _matvec(self.output_projection, attended)
            outputs.extend(
                (1.0 - self.residual_weight) * original
                + self.residual_weight * transformed
                for original, transformed in zip(source, projected)
            )

        telemetry = AttentionTelemetry(
            sum(entropies) / len(entropies),
            maximum_weight,
            maximum_row_sum_error,
        )
        return AttentionResult(
            LatentField3D(field.shape, field.channels, tuple(outputs)),
            telemetry,
        )


@dataclass(frozen=True)
class FractionalStrideDecoder3D:
    latent_channels: int
    output_features: int
    projection: Matrix
    bias: Vector
    activation: ActivationName = "identity"

    def __post_init__(self) -> None:
        if self.latent_channels < 1 or self.output_features < 1:
            raise ValueError("decoder feature counts must be positive")
        _validate_matrix(
            self.projection,
            self.output_features,
            self.latent_channels,
            "decoder projection",
        )
        if len(self.bias) != self.output_features:
            raise ValueError("decoder bias length does not match output features")
        _finite_vector(self.bias, "decoder bias")
        if self.activation not in ("identity", "tanh"):
            raise ValueError("unsupported decoder activation")

    @staticmethod
    def _sample(field: LatentField3D, x: float, y: float, z: float) -> Vector:
        nx, ny, nz = field.shape
        x0 = math.floor(x)
        y0 = math.floor(y)
        z0 = math.floor(z)
        tx, ty, tz = x - x0, y - y0, z - z0
        output = [0.0] * field.channels
        for dz in (0, 1):
            wz = (1.0 - tz) if dz == 0 else tz
            for dy in (0, 1):
                wy = (1.0 - ty) if dy == 0 else ty
                for dx in (0, 1):
                    wx = (1.0 - tx) if dx == 0 else tx
                    weight = wx * wy * wz
                    cell = field.cell(
                        (x0 + dx) % nx,
                        (y0 + dy) % ny,
                        (z0 + dz) % nz,
                    )
                    for channel, value in enumerate(cell):
                        output[channel] += weight * value
        return tuple(output)

    def decode(self, field: LatentField3D, target_shape: Shape3D) -> DenseTensor3D:
        if field.channels != self.latent_channels:
            raise ValueError("latent channels do not match decoder channels")
        _validate_shape(target_shape)
        scale = tuple(
            target / latent for target, latent in zip(target_shape, field.shape)
        )
        values: list[float] = []
        for z in range(target_shape[2]):
            latent_z = (z + 0.5) / scale[2] - 0.5
            for y in range(target_shape[1]):
                latent_y = (y + 0.5) / scale[1] - 0.5
                for x in range(target_shape[0]):
                    latent_x = (x + 0.5) / scale[0] - 0.5
                    sampled = self._sample(field, latent_x, latent_y, latent_z)
                    projected = _matvec(self.projection, sampled)
                    values.extend(
                        _activate(value + bias, self.activation)
                        for value, bias in zip(projected, self.bias)
                    )
        return DenseTensor3D(target_shape, self.output_features, tuple(values))


@dataclass(frozen=True)
class ObjectiveBreakdown:
    reconstruction: float
    kl_regularization: float
    latency_gain: float
    total: float


@dataclass(frozen=True)
class StructuralObjective:
    lambda_kl: float = 1.0e-3
    lambda_performance: float = 0.0
    latency_feature: int = 2

    def __post_init__(self) -> None:
        if not math.isfinite(self.lambda_kl) or self.lambda_kl < 0.0:
            raise ValueError("lambda_kl must be finite and non-negative")
        if (
            not math.isfinite(self.lambda_performance)
            or self.lambda_performance < 0.0
        ):
            raise ValueError("lambda_performance must be finite and non-negative")
        if self.latency_feature < 0:
            raise ValueError("latency feature index must be non-negative")

    def evaluate(
        self,
        current: DenseTensor3D,
        candidate: DenseTensor3D,
        posterior: LatentPosterior3D,
    ) -> ObjectiveBreakdown:
        if self.latency_feature >= current.feature_count:
            raise ValueError("latency feature index exceeds tensor feature count")
        residual = current.difference(candidate)
        reconstruction = sum(value * value for value in residual) / len(residual)
        kl = posterior.kl_standard_normal
        latency_gain = current.feature_mean(
            self.latency_feature
        ) - candidate.feature_mean(self.latency_feature)
        total = (
            reconstruction
            + self.lambda_kl * kl
            - self.lambda_performance * latency_gain
        )
        return ObjectiveBreakdown(reconstruction, kl, latency_gain, total)


@dataclass(frozen=True)
class FeatureBound:
    feature: int
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self) -> None:
        if self.feature < 0:
            raise ValueError("feature bound index must be non-negative")
        if self.minimum is not None and not math.isfinite(self.minimum):
            raise ValueError("feature lower bound must be finite")
        if self.maximum is not None and not math.isfinite(self.maximum):
            raise ValueError("feature upper bound must be finite")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("feature lower bound exceeds upper bound")


@dataclass(frozen=True)
class VerificationResult:
    accepted: bool
    reasons: tuple[str, ...]
    update_rms: float
    maximum_absolute_update: float


@dataclass(frozen=True)
class TransitionGuard:
    maximum_update_rms: float = math.inf
    maximum_absolute_update: float = math.inf
    feature_bounds: tuple[FeatureBound, ...] = ()

    def __post_init__(self) -> None:
        if self.maximum_update_rms <= 0.0 or math.isnan(self.maximum_update_rms):
            raise ValueError("maximum_update_rms must be positive")
        if (
            self.maximum_absolute_update <= 0.0
            or math.isnan(self.maximum_absolute_update)
        ):
            raise ValueError("maximum_absolute_update must be positive")

    def verify(
        self, current: DenseTensor3D, candidate: DenseTensor3D
    ) -> VerificationResult:
        update = current.difference(candidate)
        update_rms = _rms(update)
        maximum_absolute_update = max(abs(value) for value in update)
        reasons: list[str] = []
        if update_rms > self.maximum_update_rms:
            reasons.append("update RMS exceeds guard")
        if maximum_absolute_update > self.maximum_absolute_update:
            reasons.append("maximum absolute update exceeds guard")
        for bound in self.feature_bounds:
            if bound.feature >= candidate.feature_count:
                reasons.append(f"feature bound {bound.feature} exceeds feature count")
                continue
            values = candidate.values[bound.feature :: candidate.feature_count]
            if bound.minimum is not None and min(values) < bound.minimum:
                reasons.append(f"feature {bound.feature} violates lower bound")
            if bound.maximum is not None and max(values) > bound.maximum:
                reasons.append(f"feature {bound.feature} violates upper bound")
        return VerificationResult(
            not reasons,
            tuple(reasons),
            update_rms,
            maximum_absolute_update,
        )


@dataclass(frozen=True)
class EngineInstruction:
    sequence: int
    opcode: str
    detail: str


@dataclass(frozen=True)
class CycleResult:
    input_field: DenseTensor3D
    posterior: LatentPosterior3D
    tuned_latent: LatentField3D
    decoded_field: DenseTensor3D
    proposed_field: DenseTensor3D
    committed_field: DenseTensor3D
    objective: ObjectiveBreakdown
    attention: AttentionTelemetry
    verification: VerificationResult
    trace: tuple[EngineInstruction, ...]


@dataclass(frozen=True)
class RecursionResult:
    final_field: DenseTensor3D
    cycles: tuple[CycleResult, ...]
    committed_cycles: int
    rolled_back_cycles: int


@dataclass(frozen=True)
class RecursiveLatentEngine3D:
    encoder: FactorizedEncoder3D
    attention: SpatialSelfAttention3D
    decoder: FractionalStrideDecoder3D
    objective: StructuralObjective = field(default_factory=StructuralObjective)
    guard: TransitionGuard = field(default_factory=TransitionGuard)
    commit_blend: float = 1.0

    def __post_init__(self) -> None:
        if self.encoder.latent_channels != self.attention.channels:
            raise ValueError("encoder and attention latent dimensions must match")
        if self.encoder.latent_channels != self.decoder.latent_channels:
            raise ValueError("encoder and decoder latent dimensions must match")
        if self.encoder.input_features != self.decoder.output_features:
            raise ValueError("encoder input and decoder output features must match")
        if (
            not math.isfinite(self.commit_blend)
            or not 0.0 < self.commit_blend <= 1.0
        ):
            raise ValueError("commit_blend must be in (0, 1]")

    def cycle(self, field: DenseTensor3D) -> CycleResult:
        trace: list[EngineInstruction] = []

        def emit(opcode: str, detail: str) -> None:
            trace.append(EngineInstruction(len(trace), opcode, detail))

        emit(
            "LOAD_STRUCTURAL_TENSOR",
            f"shape={field.shape}; features={field.feature_count}",
        )
        posterior = self.encoder.encode(field)
        emit(
            "ENCODE_3D_FACTORISED",
            f"latent_shape={posterior.mean.shape}; channels={posterior.mean.channels}",
        )
        attention_result = self.attention.apply(posterior.mean)
        emit(
            "ATTEND_3D_PAIRWISE",
            (
                f"tokens={posterior.mean.token_count}; "
                f"entropy={attention_result.telemetry.mean_entropy:.6g}"
            ),
        )
        decoded = self.decoder.decode(attention_result.field, field.shape)
        scale = tuple(
            target / latent for target, latent in zip(field.shape, posterior.mean.shape)
        )
        emit("DECODE_FRACTIONAL_STRIDE", f"coordinate_scale={scale}")
        proposed = field.blend(decoded, self.commit_blend)
        objective = self.objective.evaluate(field, proposed, posterior)
        emit(
            "EVALUATE_MOAGI_OBJECTIVE",
            (
                f"reconstruction={objective.reconstruction:.6g}; "
                f"kl={objective.kl_regularization:.6g}; "
                f"latency_gain={objective.latency_gain:.6g}; "
                f"total={objective.total:.6g}"
            ),
        )
        verification = self.guard.verify(field, proposed)
        emit(
            "VERIFY_TRANSITION",
            (
                f"accepted={verification.accepted}; "
                f"update_rms={verification.update_rms:.6g}; "
                f"max_update={verification.maximum_absolute_update:.6g}"
            ),
        )
        committed = proposed if verification.accepted else field
        emit(
            "COMMIT" if verification.accepted else "ROLLBACK",
            "; ".join(verification.reasons),
        )
        return CycleResult(
            field,
            posterior,
            attention_result.field,
            decoded,
            proposed,
            committed,
            objective,
            attention_result.telemetry,
            verification,
            tuple(trace),
        )

    def run(self, field: DenseTensor3D, cycles: int) -> RecursionResult:
        if cycles < 1:
            raise ValueError("cycles must be positive")
        current = field
        results: list[CycleResult] = []
        committed = 0
        rolled_back = 0
        for _ in range(cycles):
            result = self.cycle(current)
            results.append(result)
            current = result.committed_field
            if result.verification.accepted:
                committed += 1
            else:
                rolled_back += 1
        return RecursionResult(current, tuple(results), committed, rolled_back)


def reference_projection(rows: int, columns: int) -> Matrix:
    """Return a deterministic bounded projection with diagonal preference."""

    if rows < 1 or columns < 1:
        raise ValueError("projection dimensions must be positive")
    matrix: list[Vector] = []
    for row in range(rows):
        weights = []
        for column in range(columns):
            if row % columns == column:
                weights.append(1.0)
            else:
                weights.append(0.05 / max(1, columns - 1))
        matrix.append(tuple(weights))
    return tuple(matrix)


def make_reference_engine(
    input_features: int = len(DEFAULT_FEATURES),
    latent_channels: int = 3,
    stride: Shape3D = (2, 2, 2),
    *,
    commit_blend: float = 0.5,
    maximum_update_rms: float = math.inf,
    maximum_absolute_update: float = math.inf,
    lambda_kl: float = 1.0e-3,
    lambda_performance: float = 0.0,
) -> RecursiveLatentEngine3D:
    """Construct the deterministic small-grid reference engine."""

    kernel_shape = (3, 3, 3)
    spatial_weights = (1.0,) * _product(kernel_shape)
    encoder = FactorizedEncoder3D(
        input_features,
        latent_channels,
        kernel_shape,
        spatial_weights,
        reference_projection(latent_channels, input_features),
        (0.0,) * latent_channels,
        (-4.0,) * latent_channels,
        stride,
    )
    identity = _identity_matrix(latent_channels)
    attention = SpatialSelfAttention3D(
        latent_channels,
        identity,
        identity,
        identity,
        identity,
    )
    decoder = FractionalStrideDecoder3D(
        latent_channels,
        input_features,
        reference_projection(input_features, latent_channels),
        (0.0,) * input_features,
    )
    objective = StructuralObjective(lambda_kl, lambda_performance)
    guard = TransitionGuard(maximum_update_rms, maximum_absolute_update)
    return RecursiveLatentEngine3D(
        encoder,
        attention,
        decoder,
        objective,
        guard,
        commit_blend,
    )
