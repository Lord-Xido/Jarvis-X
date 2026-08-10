"""Bounded reference implementation of the Dr Moagi 3D codec transaction.

The module implements the executable subset of ADR-002 without claiming that virtual
macro depth is physical throughput. The reference transform is deliberately simple:
it mean-centres a scalar 3D field, quantizes the residual, entropy-codes the discrete
latents with zlib, validates a versioned bitstream, reconstructs the field, measures
rate/distortion, applies a bounded admissibility gate, and atomically commits adaptive
statistics only when the candidate transaction is valid.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import struct
import zlib
from dataclasses import dataclass
from time import perf_counter

_MAGIC = b"JX3D"
_FORMAT_VERSION = 1
_CODEC_VERSION = 1
_ENTROPY_VERSION = 1
_HEADER = struct.Struct(">4sHHHIIIddQQ32s")
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


class CodecFormatError(ValueError):
    """Raised when an encoded Dr Moagi bitstream violates the format contract."""


@dataclass(frozen=True)
class Volume3D:
    """Dense scalar 3D correctness fixture stored in deterministic x-major order."""

    shape: tuple[int, int, int]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.shape) != 3:
            raise ValueError("shape must contain exactly three dimensions")
        if any(dimension <= 0 for dimension in self.shape):
            raise ValueError("all shape dimensions must be positive")
        expected = self.shape[0] * self.shape[1] * self.shape[2]
        if len(self.values) != expected:
            raise ValueError(f"volume contains {len(self.values)} values; expected {expected}")
        if not all(math.isfinite(float(value)) for value in self.values):
            raise ValueError("volume values must be finite")

    @property
    def voxel_count(self) -> int:
        return len(self.values)

    @property
    def mean(self) -> float:
        return math.fsum(self.values) / self.voxel_count

    def mse(self, other: "Volume3D") -> float:
        if self.shape != other.shape:
            raise ValueError("cannot compare volumes with different shapes")
        squared_error = math.fsum(
            (left - right) ** 2 for left, right in zip(self.values, other.values)
        )
        return squared_error / self.voxel_count


@dataclass(frozen=True)
class CodecConfig:
    """Resource and admissibility limits for the bounded reference runtime."""

    quant_step: float = 0.25
    zlib_level: int = 9
    max_voxels: int = 1_000_000
    max_bitstream_bytes: int = 128 * 1024 * 1024
    max_abs_value: float = 1.0e12
    max_anchor_mse: float | None = None
    max_rate_bpv: float | None = None
    omega_decay: float = 0.90
    virtual_depth: int = 1
    max_virtual_depth: int = 1_000_000

    def __post_init__(self) -> None:
        if not math.isfinite(self.quant_step) or self.quant_step <= 0.0:
            raise ValueError("quant_step must be finite and positive")
        if not 0 <= self.zlib_level <= 9:
            raise ValueError("zlib_level must be between 0 and 9")
        if self.max_voxels < 1:
            raise ValueError("max_voxels must be positive")
        if self.max_bitstream_bytes < _HEADER.size:
            raise ValueError("max_bitstream_bytes is smaller than the codec header")
        if not math.isfinite(self.max_abs_value) or self.max_abs_value <= 0.0:
            raise ValueError("max_abs_value must be finite and positive")
        if self.max_anchor_mse is not None and (
            not math.isfinite(self.max_anchor_mse) or self.max_anchor_mse < 0.0
        ):
            raise ValueError("max_anchor_mse must be finite and non-negative")
        if self.max_rate_bpv is not None and (
            not math.isfinite(self.max_rate_bpv) or self.max_rate_bpv <= 0.0
        ):
            raise ValueError("max_rate_bpv must be finite and positive")
        if not 0.0 <= self.omega_decay < 1.0:
            raise ValueError("omega_decay must be in [0, 1)")
        if not 1 <= self.virtual_depth <= self.max_virtual_depth:
            raise ValueError("virtual_depth is outside the configured bound")


@dataclass(frozen=True)
class BitstreamMetadata:
    shape: tuple[int, int, int]
    quant_step: float
    mean: float
    latent_count: int
    payload_bytes: int
    payload_digest_sha256: str
    format_version: int = _FORMAT_VERSION
    codec_version: int = _CODEC_VERSION
    entropy_version: int = _ENTROPY_VERSION


@dataclass(frozen=True)
class CodecMetrics:
    local_mse: float
    anchor_mse: float
    rate_bpv: float
    bitstream_bytes: int
    raw_latent_bytes: int
    compression_ratio: float


@dataclass(frozen=True)
class CodecMemory:
    """Bounded Omega_codec state, separate from the VM provenance journal."""

    generation: int = 0
    accepted_transactions: int = 0
    ewma_local_mse: float = 0.0
    ewma_anchor_mse: float = 0.0
    ewma_rate_bpv: float = 0.0
    last_bitstream_digest_sha256: str = ""


@dataclass(frozen=True)
class CodecTransactionResult:
    committed: bool
    reconstructed: Volume3D
    bitstream: bytes
    metadata: BitstreamMetadata
    metrics: CodecMetrics
    memory_before: CodecMemory
    memory_after: CodecMemory
    rejection_reason: str | None
    virtual_depth: int
    measured_microsteps_executed: int
    wall_clock_seconds: float

    @property
    def measured_throughput_voxels_per_second(self) -> float:
        if self.wall_clock_seconds <= 0.0:
            return 0.0
        return self.reconstructed.voxel_count / self.wall_clock_seconds


def _round_nearest(value: float) -> int:
    """Deterministic round-to-nearest with halves away from zero."""

    if value >= 0.0:
        return math.floor(value + 0.5)
    return math.ceil(value - 0.5)


class DrMoagiCodec3D:
    """Transactional bounded 3D codec reference aligned with ADR-002."""

    def __init__(self, config: CodecConfig | None = None) -> None:
        self.config = config or CodecConfig()
        self._memory = CodecMemory()
        self._anchor: Volume3D | None = None

    @property
    def memory(self) -> CodecMemory:
        return self._memory

    @property
    def anchor(self) -> Volume3D | None:
        return self._anchor

    def _validate_volume(self, volume: Volume3D) -> None:
        if volume.voxel_count > self.config.max_voxels:
            raise ValueError("volume exceeds max_voxels")
        if any(abs(value) > self.config.max_abs_value for value in volume.values):
            raise ValueError("volume exceeds max_abs_value")

    def encode(self, volume: Volume3D) -> bytes:
        """Encode one 3D scalar field into a deterministic versioned bitstream."""

        self._validate_volume(volume)
        mean = volume.mean
        quant_step = self.config.quant_step
        latents: list[int] = []
        for value in volume.values:
            latent = _round_nearest((value - mean) / quant_step)
            if not _INT64_MIN <= latent <= _INT64_MAX:
                raise OverflowError("quantized latent is outside signed 64-bit range")
            latents.append(latent)

        raw = struct.pack(f">{len(latents)}q", *latents)
        payload = zlib.compress(raw, self.config.zlib_level)
        payload_digest = hashlib.sha256(payload).digest()
        header = _HEADER.pack(
            _MAGIC,
            _FORMAT_VERSION,
            _CODEC_VERSION,
            _ENTROPY_VERSION,
            *volume.shape,
            quant_step,
            mean,
            len(latents),
            len(payload),
            payload_digest,
        )
        bitstream = header + payload
        if len(bitstream) > self.config.max_bitstream_bytes:
            raise ValueError("encoded bitstream exceeds max_bitstream_bytes")
        return bitstream

    def inspect(self, bitstream: bytes) -> BitstreamMetadata:
        """Validate the fixed header and return its deterministic metadata."""

        if len(bitstream) < _HEADER.size:
            raise CodecFormatError("bitstream is shorter than the codec header")
        if len(bitstream) > self.config.max_bitstream_bytes:
            raise CodecFormatError("bitstream exceeds max_bitstream_bytes")

        (
            magic,
            format_version,
            codec_version,
            entropy_version,
            x_size,
            y_size,
            z_size,
            quant_step,
            mean,
            latent_count,
            payload_bytes,
            expected_digest,
        ) = _HEADER.unpack_from(bitstream)

        if magic != _MAGIC:
            raise CodecFormatError("invalid codec magic")
        if format_version != _FORMAT_VERSION:
            raise CodecFormatError("unsupported bitstream format version")
        if codec_version != _CODEC_VERSION:
            raise CodecFormatError("incompatible codec architecture version")
        if entropy_version != _ENTROPY_VERSION:
            raise CodecFormatError("incompatible entropy model version")
        if not math.isfinite(quant_step) or quant_step <= 0.0:
            raise CodecFormatError("invalid quantizer parameter")
        if not math.isfinite(mean) or abs(mean) > self.config.max_abs_value:
            raise CodecFormatError("invalid encoded mean")

        shape = (x_size, y_size, z_size)
        if any(dimension <= 0 for dimension in shape):
            raise CodecFormatError("invalid tensor shape")
        expected_count = x_size * y_size * z_size
        if latent_count != expected_count:
            raise CodecFormatError("latent count does not match tensor shape")
        if latent_count > self.config.max_voxels:
            raise CodecFormatError("latent count exceeds max_voxels")
        if payload_bytes != len(bitstream) - _HEADER.size:
            raise CodecFormatError("payload length does not match bitstream length")

        payload = bitstream[_HEADER.size :]
        observed_digest = hashlib.sha256(payload).digest()
        if not hmac.compare_digest(observed_digest, expected_digest):
            raise CodecFormatError("payload integrity digest mismatch")

        return BitstreamMetadata(
            shape=shape,
            quant_step=quant_step,
            mean=mean,
            latent_count=latent_count,
            payload_bytes=payload_bytes,
            payload_digest_sha256=observed_digest.hex(),
            format_version=format_version,
            codec_version=codec_version,
            entropy_version=entropy_version,
        )

    def decode(self, bitstream: bytes) -> Volume3D:
        """Integrity-check and reconstruct a 3D field from its discrete latent bitstream."""

        metadata = self.inspect(bitstream)
        payload = bitstream[_HEADER.size :]
        expected_raw_bytes = metadata.latent_count * 8
        decompressor = zlib.decompressobj()
        raw = decompressor.decompress(payload, expected_raw_bytes + 1)
        if len(raw) != expected_raw_bytes:
            raise CodecFormatError("decompressed latent payload has an invalid size")
        if not decompressor.eof or decompressor.unused_data or decompressor.unconsumed_tail:
            raise CodecFormatError("compressed latent payload has trailing or incomplete data")

        latents = struct.unpack(f">{metadata.latent_count}q", raw)
        values = tuple(metadata.mean + metadata.quant_step * latent for latent in latents)
        if not all(
            math.isfinite(value) and abs(value) <= self.config.max_abs_value for value in values
        ):
            raise CodecFormatError("decoded values exceed numerical admissibility bounds")
        return Volume3D(metadata.shape, values)

    def _metrics(
        self,
        source: Volume3D,
        reconstructed: Volume3D,
        anchor: Volume3D,
        bitstream: bytes,
    ) -> CodecMetrics:
        local_mse = source.mse(reconstructed)
        anchor_mse = anchor.mse(reconstructed)
        raw_latent_bytes = source.voxel_count * 8
        rate_bpv = (8.0 * len(bitstream)) / source.voxel_count
        compression_ratio = raw_latent_bytes / len(bitstream)
        return CodecMetrics(
            local_mse=local_mse,
            anchor_mse=anchor_mse,
            rate_bpv=rate_bpv,
            bitstream_bytes=len(bitstream),
            raw_latent_bytes=raw_latent_bytes,
            compression_ratio=compression_ratio,
        )

    def _admissibility_error(self, metrics: CodecMetrics) -> str | None:
        if not all(
            math.isfinite(value)
            for value in (metrics.local_mse, metrics.anchor_mse, metrics.rate_bpv)
        ):
            return "non-finite codec telemetry"
        if metrics.bitstream_bytes > self.config.max_bitstream_bytes:
            return "bitstream exceeds configured resource ceiling"
        if self.config.max_anchor_mse is not None and metrics.anchor_mse > self.config.max_anchor_mse:
            return "anchor distortion exceeds configured ceiling"
        if self.config.max_rate_bpv is not None and metrics.rate_bpv > self.config.max_rate_bpv:
            return "encoded rate exceeds configured ceiling"
        return None

    def _commit_memory(self, metrics: CodecMetrics, bitstream: bytes) -> CodecMemory:
        decay = self.config.omega_decay
        new_weight = 1.0 - decay
        if self._memory.accepted_transactions == 0:
            local_ewma = metrics.local_mse
            anchor_ewma = metrics.anchor_mse
            rate_ewma = metrics.rate_bpv
        else:
            local_ewma = decay * self._memory.ewma_local_mse + new_weight * metrics.local_mse
            anchor_ewma = decay * self._memory.ewma_anchor_mse + new_weight * metrics.anchor_mse
            rate_ewma = decay * self._memory.ewma_rate_bpv + new_weight * metrics.rate_bpv
        return CodecMemory(
            generation=self._memory.generation + 1,
            accepted_transactions=self._memory.accepted_transactions + 1,
            ewma_local_mse=local_ewma,
            ewma_anchor_mse=anchor_ewma,
            ewma_rate_bpv=rate_ewma,
            last_bitstream_digest_sha256=hashlib.sha256(bitstream).hexdigest(),
        )

    def process(self, volume: Volume3D, *, anchor: Volume3D | None = None) -> CodecTransactionResult:
        """Execute one authoritative codec transaction with commit-or-rollback semantics."""

        started = perf_counter()
        self._validate_volume(volume)
        candidate_anchor = self._anchor or anchor or volume
        if candidate_anchor.shape != volume.shape:
            raise ValueError("anchor shape must match the source volume")
        self._validate_volume(candidate_anchor)

        memory_before = self._memory
        bitstream = self.encode(volume)
        metadata = self.inspect(bitstream)
        reconstructed = self.decode(bitstream)
        metrics = self._metrics(volume, reconstructed, candidate_anchor, bitstream)
        rejection_reason = self._admissibility_error(metrics)

        if rejection_reason is None:
            memory_after = self._commit_memory(metrics, bitstream)
            self._memory = memory_after
            if self._anchor is None:
                self._anchor = candidate_anchor
            committed = True
        else:
            # Rejection telemetry is returned, but authoritative Omega_codec state remains
            # unchanged. This is the codec equivalent of the VM transaction rollback.
            memory_after = memory_before
            committed = False

        elapsed = perf_counter() - started
        return CodecTransactionResult(
            committed=committed,
            reconstructed=reconstructed,
            bitstream=bitstream,
            metadata=metadata,
            metrics=metrics,
            memory_before=memory_before,
            memory_after=memory_after,
            rejection_reason=rejection_reason,
            virtual_depth=self.config.virtual_depth,
            measured_microsteps_executed=1,
            wall_clock_seconds=elapsed,
        )
