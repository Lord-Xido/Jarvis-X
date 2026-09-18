"""Bounded 1 GiB Dr Moagi recursive 3D codec reference runtime.

The 1 GiB address space is exact and virtual: 1024^3 one-byte voxels.
Execution is brick-streamed over 64^3-byte cubes, so this reference never needs
to materialize the full GiB at once.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Iterator, Sequence
import zlib

GIB_BYTES = 1 << 30
VOLUME_SIDE = 1024
BRICK_SIDE = 64
BRICK_BYTES = BRICK_SIDE**3
BRICKS_PER_AXIS = VOLUME_SIDE // BRICK_SIDE
BRICK_COUNT = BRICKS_PER_AXIS**3

RAW_HEADER_BYTES = 24
COMPRESSED_HEADER_BYTES = 48

assert VOLUME_SIDE**3 == GIB_BYTES
assert BRICK_COUNT * BRICK_BYTES == GIB_BYTES


def _require_range(name: str, value: int, lower: int, upper: int) -> None:
    if not lower <= value < upper:
        raise ValueError(f"{name}={value} outside [{lower}, {upper})")


@dataclass(frozen=True)
class VolumeGeometry1GiB:
    """Exact 1024^3 byte-addressed geometry with 64^3 streaming bricks."""

    side: int = VOLUME_SIDE
    brick_side: int = BRICK_SIDE

    def __post_init__(self) -> None:
        if self.side != VOLUME_SIDE or self.brick_side != BRICK_SIDE:
            raise ValueError("the canonical 1 GiB profile is fixed at 1024^3 with 64^3 bricks")

    @property
    def volume_bytes(self) -> int:
        return self.side**3

    @property
    def brick_bytes(self) -> int:
        return self.brick_side**3

    @property
    def bricks_per_axis(self) -> int:
        return self.side // self.brick_side

    @property
    def brick_count(self) -> int:
        return self.bricks_per_axis**3

    def xyz_to_offset(self, x: int, y: int, z: int) -> int:
        _require_range("x", x, 0, self.side)
        _require_range("y", y, 0, self.side)
        _require_range("z", z, 0, self.side)
        return x + self.side * (y + self.side * z)

    def offset_to_xyz(self, offset: int) -> tuple[int, int, int]:
        _require_range("offset", offset, 0, self.volume_bytes)
        x = offset % self.side
        q = offset // self.side
        y = q % self.side
        z = q // self.side
        return x, y, z

    def brick_index_to_coords(self, brick_index: int) -> tuple[int, int, int]:
        _require_range("brick_index", brick_index, 0, self.brick_count)
        n = self.bricks_per_axis
        bx = brick_index % n
        q = brick_index // n
        by = q % n
        bz = q // n
        return bx, by, bz

    def brick_origin(self, brick_index: int) -> tuple[int, int, int]:
        bx, by, bz = self.brick_index_to_coords(brick_index)
        s = self.brick_side
        return bx * s, by * s, bz * s


@dataclass(frozen=True)
class CodecPolicy:
    """Bounded policy surface; optimization never changes validation gates."""

    latent_side: int = 8
    compression_level: int = 6
    max_refine_steps: int = 4
    alpha: float = 1.0

    def validate(self) -> None:
        if self.latent_side not in (4, 8, 16, 32):
            raise ValueError("latent_side must be one of 4, 8, 16, 32")
        if BRICK_SIDE % self.latent_side:
            raise ValueError("latent_side must exactly divide the 64^3 brick")
        if not 0 <= self.compression_level <= 9:
            raise ValueError("compression_level must be in [0, 9]")
        if not 1 <= self.max_refine_steps <= 32:
            raise ValueError("max_refine_steps must be in [1, 32]")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")


@dataclass(frozen=True)
class EncodedBrick:
    brick_index: int
    original_size: int
    source_crc32: int
    mode: str
    latent_side: int
    latent_payload: bytes
    residual_payload: bytes
    raw_payload: bytes
    refine_steps: int
    converged: bool

    @property
    def encoded_size(self) -> int:
        if self.mode == "raw":
            return RAW_HEADER_BYTES + len(self.raw_payload)
        if self.mode != "latent+xor":
            raise ValueError(f"unknown frame mode: {self.mode}")
        return COMPRESSED_HEADER_BYTES + len(self.latent_payload) + len(self.residual_payload)


@dataclass(frozen=True)
class Evaluation:
    policy: CodecPolicy
    encoded_size: int
    exact_roundtrip: bool
    mode: str
    refine_steps: int
    converged: bool


@dataclass(frozen=True)
class OptimizationDecision:
    baseline_policy: CodecPolicy
    candidate_policy: CodecPolicy
    baseline_size: int
    candidate_size: int
    exact_roundtrip: bool
    accepted: bool
    reason: str


def _contract_once(cube: bytes, side: int) -> tuple[bytes, int]:
    if side <= 1 or side % 2:
        raise ValueError("cube side must be even and greater than one")
    if len(cube) != side**3:
        raise ValueError("cube byte length does not match side^3")

    out_side = side // 2
    out = bytearray(out_side**3)
    out_i = 0
    plane = side * side
    for z in range(0, side, 2):
        z0 = z * plane
        z1 = (z + 1) * plane
        for y in range(0, side, 2):
            y0 = y * side
            y1 = (y + 1) * side
            for x in range(0, side, 2):
                total = (
                    cube[z0 + y0 + x]
                    + cube[z0 + y0 + x + 1]
                    + cube[z0 + y1 + x]
                    + cube[z0 + y1 + x + 1]
                    + cube[z1 + y0 + x]
                    + cube[z1 + y0 + x + 1]
                    + cube[z1 + y1 + x]
                    + cube[z1 + y1 + x + 1]
                )
                out[out_i] = (total + 4) // 8
                out_i += 1
    return bytes(out), out_side


def _expand_once(cube: bytes, side: int) -> tuple[bytes, int]:
    if len(cube) != side**3:
        raise ValueError("cube byte length does not match side^3")

    out_side = side * 2
    out = bytearray(out_side**3)
    in_plane = side * side
    out_plane = out_side * out_side
    for z in range(side):
        oz = z * 2
        for y in range(side):
            oy = y * 2
            for x in range(side):
                value = cube[z * in_plane + y * side + x]
                ox = x * 2
                for dz in (0, 1):
                    base_z = (oz + dz) * out_plane
                    for dy in (0, 1):
                        base = base_z + (oy + dy) * out_side + ox
                        out[base] = value
                        out[base + 1] = value
    return bytes(out), out_side


def _encode_latent(source: bytes, source_side: int, latent_side: int) -> bytes:
    if latent_side > source_side or source_side % latent_side:
        raise ValueError("latent side must divide source side")
    cube = source
    side = source_side
    while side > latent_side:
        cube, side = _contract_once(cube, side)
    return cube


def _decode_latent(latent: bytes, latent_side: int, target_side: int) -> bytes:
    if target_side < latent_side or target_side % latent_side:
        raise ValueError("target side must be a multiple of latent side")
    cube = latent
    side = latent_side
    while side < target_side:
        cube, side = _expand_once(cube, side)
    return cube


def _xor_bytes(a: bytes, b: bytes) -> bytes:
    if len(a) != len(b):
        raise ValueError("XOR operands must have equal length")
    return bytes(x ^ y for x, y in zip(a, b))


class Recursive3DCodec:
    """Operational, lossless brick codec with bounded inward self-refinement."""

    def __init__(self, policy: CodecPolicy | None = None) -> None:
        self._policy = policy or CodecPolicy()
        self._policy.validate()

    @property
    def policy(self) -> CodecPolicy:
        return self._policy

    def _refine_latent(self, source: bytes, policy: CodecPolicy) -> tuple[bytes, int, bool]:
        latent = _encode_latent(source, BRICK_SIDE, policy.latent_side)
        for step in range(1, policy.max_refine_steps + 1):
            reconstruction = _decode_latent(latent, policy.latent_side, BRICK_SIDE)
            cycled = _encode_latent(reconstruction, BRICK_SIDE, policy.latent_side)
            if cycled == latent:
                return latent, step, True
            alpha = policy.alpha
            latent = bytes(
                max(0, min(255, round((1.0 - alpha) * old + alpha * new)))
                for old, new in zip(latent, cycled)
            )
        return latent, policy.max_refine_steps, False

    def encode_brick(
        self,
        source: bytes,
        brick_index: int = 0,
        *,
        policy: CodecPolicy | None = None,
    ) -> EncodedBrick:
        selected = policy or self._policy
        selected.validate()
        _require_range("brick_index", brick_index, 0, BRICK_COUNT)
        if len(source) != BRICK_BYTES:
            raise ValueError(f"a canonical brick is exactly {BRICK_BYTES} bytes")

        latent, refine_steps, converged = self._refine_latent(source, selected)
        prediction = _decode_latent(latent, selected.latent_side, BRICK_SIDE)
        residual = _xor_bytes(source, prediction)
        level = selected.compression_level
        latent_payload = zlib.compress(latent, level)
        residual_payload = zlib.compress(residual, level)
        compressed_size = COMPRESSED_HEADER_BYTES + len(latent_payload) + len(residual_payload)
        raw_size = RAW_HEADER_BYTES + len(source)
        checksum = zlib.crc32(source) & 0xFFFFFFFF

        if compressed_size >= raw_size:
            return EncodedBrick(
                brick_index=brick_index,
                original_size=len(source),
                source_crc32=checksum,
                mode="raw",
                latent_side=selected.latent_side,
                latent_payload=b"",
                residual_payload=b"",
                raw_payload=source,
                refine_steps=refine_steps,
                converged=converged,
            )

        return EncodedBrick(
            brick_index=brick_index,
            original_size=len(source),
            source_crc32=checksum,
            mode="latent+xor",
            latent_side=selected.latent_side,
            latent_payload=latent_payload,
            residual_payload=residual_payload,
            raw_payload=b"",
            refine_steps=refine_steps,
            converged=converged,
        )

    def decode_brick(self, frame: EncodedBrick) -> bytes:
        _require_range("brick_index", frame.brick_index, 0, BRICK_COUNT)
        if frame.original_size != BRICK_BYTES:
            raise ValueError("frame size does not match canonical brick size")

        if frame.mode == "raw":
            output = frame.raw_payload
        elif frame.mode == "latent+xor":
            latent = zlib.decompress(frame.latent_payload)
            expected_latent = frame.latent_side**3
            if len(latent) != expected_latent:
                raise ValueError("corrupt latent payload length")
            prediction = _decode_latent(latent, frame.latent_side, BRICK_SIDE)
            residual = zlib.decompress(frame.residual_payload)
            if len(residual) != BRICK_BYTES:
                raise ValueError("corrupt residual payload length")
            output = _xor_bytes(prediction, residual)
        else:
            raise ValueError(f"unknown frame mode: {frame.mode}")

        if len(output) != frame.original_size:
            raise ValueError("decoded output length mismatch")
        if (zlib.crc32(output) & 0xFFFFFFFF) != frame.source_crc32:
            raise ValueError("CRC verification failed")
        return output

    def evaluate(self, source: bytes, policy: CodecPolicy) -> Evaluation:
        frame = self.encode_brick(source, policy=policy)
        exact = self.decode_brick(frame) == source
        return Evaluation(
            policy=policy,
            encoded_size=frame.encoded_size,
            exact_roundtrip=exact,
            mode=frame.mode,
            refine_steps=frame.refine_steps,
            converged=frame.converged,
        )

    def optimize_policy(
        self,
        source: bytes,
        candidates: Sequence[CodecPolicy] | None = None,
    ) -> OptimizationDecision:
        baseline = self.evaluate(source, self._policy)
        candidate_policies = candidates or tuple(
            replace(self._policy, latent_side=latent_side, compression_level=level)
            for latent_side in (4, 8, 16, 32)
            for level in (1, 6, 9)
        )

        best = baseline
        for candidate in candidate_policies:
            candidate.validate()
            evaluation = self.evaluate(source, candidate)
            if not evaluation.exact_roundtrip:
                continue
            if evaluation.encoded_size < best.encoded_size:
                best = evaluation

        accepted = best.policy != baseline.policy and best.encoded_size < baseline.encoded_size
        reason = (
            "candidate improves encoded size under exact round-trip gate"
            if accepted
            else "no candidate improved the immutable exact-roundtrip baseline"
        )
        return OptimizationDecision(
            baseline_policy=baseline.policy,
            candidate_policy=best.policy,
            baseline_size=baseline.encoded_size,
            candidate_size=best.encoded_size,
            exact_roundtrip=best.exact_roundtrip,
            accepted=accepted,
            reason=reason,
        )

    def promote(self, decision: OptimizationDecision) -> bool:
        """Atomically promote a previously evaluated candidate or keep the stable policy."""
        if decision.baseline_policy != self._policy:
            return False
        if not decision.accepted or not decision.exact_roundtrip:
            return False
        if decision.candidate_size >= decision.baseline_size:
            return False
        decision.candidate_policy.validate()
        self._policy = decision.candidate_policy
        return True

    def encode_stream(
        self,
        bricks: Iterable[bytes],
        *,
        start_brick: int = 0,
    ) -> Iterator[EncodedBrick]:
        """Stream bricks without materializing the 1 GiB logical volume."""
        _require_range("start_brick", start_brick, 0, BRICK_COUNT)
        for offset, brick in enumerate(bricks):
            brick_index = start_brick + offset
            if brick_index >= BRICK_COUNT:
                raise ValueError("stream exceeds the canonical 1 GiB geometry")
            yield self.encode_brick(brick, brick_index=brick_index)
