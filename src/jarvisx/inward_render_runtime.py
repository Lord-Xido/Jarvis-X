"""Bounded end-to-end bytecode -> SDF -> autoencoder -> ray-march renderer.

This deterministic reference implements the Dr. Moagi inward self-optimizing
rendering equation using only the Python standard library so every arithmetic
stage remains inspectable and portable.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

Vec3 = tuple[float, float, float]
Mat4 = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]
Pixel = tuple[float, float, float]

OP_TRANSLATE = 0x01
OP_ROTATE_Y = 0x02
OP_SCALE = 0x03
OP_HALT = 0xFF


@dataclass(frozen=True)
class RendererConfig:
    grid_size: int = 64
    root_size: float = 4.0
    latent_grid: int = 4
    image_width: int = 32
    image_height: int = 24
    max_ray_steps: int = 64
    max_ray_distance: float = 8.0
    hit_epsilon: float = 0.025
    min_ray_step: float = 0.01
    sphere_radius: float = 0.8
    camera_origin: Vec3 = (0.0, 0.0, 3.2)
    fov_degrees: float = 48.0
    ray_target_steps: float = 24.0
    w_reconstruction: float = 1.0
    w_eikonal: float = 0.15
    w_bytecode: float = 0.01
    w_telemetry: float = 0.05
    finite_difference_epsilon: float = 1.0e-3
    inward_dt: float = 0.05
    gradient_clip: float = 5.0
    non_regression_tolerance: float = 1.0e-10

    def __post_init__(self) -> None:
        if self.grid_size < 4:
            raise ValueError("grid_size must be at least 4")
        if self.root_size <= 0.0:
            raise ValueError("root_size must be positive")
        if self.latent_grid < 1 or self.latent_grid > self.grid_size:
            raise ValueError("latent_grid must be in [1, grid_size]")
        if self.image_width < 1 or self.image_height < 1:
            raise ValueError("image dimensions must be positive")
        if self.max_ray_steps < 1:
            raise ValueError("max_ray_steps must be positive")
        if self.max_ray_distance <= 0.0 or self.hit_epsilon <= 0.0:
            raise ValueError("ray distances and epsilon must be positive")
        if self.sphere_radius <= 0.0:
            raise ValueError("sphere_radius must be positive")
        if self.finite_difference_epsilon <= 0.0 or self.inward_dt <= 0.0:
            raise ValueError("optimizer epsilon and dt must be positive")


@dataclass(frozen=True)
class Instruction:
    opcode: int
    args: tuple[float, ...] = ()


@dataclass(frozen=True)
class BytecodeProgram:
    instructions: tuple[Instruction, ...]

    @classmethod
    def from_bytes(cls, payload: bytes) -> "BytecodeProgram":
        instructions: list[Instruction] = []
        offset = 0
        halted = False
        while offset < len(payload):
            opcode = payload[offset]
            offset += 1
            if opcode == OP_TRANSLATE:
                if offset + 12 > len(payload):
                    raise ValueError("truncated TRANSLATE instruction")
                args = struct.unpack_from("<fff", payload, offset)
                offset += 12
            elif opcode in (OP_ROTATE_Y, OP_SCALE):
                if offset + 4 > len(payload):
                    raise ValueError("truncated scalar instruction")
                args = (struct.unpack_from("<f", payload, offset)[0],)
                offset += 4
            elif opcode == OP_HALT:
                instructions.append(Instruction(opcode))
                halted = True
                if offset != len(payload):
                    raise ValueError("bytes after HALT are not permitted")
                break
            else:
                raise ValueError(f"unsupported opcode 0x{opcode:02X}")
            instructions.append(Instruction(opcode, tuple(float(v) for v in args)))
        if not halted:
            raise ValueError("program must terminate with HALT")
        return cls(tuple(instructions))

    def to_bytes(self) -> bytes:
        output = bytearray()
        for instruction in self.instructions:
            output.append(instruction.opcode)
            if instruction.opcode == OP_TRANSLATE:
                if len(instruction.args) != 3:
                    raise ValueError("TRANSLATE requires three arguments")
                output.extend(struct.pack("<fff", *instruction.args))
            elif instruction.opcode in (OP_ROTATE_Y, OP_SCALE):
                if len(instruction.args) != 1:
                    raise ValueError("scalar opcode requires one argument")
                output.extend(struct.pack("<f", instruction.args[0]))
            elif instruction.opcode == OP_HALT:
                if instruction.args:
                    raise ValueError("HALT takes no arguments")
            else:
                raise ValueError(f"unsupported opcode 0x{instruction.opcode:02X}")
        return bytes(output)

    def parameter_vector(self) -> tuple[float, ...]:
        values: list[float] = []
        for instruction in self.instructions:
            if instruction.opcode != OP_HALT:
                values.extend(instruction.args)
        return tuple(values)

    def with_parameter_vector(self, values: Sequence[float]) -> "BytecodeProgram":
        values = tuple(float(v) for v in values)
        cursor = 0
        updated: list[Instruction] = []
        for instruction in self.instructions:
            count = len(instruction.args)
            if instruction.opcode == OP_HALT:
                updated.append(instruction)
                continue
            if cursor + count > len(values):
                raise ValueError("parameter vector is too short")
            args = tuple(values[cursor : cursor + count])
            cursor += count
            updated.append(Instruction(instruction.opcode, args))
        if cursor != len(values):
            raise ValueError("parameter vector is too long")
        return BytecodeProgram(tuple(updated))

    def parameter_bounds(self) -> tuple[tuple[float, float], ...]:
        bounds: list[tuple[float, float]] = []
        for instruction in self.instructions:
            if instruction.opcode == OP_TRANSLATE:
                bounds.extend(((-1.5, 1.5),) * 3)
            elif instruction.opcode == OP_ROTATE_Y:
                bounds.append((-math.pi, math.pi))
            elif instruction.opcode == OP_SCALE:
                bounds.append((0.25, 3.0))
        return tuple(bounds)


@dataclass(frozen=True)
class VmState:
    matrix: Mat4
    inverse_matrix: Mat4
    uniform_scale: float


@dataclass(frozen=True)
class Volume:
    size: int
    root_size: float
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.values) != self.size**3:
            raise ValueError("volume value count does not match size^3")

    def at(self, x: int, y: int, z: int) -> float:
        return self.values[(z * self.size + y) * self.size + x]

    def sample(self, point: Vec3) -> float:
        half = self.root_size / 2.0
        x, y, z = point
        if x < -half or x > half or y < -half or y > half or z < -half or z > half:
            dx = max(abs(x) - half, 0.0)
            dy = max(abs(y) - half, 0.0)
            dz = max(abs(z) - half, 0.0)
            return math.sqrt(dx * dx + dy * dy + dz * dz) + self.root_size / self.size

        def axis(value: float) -> tuple[int, int, float]:
            u = ((value + half) / self.root_size) * (self.size - 1)
            i0 = max(0, min(self.size - 1, int(math.floor(u))))
            i1 = min(self.size - 1, i0 + 1)
            return i0, i1, u - i0

        x0, x1, fx = axis(x)
        y0, y1, fy = axis(y)
        z0, z1, fz = axis(z)
        c000 = self.at(x0, y0, z0)
        c100 = self.at(x1, y0, z0)
        c010 = self.at(x0, y1, z0)
        c110 = self.at(x1, y1, z0)
        c001 = self.at(x0, y0, z1)
        c101 = self.at(x1, y0, z1)
        c011 = self.at(x0, y1, z1)
        c111 = self.at(x1, y1, z1)
        c00 = c000 * (1.0 - fx) + c100 * fx
        c10 = c010 * (1.0 - fx) + c110 * fx
        c01 = c001 * (1.0 - fx) + c101 * fx
        c11 = c011 * (1.0 - fx) + c111 * fx
        c0 = c00 * (1.0 - fy) + c10 * fy
        c1 = c01 * (1.0 - fy) + c11 * fy
        return c0 * (1.0 - fz) + c1 * fz


@dataclass(frozen=True)
class AutoencoderParameters:
    encoder_gain: float = 1.0
    encoder_bias: float = 0.0
    decoder_gain: float = 1.0
    decoder_bias: float = 0.0

    def vector(self) -> tuple[float, float, float, float]:
        return self.encoder_gain, self.encoder_bias, self.decoder_gain, self.decoder_bias

    @classmethod
    def from_vector(cls, values: Sequence[float]) -> "AutoencoderParameters":
        if len(values) != 4:
            raise ValueError("autoencoder parameter vector must have length 4")
        return cls(*(float(v) for v in values))

    @staticmethod
    def bounds() -> tuple[tuple[float, float], ...]:
        return ((0.1, 3.0), (-1.0, 1.0), (0.1, 3.0), (-1.0, 1.0))


@dataclass(frozen=True)
class LossBreakdown:
    reconstruction: float
    eikonal: float
    bytecode: float
    telemetry: float
    total: float


@dataclass(frozen=True)
class RenderTelemetry:
    rays: int
    hits: int
    mean_steps: float
    max_steps_observed: int


@dataclass(frozen=True)
class ForwardResult:
    vm: VmState
    raw_volume: Volume
    latent: tuple[float, ...]
    reconstructed_volume: Volume
    image: tuple[tuple[Pixel, ...], ...]
    telemetry: RenderTelemetry
    loss: LossBreakdown


@dataclass(frozen=True)
class EngineState:
    program: BytecodeProgram
    autoencoder: AutoencoderParameters = AutoencoderParameters()
    previous_latent: tuple[float, ...] = ()

    def parameter_vector(self) -> tuple[float, ...]:
        return self.autoencoder.vector() + self.program.parameter_vector()

    def parameter_bounds(self) -> tuple[tuple[float, float], ...]:
        return self.autoencoder.bounds() + self.program.parameter_bounds()

    def with_parameter_vector(self, values: Sequence[float]) -> "EngineState":
        values = tuple(float(v) for v in values)
        return replace(
            self,
            autoencoder=AutoencoderParameters.from_vector(values[:4]),
            program=self.program.with_parameter_vector(values[4:]),
        )


@dataclass(frozen=True)
class OptimizationCandidate:
    state: EngineState
    result: ForwardResult
    gradient_norm: float
    dt: float


def _identity() -> Mat4:
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _matmul(a: Mat4, b: Mat4) -> Mat4:
    return tuple(
        tuple(sum(a[r][k] * b[k][c] for k in range(4)) for c in range(4)) for r in range(4)
    )  # type: ignore[return-value]


def _apply(matrix: Mat4, point: Vec3) -> Vec3:
    v = (point[0], point[1], point[2], 1.0)
    out = tuple(sum(matrix[r][c] * v[c] for c in range(4)) for r in range(4))
    if abs(out[3]) < 1.0e-12:
        raise ValueError("invalid homogeneous transform")
    return out[0] / out[3], out[1] / out[3], out[2] / out[3]


def _translation(tx: float, ty: float, tz: float) -> Mat4:
    return (
        (1.0, 0.0, 0.0, tx),
        (0.0, 1.0, 0.0, ty),
        (0.0, 0.0, 1.0, tz),
        (0.0, 0.0, 0.0, 1.0),
    )


def _rotation_y(theta: float) -> Mat4:
    c, s = math.cos(theta), math.sin(theta)
    return (
        (c, 0.0, s, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (-s, 0.0, c, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _scale(scale: float) -> Mat4:
    return (
        (scale, 0.0, 0.0, 0.0),
        (0.0, scale, 0.0, 0.0),
        (0.0, 0.0, scale, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def execute_program(program: BytecodeProgram) -> VmState:
    matrix, inverse = _identity(), _identity()
    uniform_scale = 1.0
    halted = False
    for instruction in program.instructions:
        if instruction.opcode == OP_TRANSLATE:
            tx, ty, tz = instruction.args
            transform, transform_inverse = _translation(tx, ty, tz), _translation(-tx, -ty, -tz)
        elif instruction.opcode == OP_ROTATE_Y:
            (theta,) = instruction.args
            transform, transform_inverse = _rotation_y(theta), _rotation_y(-theta)
        elif instruction.opcode == OP_SCALE:
            (scale,) = instruction.args
            if scale <= 0.0:
                raise ValueError("SCALE must be strictly positive")
            transform, transform_inverse = _scale(scale), _scale(1.0 / scale)
            uniform_scale *= scale
        elif instruction.opcode == OP_HALT:
            halted = True
            break
        else:
            raise ValueError(f"unsupported opcode 0x{instruction.opcode:02X}")
        matrix = _matmul(matrix, transform)
        inverse = _matmul(transform_inverse, inverse)
    if not halted:
        raise ValueError("program did not HALT")
    return VmState(matrix, inverse, uniform_scale)


def rasterize_sdf(vm: VmState, config: RendererConfig) -> Volume:
    size = config.grid_size
    step = config.root_size / size
    first = -config.root_size / 2.0 + step / 2.0
    values: list[float] = []
    for z in range(size):
        pz = first + z * step
        for y in range(size):
            py = first + y * step
            for x in range(size):
                px = first + x * step
                lx, ly, lz = _apply(vm.inverse_matrix, (px, py, pz))
                distance = math.sqrt(lx * lx + ly * ly + lz * lz) - config.sphere_radius
                values.append(distance * vm.uniform_scale)
    return Volume(size, config.root_size, tuple(values))


def autoencode(
    volume: Volume, params: AutoencoderParameters, latent_grid: int
) -> tuple[tuple[float, ...], Volume]:
    size = volume.size
    count = latent_grid**3
    sums, counts = [0.0] * count, [0] * count

    def bucket(x: int, y: int, z: int) -> int:
        bx = min(latent_grid - 1, x * latent_grid // size)
        by = min(latent_grid - 1, y * latent_grid // size)
        bz = min(latent_grid - 1, z * latent_grid // size)
        return (bz * latent_grid + by) * latent_grid + bx

    for z in range(size):
        for y in range(size):
            for x in range(size):
                index = (z * size + y) * size + x
                b = bucket(x, y, z)
                sums[b] += volume.values[index]
                counts[b] += 1
    latent = tuple(
        math.tanh(params.encoder_gain * (sums[i] / counts[i]) + params.encoder_bias)
        for i in range(count)
    )
    reconstructed = tuple(
        params.decoder_gain * latent[bucket(x, y, z)] + params.decoder_bias
        for z in range(size)
        for y in range(size)
        for x in range(size)
    )
    return latent, Volume(size, volume.root_size, reconstructed)


def _add(a: Vec3, b: Vec3) -> Vec3:
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def _mul(a: Vec3, scalar: float) -> Vec3:
    return a[0] * scalar, a[1] * scalar, a[2] * scalar


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _normalize(a: Vec3) -> Vec3:
    length = math.sqrt(max(_dot(a, a), 0.0))
    return (0.0, 0.0, 1.0) if length < 1.0e-12 else _mul(a, 1.0 / length)


def _normal(volume: Volume, point: Vec3) -> Vec3:
    e = volume.root_size / volume.size
    x, y, z = point
    return _normalize(
        (
            volume.sample((x + e, y, z)) - volume.sample((x - e, y, z)),
            volume.sample((x, y + e, z)) - volume.sample((x, y - e, z)),
            volume.sample((x, y, z + e)) - volume.sample((x, y, z - e)),
        )
    )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _cook_torrance(normal: Vec3, view: Vec3, light: Vec3) -> Pixel:
    albedo = (0.62, 0.72, 0.92)
    roughness, metallic = 0.34, 0.08
    n_dot_l, n_dot_v = max(_dot(normal, light), 0.0), max(_dot(normal, view), 1.0e-6)
    if n_dot_l <= 0.0:
        return tuple(0.04 * channel for channel in albedo)  # type: ignore[return-value]
    half_vector = _normalize(_add(light, view))
    n_dot_h, v_dot_h = max(_dot(normal, half_vector), 0.0), max(_dot(view, half_vector), 0.0)
    alpha2 = roughness**4
    denominator = n_dot_h * n_dot_h * (alpha2 - 1.0) + 1.0
    distribution = alpha2 / max(math.pi * denominator * denominator, 1.0e-8)
    k = ((roughness + 1.0) ** 2) / 8.0
    geometry = (n_dot_v / (n_dot_v * (1.0 - k) + k)) * (n_dot_l / (n_dot_l * (1.0 - k) + k))
    f0 = tuple(0.04 * (1.0 - metallic) + channel * metallic for channel in albedo)
    fresnel_factor = (1.0 - v_dot_h) ** 5
    fresnel = tuple(base + (1.0 - base) * fresnel_factor for base in f0)
    specular = tuple(
        distribution * geometry * channel / max(4.0 * n_dot_v * n_dot_l, 1.0e-8)
        for channel in fresnel
    )
    diffuse = tuple((1.0 - f) * (1.0 - metallic) * a / math.pi for f, a in zip(fresnel, albedo))
    ambient = tuple(0.035 * channel for channel in albedo)
    return tuple(
        _clamp01(ambient[i] + (diffuse[i] + specular[i]) * n_dot_l) for i in range(3)
    )  # type: ignore[return-value]


def _ray_direction(i: int, j: int, config: RendererConfig) -> Vec3:
    aspect = config.image_width / config.image_height
    scale = math.tan(math.radians(config.fov_degrees) / 2.0)
    x = (2.0 * ((i + 0.5) / config.image_width) - 1.0) * aspect * scale
    y = (1.0 - 2.0 * ((j + 0.5) / config.image_height)) * scale
    return _normalize((x, y, -1.0))


def render(volume: Volume, config: RendererConfig) -> tuple[tuple[tuple[Pixel, ...], ...], RenderTelemetry]:
    light = _normalize((0.6, 0.8, 0.7))
    rows: list[tuple[Pixel, ...]] = []
    total_steps = max_steps_observed = hits = 0
    for j in range(config.image_height):
        row: list[Pixel] = []
        for i in range(config.image_width):
            direction = _ray_direction(i, j, config)
            t, hit, steps = 0.0, False, 0
            point = config.camera_origin
            for step_index in range(config.max_ray_steps):
                steps = step_index + 1
                point = _add(config.camera_origin, _mul(direction, t))
                sdf = volume.sample(point)
                if abs(sdf) <= config.hit_epsilon:
                    hit = True
                    break
                t += max(abs(sdf) * 0.75, config.min_ray_step)
                if t > config.max_ray_distance:
                    break
            total_steps += steps
            max_steps_observed = max(max_steps_observed, steps)
            if hit:
                hits += 1
                normal = _normal(volume, point)
                row.append(_cook_torrance(normal, _normalize(_mul(direction, -1.0)), light))
            else:
                row.append((0.012, 0.016, 0.024))
        rows.append(tuple(row))
    rays = config.image_width * config.image_height
    return tuple(rows), RenderTelemetry(rays, hits, total_steps / rays, max_steps_observed)


def reconstruction_loss(raw: Volume, reconstructed: Volume) -> float:
    return sum((a - b) ** 2 for a, b in zip(raw.values, reconstructed.values)) / len(raw.values)


def eikonal_loss(volume: Volume) -> float:
    spacing = volume.root_size / volume.size
    error, count = 0.0, 0
    for z in range(1, volume.size - 1):
        for y in range(1, volume.size - 1):
            for x in range(1, volume.size - 1):
                gx = (volume.at(x + 1, y, z) - volume.at(x - 1, y, z)) / (2.0 * spacing)
                gy = (volume.at(x, y + 1, z) - volume.at(x, y - 1, z)) / (2.0 * spacing)
                gz = (volume.at(x, y, z + 1) - volume.at(x, y, z - 1)) / (2.0 * spacing)
                gradient_norm = math.sqrt(gx * gx + gy * gy + gz * gz)
                error += (gradient_norm - 1.0) ** 2
                count += 1
    return error / max(count, 1)


def bytecode_loss(program: BytecodeProgram, latent: Sequence[float], previous_latent: Sequence[float]) -> float:
    parameters = program.parameter_vector()
    transform_penalty = sum(value * value for value in parameters) / max(len(parameters), 1)
    latent_penalty = 0.0
    if previous_latent and len(previous_latent) == len(latent):
        latent_penalty = sum((a - b) ** 2 for a, b in zip(latent, previous_latent)) / len(latent)
    return transform_penalty + latent_penalty


def telemetry_loss(telemetry: RenderTelemetry, config: RendererConfig) -> float:
    excess = max(telemetry.mean_steps - config.ray_target_steps, 0.0)
    return (excess / max(config.ray_target_steps, 1.0)) ** 2


def forward(state: EngineState, config: RendererConfig) -> ForwardResult:
    vm = execute_program(state.program)
    raw = rasterize_sdf(vm, config)
    latent, reconstructed = autoencode(raw, state.autoencoder, config.latent_grid)
    image, telemetry = render(reconstructed, config)
    recon = reconstruction_loss(raw, reconstructed)
    eikonal = eikonal_loss(reconstructed)
    bytecode = bytecode_loss(state.program, latent, state.previous_latent)
    ray_cost = telemetry_loss(telemetry, config)
    total = (
        config.w_reconstruction * recon
        + config.w_eikonal * eikonal
        + config.w_bytecode * bytecode
        + config.w_telemetry * ray_cost
    )
    return ForwardResult(vm, raw, latent, reconstructed, image, telemetry, LossBreakdown(recon, eikonal, bytecode, ray_cost, total))


def _project(values: Sequence[float], bounds: Sequence[tuple[float, float]]) -> tuple[float, ...]:
    return tuple(max(low, min(high, value)) for value, (low, high) in zip(values, bounds))


def finite_difference_gradient(state: EngineState, config: RendererConfig) -> tuple[float, ...]:
    base, bounds = state.parameter_vector(), state.parameter_bounds()
    epsilon = config.finite_difference_epsilon
    gradient: list[float] = []
    for index in range(len(base)):
        plus, minus = list(base), list(base)
        plus[index] += epsilon
        minus[index] -= epsilon
        plus_vector, minus_vector = _project(plus, bounds), _project(minus, bounds)
        plus_loss = forward(state.with_parameter_vector(plus_vector), config).loss.total
        minus_loss = forward(state.with_parameter_vector(minus_vector), config).loss.total
        denominator = max(plus_vector[index] - minus_vector[index], 1.0e-12)
        gradient.append((plus_loss - minus_loss) / denominator)
    norm = math.sqrt(sum(value * value for value in gradient))
    if norm > config.gradient_clip:
        scale = config.gradient_clip / norm
        gradient = [value * scale for value in gradient]
    return tuple(gradient)


class InwardOptimizer:
    """Euler-discretized gradient flow with shadow verify/commit/rollback gating."""

    def __init__(self, config: RendererConfig) -> None:
        self.config = config

    def step(self, authoritative_state: EngineState):
        from .kinetic_runtime import KineticTransactionEngine, ValidatorResult

        config = self.config

        def observe(state: EngineState) -> ForwardResult:
            return forward(state, config)

        def propose(state: EngineState, observation: ForwardResult, encoded: tuple[float, ...]) -> OptimizationCandidate:
            gradient = finite_difference_gradient(state, config)
            norm = math.sqrt(sum(value * value for value in gradient))
            base, bounds = state.parameter_vector(), state.parameter_bounds()
            dt = config.inward_dt
            best_state, best_result, accepted_dt = state, observation, 0.0
            for _ in range(6):
                candidate_vector = _project(tuple(v - dt * g for v, g in zip(base, gradient)), bounds)
                candidate_state = state.with_parameter_vector(candidate_vector)
                candidate_result = forward(candidate_state, config)
                if candidate_result.loss.total <= best_result.loss.total:
                    best_state, best_result, accepted_dt = candidate_state, candidate_result, dt
                    break
                dt *= 0.5
            return OptimizationCandidate(best_state, best_result, norm, accepted_dt)

        def non_regression_validator(state: EngineState, candidate: OptimizationCandidate):
            before, after = forward(state, config).loss.total, candidate.result.loss.total
            return ValidatorResult(
                name="non_regression",
                passed=math.isfinite(after) and after <= before + config.non_regression_tolerance,
                metrics={"before_loss": before, "after_loss": after, "delta": after - before},
                reason="candidate must be finite and non-regressing",
            )

        def finite_validator(state: EngineState, candidate: OptimizationCandidate):
            finite = all(math.isfinite(value) for value in candidate.state.parameter_vector())
            return ValidatorResult(
                name="finite_parameters",
                passed=finite,
                metrics={"gradient_norm": candidate.gradient_norm, "accepted_dt": candidate.dt},
                reason="all candidate parameters must remain finite",
            )

        def commit(state: EngineState, candidate: OptimizationCandidate) -> EngineState:
            return replace(candidate.state, previous_latent=candidate.result.latent)

        engine = KineticTransactionEngine[
            EngineState, ForwardResult, tuple[float, ...], OptimizationCandidate
        ](
            snapshot=lambda state: state,
            observe=observe,
            encode=lambda state, observation: observation.latent,
            propose=propose,
            shadow=lambda state, candidate: {
                "loss": candidate.result.loss.total,
                "mean_ray_steps": candidate.result.telemetry.mean_steps,
                "hits": candidate.result.telemetry.hits,
                "gradient_norm": candidate.gradient_norm,
                "accepted_dt": candidate.dt,
            },
            validators=(non_regression_validator, finite_validator),
            commit=commit,
            rollback=lambda state: state,
            state_identity=lambda state: {
                "program": state.program.to_bytes().hex(),
                "autoencoder": state.autoencoder.vector(),
                "previous_latent": state.previous_latent,
            },
            candidate_identity=lambda candidate: {
                "program": candidate.state.program.to_bytes().hex(),
                "autoencoder": candidate.state.autoencoder.vector(),
                "loss": candidate.result.loss.total,
            },
        )
        return engine.step(authoritative_state)


def demo_program() -> BytecodeProgram:
    return BytecodeProgram(
        (
            Instruction(OP_TRANSLATE, (0.30, -0.08, 0.0)),
            Instruction(OP_ROTATE_Y, (0.35,)),
            Instruction(OP_SCALE, (1.05,)),
            Instruction(OP_HALT),
        )
    )


def write_ppm(image: Sequence[Sequence[Pixel]], path: str | Path) -> None:
    path = Path(path)
    height, width = len(image), len(image[0]) if image else 0
    with path.open("w", encoding="ascii") as handle:
        handle.write(f"P3\n{width} {height}\n255\n")
        for row in image:
            for pixel in row:
                rgb = tuple(round(_clamp01(channel) * 255.0) for channel in pixel)
                handle.write(f"{rgb[0]} {rgb[1]} {rgb[2]}\n")


def _metrics(result: ForwardResult) -> dict[str, object]:
    return {
        "loss": {
            "reconstruction": result.loss.reconstruction,
            "eikonal": result.loss.eikonal,
            "bytecode": result.loss.bytecode,
            "telemetry": result.loss.telemetry,
            "total": result.loss.total,
        },
        "telemetry": {
            "rays": result.telemetry.rays,
            "hits": result.telemetry.hits,
            "mean_steps": result.telemetry.mean_steps,
            "max_steps_observed": result.telemetry.max_steps_observed,
        },
        "latent_dim": len(result.latent),
        "uniform_scale": result.vm.uniform_scale,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=24)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--height", type=int, default=24)
    parser.add_argument("--optimize-steps", type=int, default=0)
    parser.add_argument("--output", default="inward_frame.ppm")
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = RendererConfig(grid_size=args.grid_size, image_width=args.width, image_height=args.height)
    state = EngineState(program=demo_program())
    for _ in range(args.optimize_steps):
        state = InwardOptimizer(config).step(state).state
    result = forward(state, config)
    write_ppm(result.image, args.output)
    print(json.dumps(_metrics(result), indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
