"""Deterministic electromagnetic-photonic rendering reference.

This module treats a rendered pixel as a finite-area spectral detector. It is a
bounded, dependency-free correctness reference: light transport is represented
with geometric optics, inverse-square attenuation, wavelength-dependent sensor
response, Lambertian reflection and a bounded specular term. It is not a
full-wave Maxwell solver, a production path tracer or a calibrated camera model.

The authoritative transition is transactional::

    observe -> partition -> transport -> integrate -> quantize
            -> verify -> commit or rollback -> journal

Tile scheduling is deliberately separated from pixel semantics so a future CPU,
GPU or distributed backend can replace the execution strategy without changing
frame digests, bounds, quantization or transaction behavior.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence

Vec3 = tuple[float, float, float]
RGB = tuple[float, float, float]
RGB8 = tuple[int, int, int]

_EPSILON = 1.0e-12
_VISIBLE_MIN_NM = 380.0
_VISIBLE_MAX_NM = 700.0


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _vec3(value: Sequence[float], name: str) -> Vec3:
    if len(value) != 3:
        raise ValueError(f"{name} must have exactly three components")
    return tuple(
        _finite(component, f"{name}[{index}]") for index, component in enumerate(value)
    )  # type: ignore[return-value]


def _rgb(value: Sequence[float], name: str) -> RGB:
    result = _vec3(value, name)
    if any(component < 0.0 or component > 1.0 for component in result):
        raise ValueError(f"{name} components must be in [0, 1]")
    return result


def _add(a: Vec3, b: Vec3) -> Vec3:
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def _mul(a: Vec3, scalar: float) -> Vec3:
    return a[0] * scalar, a[1] * scalar, a[2] * scalar


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _length(value: Vec3) -> float:
    return math.sqrt(_dot(value, value))


def _normalise(value: Vec3, name: str = "vector") -> Vec3:
    magnitude = _length(value)
    if magnitude <= _EPSILON:
        raise ValueError(f"{name} must have non-zero length")
    return _mul(value, 1.0 / magnitude)


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _gaussian(value: float, mean: float, sigma: float) -> float:
    return math.exp(-0.5 * ((value - mean) / sigma) ** 2)


def wavelength_to_linear_rgb(wavelength_nm: float) -> RGB:
    """Approximate a visible wavelength with a bounded linear RGB triplet."""

    wavelength = _finite(wavelength_nm, "wavelength_nm")
    if not _VISIBLE_MIN_NM <= wavelength <= _VISIBLE_MAX_NM:
        raise ValueError("wavelength_nm must be in [380, 700]")

    red = _gaussian(wavelength, 610.0, 48.0)
    green = _gaussian(wavelength, 545.0, 42.0)
    blue = _gaussian(wavelength, 455.0, 36.0)
    peak = max(red, green, blue, _EPSILON)
    return red / peak, green / peak, blue / peak


def sensor_response(wavelength_nm: float) -> RGB:
    """Return a simple three-channel photodetector response."""

    wavelength = _finite(wavelength_nm, "wavelength_nm")
    if not _VISIBLE_MIN_NM <= wavelength <= _VISIBLE_MAX_NM:
        raise ValueError("wavelength_nm must be in [380, 700]")
    return (
        _gaussian(wavelength, 610.0, 52.0),
        _gaussian(wavelength, 545.0, 46.0),
        _gaussian(wavelength, 455.0, 40.0),
    )


@dataclass(frozen=True, order=True)
class SpectralSample:
    wavelength_nm: float
    power: float

    def __post_init__(self) -> None:
        wavelength = _finite(self.wavelength_nm, "wavelength_nm")
        power = _finite(self.power, "power")
        if not _VISIBLE_MIN_NM <= wavelength <= _VISIBLE_MAX_NM:
            raise ValueError("wavelength_nm must be in [380, 700]")
        if power < 0.0:
            raise ValueError("power must be non-negative")
        object.__setattr__(self, "wavelength_nm", wavelength)
        object.__setattr__(self, "power", power)


@dataclass(frozen=True)
class Spectrum:
    samples: tuple[SpectralSample, ...]

    def __post_init__(self) -> None:
        if not self.samples:
            raise ValueError("spectrum must contain at least one sample")
        wavelengths = [sample.wavelength_nm for sample in self.samples]
        if wavelengths != sorted(wavelengths):
            raise ValueError("spectrum samples must be ordered by wavelength")
        if len(set(wavelengths)) != len(wavelengths):
            raise ValueError("spectrum wavelengths must be unique")

    @classmethod
    def monochromatic(cls, wavelength_nm: float, power: float = 1.0) -> "Spectrum":
        return cls((SpectralSample(wavelength_nm, power),))

    @classmethod
    def white_reference(cls, power: float = 1.0) -> "Spectrum":
        value = _finite(power, "power")
        if value < 0.0:
            raise ValueError("power must be non-negative")
        wavelengths = (420.0, 460.0, 500.0, 540.0, 580.0, 620.0, 660.0)
        per_sample = value / len(wavelengths)
        return cls(
            tuple(SpectralSample(wavelength, per_sample) for wavelength in wavelengths)
        )

    @property
    def total_power(self) -> float:
        return sum(sample.power for sample in self.samples)


@dataclass(frozen=True)
class Material:
    reflectance: RGB = (0.7, 0.7, 0.7)
    roughness: float = 0.5
    emission: Spectrum | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "reflectance", _rgb(self.reflectance, "reflectance"))
        roughness = _finite(self.roughness, "roughness")
        if not 0.0 <= roughness <= 1.0:
            raise ValueError("roughness must be in [0, 1]")
        object.__setattr__(self, "roughness", roughness)


@dataclass(frozen=True)
class Sphere:
    center: Vec3
    radius: float
    material: Material = Material()

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _vec3(self.center, "center"))
        radius = _finite(self.radius, "radius")
        if radius <= 0.0:
            raise ValueError("radius must be positive")
        object.__setattr__(self, "radius", radius)


@dataclass(frozen=True)
class PointEmitter:
    position: Vec3
    spectrum: Spectrum
    intensity: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _vec3(self.position, "position"))
        intensity = _finite(self.intensity, "intensity")
        if intensity < 0.0:
            raise ValueError("intensity must be non-negative")
        object.__setattr__(self, "intensity", intensity)


@dataclass(frozen=True)
class PhotonicScene:
    spheres: tuple[Sphere, ...]
    emitters: tuple[PointEmitter, ...]
    background: RGB = (0.01, 0.02, 0.04)

    def __post_init__(self) -> None:
        object.__setattr__(self, "background", _rgb(self.background, "background"))
        if not self.emitters:
            raise ValueError("scene must contain at least one emitter")


@dataclass(frozen=True)
class Camera:
    width: int
    height: int
    origin: Vec3 = (0.0, 0.0, 3.5)
    forward: Vec3 = (0.0, 0.0, -1.0)
    up: Vec3 = (0.0, 1.0, 0.0)
    vertical_fov_degrees: float = 52.0

    def __post_init__(self) -> None:
        for name in ("width", "height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        object.__setattr__(self, "origin", _vec3(self.origin, "origin"))
        forward = _normalise(_vec3(self.forward, "forward"), "forward")
        up = _normalise(_vec3(self.up, "up"), "up")
        if abs(_dot(forward, up)) >= 1.0 - 1.0e-9:
            raise ValueError("forward and up must not be parallel")
        object.__setattr__(self, "forward", forward)
        object.__setattr__(self, "up", up)
        fov = _finite(self.vertical_fov_degrees, "vertical_fov_degrees")
        if not 1.0 <= fov < 179.0:
            raise ValueError("vertical_fov_degrees must be in [1, 179)")
        object.__setattr__(self, "vertical_fov_degrees", fov)


@dataclass(frozen=True)
class RenderConfig:
    exposure: float = 1.0
    gamma: float = 2.2
    samples_per_axis: int = 1
    tile_edge: int = 16
    max_pixels: int = 1_000_000
    max_optical_path: float = 1_000_000.0

    def __post_init__(self) -> None:
        exposure = _finite(self.exposure, "exposure")
        gamma = _finite(self.gamma, "gamma")
        max_path = _finite(self.max_optical_path, "max_optical_path")
        if exposure < 0.0:
            raise ValueError("exposure must be non-negative")
        if gamma <= 0.0:
            raise ValueError("gamma must be positive")
        if max_path <= 0.0:
            raise ValueError("max_optical_path must be positive")
        for name in ("samples_per_axis", "tile_edge", "max_pixels"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.samples_per_axis > 8:
            raise ValueError("samples_per_axis must not exceed 8 in the reference renderer")
        object.__setattr__(self, "exposure", exposure)
        object.__setattr__(self, "gamma", gamma)
        object.__setattr__(self, "max_optical_path", max_path)


@dataclass(frozen=True)
class PixelMeasurement:
    linear_rgb: RGB
    quantized_rgb: RGB8
    irradiance: float
    optical_path_length: float
    depth: float | None


@dataclass(frozen=True)
class WorkTile:
    tile_id: int
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def pixel_count(self) -> int:
        return (self.x1 - self.x0) * (self.y1 - self.y0)


@dataclass(frozen=True)
class PhotonicFrame:
    width: int
    height: int
    pixels: tuple[PixelMeasurement, ...]
    digest: str

    def at(self, x: int, y: int) -> PixelMeasurement:
        if not 0 <= x < self.width or not 0 <= y < self.height:
            raise IndexError("pixel coordinate out of range")
        return self.pixels[x + self.width * y]


class RuntimeStage(str, Enum):
    OBSERVE = "observe"
    PARTITION = "partition"
    TRANSPORT = "transport"
    INTEGRATE = "integrate"
    QUANTIZE = "quantize"
    VERIFY = "verify"
    COMMIT = "commit"
    ROLLBACK = "rollback"


@dataclass(frozen=True)
class PhotonicRuntimeState:
    cycle: int = 0
    frame_digest: str = "0" * 64
    omega_digest: str = "0" * 64


@dataclass(frozen=True)
class RuntimeReceipt:
    cycle: int
    committed: bool
    stage: RuntimeStage
    frame_digest: str
    previous_state_digest: str
    state_digest: str
    reason: str | None
    tile_count: int
    pixel_count: int


@dataclass(frozen=True)
class _Hit:
    distance: float
    point: Vec3
    normal: Vec3
    material: Material


def partition_tiles(width: int, height: int, tile_edge: int) -> tuple[WorkTile, ...]:
    """Partition a frame deterministically in row-major tile order."""

    for name, value in (("width", width), ("height", height), ("tile_edge", tile_edge)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    tiles: list[WorkTile] = []
    tile_id = 0
    for y0 in range(0, height, tile_edge):
        for x0 in range(0, width, tile_edge):
            tiles.append(
                WorkTile(
                    tile_id=tile_id,
                    x0=x0,
                    y0=y0,
                    x1=min(width, x0 + tile_edge),
                    y1=min(height, y0 + tile_edge),
                )
            )
            tile_id += 1
    return tuple(tiles)


def pixel_to_lattice_coordinate(
    x: int,
    y: int,
    depth: float | None,
    width: int,
    height: int,
    side: int = 1000,
) -> tuple[int, int, int]:
    """Encode a 2D detector sample and normalised depth into a bounded 3D lattice."""

    fields = (("x", x), ("y", y), ("width", width), ("height", height), ("side", side))
    for name, value in fields:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if width <= 0 or height <= 0 or side <= 0:
        raise ValueError("width, height and side must be positive")
    if not 0 <= x < width or not 0 <= y < height:
        raise ValueError("pixel coordinate lies outside the frame")
    if depth is None:
        normalised_depth = 1.0
    else:
        depth_value = _finite(depth, "depth")
        if depth_value < 0.0:
            raise ValueError("depth must be non-negative")
        normalised_depth = depth_value / (1.0 + depth_value)
    lattice_x = min(side - 1, int((x + 0.5) * side / width))
    lattice_y = min(side - 1, int((y + 0.5) * side / height))
    lattice_z = min(side - 1, int(normalised_depth * (side - 1)))
    return lattice_x, lattice_y, lattice_z


class PhotonicRenderer:
    """Bounded deterministic renderer with transactional runtime state."""

    def __init__(self, state: PhotonicRuntimeState | None = None) -> None:
        self._state = state or PhotonicRuntimeState()

    @property
    def state(self) -> PhotonicRuntimeState:
        return self._state

    @staticmethod
    def _camera_basis(camera: Camera) -> tuple[Vec3, Vec3, Vec3]:
        forward = camera.forward
        right = _normalise(_cross(forward, camera.up), "camera right")
        up = _normalise(_cross(right, forward), "camera up")
        return right, up, forward

    @staticmethod
    def _ray_sphere(origin: Vec3, direction: Vec3, sphere: Sphere) -> _Hit | None:
        offset = _sub(origin, sphere.center)
        half_b = _dot(offset, direction)
        c = _dot(offset, offset) - sphere.radius * sphere.radius
        discriminant = half_b * half_b - c
        if discriminant < 0.0:
            return None
        root = math.sqrt(discriminant)
        distance = -half_b - root
        if distance <= 1.0e-9:
            distance = -half_b + root
        if distance <= 1.0e-9:
            return None
        point = _add(origin, _mul(direction, distance))
        normal = _normalise(_sub(point, sphere.center), "surface normal")
        return _Hit(distance, point, normal, sphere.material)

    def _nearest_hit(self, scene: PhotonicScene, origin: Vec3, direction: Vec3) -> _Hit | None:
        nearest: _Hit | None = None
        for sphere in scene.spheres:
            hit = self._ray_sphere(origin, direction, sphere)
            if hit is not None and (nearest is None or hit.distance < nearest.distance):
                nearest = hit
        return nearest

    def _visible(
        self,
        scene: PhotonicScene,
        point: Vec3,
        normal: Vec3,
        emitter: PointEmitter,
    ) -> bool:
        origin = _add(point, _mul(normal, 1.0e-7))
        to_emitter = _sub(emitter.position, origin)
        distance = _length(to_emitter)
        direction = _normalise(to_emitter, "shadow direction")
        hit = self._nearest_hit(scene, origin, direction)
        return hit is None or hit.distance >= distance - 1.0e-7

    @staticmethod
    def _emission_rgb(spectrum: Spectrum) -> RGB:
        result = [0.0, 0.0, 0.0]
        for sample in spectrum.samples:
            detector = sensor_response(sample.wavelength_nm)
            colour = wavelength_to_linear_rgb(sample.wavelength_nm)
            for channel in range(3):
                result[channel] += sample.power * detector[channel] * colour[channel]
        return result[0], result[1], result[2]

    def _shade(
        self,
        scene: PhotonicScene,
        hit: _Hit,
        view_direction: Vec3,
    ) -> tuple[RGB, float, float]:
        accumulated = [0.0, 0.0, 0.0]
        irradiance = 0.0
        optical_path = 0.0
        visible_emitters = 0

        if hit.material.emission is not None:
            emitted = self._emission_rgb(hit.material.emission)
            for channel in range(3):
                accumulated[channel] += emitted[channel]

        for emitter in scene.emitters:
            if not self._visible(scene, hit.point, hit.normal, emitter):
                continue
            to_emitter = _sub(emitter.position, hit.point)
            distance = _length(to_emitter)
            if distance <= _EPSILON:
                continue
            light_direction = _mul(to_emitter, 1.0 / distance)
            cosine = max(0.0, _dot(hit.normal, light_direction))
            if cosine <= 0.0:
                continue
            attenuation = emitter.intensity / (4.0 * math.pi * distance * distance)
            half_vector = _normalise(
                _sub(light_direction, view_direction),
                "half vector",
            )
            shininess = 2.0 + 254.0 * (1.0 - hit.material.roughness) ** 2
            specular = max(0.0, _dot(hit.normal, half_vector)) ** shininess
            specular_weight = (1.0 - hit.material.roughness) * 0.35

            for sample in emitter.spectrum.samples:
                incident = sample.power * attenuation * cosine
                detector = sensor_response(sample.wavelength_nm)
                colour = wavelength_to_linear_rgb(sample.wavelength_nm)
                for channel in range(3):
                    diffuse = hit.material.reflectance[channel] * incident / math.pi
                    glossy = specular_weight * sample.power * attenuation * specular
                    accumulated[channel] += detector[channel] * colour[channel] * (
                        diffuse + glossy
                    )
                irradiance += incident
            optical_path += distance
            visible_emitters += 1

        mean_path = optical_path / visible_emitters if visible_emitters else 0.0
        return (accumulated[0], accumulated[1], accumulated[2]), irradiance, mean_path

    @staticmethod
    def _tone_map(linear: RGB, config: RenderConfig) -> tuple[RGB, RGB8]:
        exposed = tuple(
            1.0 - math.exp(-max(0.0, channel) * config.exposure) for channel in linear
        )
        encoded = tuple(_clamp01(channel) ** (1.0 / config.gamma) for channel in exposed)
        quantized = tuple(
            min(255, max(0, int(round(channel * 255.0)))) for channel in encoded
        )
        return encoded, quantized  # type: ignore[return-value]

    def _sample_pixel(
        self,
        scene: PhotonicScene,
        camera: Camera,
        config: RenderConfig,
        x: int,
        y: int,
    ) -> PixelMeasurement:
        right, up, forward = self._camera_basis(camera)
        aspect = camera.width / camera.height
        half_height = math.tan(math.radians(camera.vertical_fov_degrees) / 2.0)
        samples = config.samples_per_axis
        linear_sum = [0.0, 0.0, 0.0]
        irradiance_sum = 0.0
        path_sum = 0.0
        depth_sum = 0.0
        hit_count = 0

        for sample_y_index in range(samples):
            for sample_x_index in range(samples):
                sample_x = x + (sample_x_index + 0.5) / samples
                sample_y = y + (sample_y_index + 0.5) / samples
                ndc_x = (2.0 * sample_x / camera.width - 1.0) * aspect * half_height
                ndc_y = (1.0 - 2.0 * sample_y / camera.height) * half_height
                direction = _normalise(
                    _add(forward, _add(_mul(right, ndc_x), _mul(up, ndc_y))),
                    "camera ray",
                )
                hit = self._nearest_hit(scene, camera.origin, direction)
                if hit is None:
                    for channel in range(3):
                        linear_sum[channel] += scene.background[channel]
                    continue
                shaded, irradiance, light_path = self._shade(scene, hit, direction)
                total_path = hit.distance + light_path
                if total_path > config.max_optical_path:
                    raise ValueError("optical path exceeds configured bound")
                for channel in range(3):
                    linear_sum[channel] += shaded[channel]
                irradiance_sum += irradiance
                path_sum += total_path
                depth_sum += hit.distance
                hit_count += 1

        sample_count = samples * samples
        linear = tuple(channel / sample_count for channel in linear_sum)
        encoded, quantized = self._tone_map(linear, config)  # type: ignore[arg-type]
        depth = depth_sum / hit_count if hit_count else None
        return PixelMeasurement(
            linear_rgb=encoded,
            quantized_rgb=quantized,
            irradiance=irradiance_sum / sample_count,
            optical_path_length=path_sum / hit_count if hit_count else 0.0,
            depth=depth,
        )

    @staticmethod
    def _frame_digest(
        width: int,
        height: int,
        pixels: Sequence[PixelMeasurement],
    ) -> str:
        payload = {
            "height": height,
            "pixels": [
                {
                    "depth": None if pixel.depth is None else format(pixel.depth, ".17g"),
                    "irradiance": format(pixel.irradiance, ".17g"),
                    "optical_path_length": format(pixel.optical_path_length, ".17g"),
                    "rgb": list(pixel.quantized_rgb),
                }
                for pixel in pixels
            ],
            "width": width,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _state_digest(state: PhotonicRuntimeState) -> str:
        payload = json.dumps(
            {
                "cycle": state.cycle,
                "frame_digest": state.frame_digest,
                "omega_digest": state.omega_digest,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _verify_frame(frame: PhotonicFrame, camera: Camera) -> None:
        if frame.width != camera.width or frame.height != camera.height:
            raise ValueError("frame dimensions do not match camera")
        if len(frame.pixels) != camera.width * camera.height:
            raise ValueError("frame pixel count is invalid")
        if len(frame.digest) != 64:
            raise ValueError("frame digest is invalid")
        for pixel in frame.pixels:
            if any(not 0 <= channel <= 255 for channel in pixel.quantized_rgb):
                raise ValueError("quantized channel lies outside [0, 255]")
            if pixel.irradiance < 0.0 or not math.isfinite(pixel.irradiance):
                raise ValueError("pixel irradiance is invalid")

    def render(
        self,
        scene: PhotonicScene,
        camera: Camera,
        config: RenderConfig,
    ) -> PhotonicFrame:
        """Render one deterministic frame without mutating runtime state."""

        pixel_count = camera.width * camera.height
        if pixel_count > config.max_pixels:
            raise ValueError("frame exceeds max_pixels")
        pixels: list[PixelMeasurement | None] = [None] * pixel_count
        for tile in partition_tiles(camera.width, camera.height, config.tile_edge):
            for y in range(tile.y0, tile.y1):
                for x in range(tile.x0, tile.x1):
                    pixels[x + camera.width * y] = self._sample_pixel(
                        scene,
                        camera,
                        config,
                        x,
                        y,
                    )
        completed = tuple(pixel for pixel in pixels if pixel is not None)
        if len(completed) != pixel_count:
            raise RuntimeError("renderer left incomplete pixels")
        digest = self._frame_digest(camera.width, camera.height, completed)
        return PhotonicFrame(camera.width, camera.height, completed, digest)

    def cycle(
        self,
        scene: PhotonicScene,
        camera: Camera,
        config: RenderConfig,
    ) -> tuple[PhotonicFrame | None, PhotonicRuntimeState, RuntimeReceipt]:
        """Render, verify and commit one complete transactional frame cycle."""

        previous_state = self._state
        previous_digest = self._state_digest(previous_state)
        tile_count = len(partition_tiles(camera.width, camera.height, config.tile_edge))
        pixel_count = camera.width * camera.height
        try:
            frame = self.render(scene, camera, config)
            self._verify_frame(frame, camera)
            omega_payload = (
                f"{previous_state.omega_digest}:{frame.digest}:{previous_state.cycle + 1}"
            )
            next_state = PhotonicRuntimeState(
                cycle=previous_state.cycle + 1,
                frame_digest=frame.digest,
                omega_digest=hashlib.sha256(omega_payload.encode("utf-8")).hexdigest(),
            )
            self._state = next_state
            state_digest = self._state_digest(next_state)
            receipt = RuntimeReceipt(
                cycle=next_state.cycle,
                committed=True,
                stage=RuntimeStage.COMMIT,
                frame_digest=frame.digest,
                previous_state_digest=previous_digest,
                state_digest=state_digest,
                reason=None,
                tile_count=tile_count,
                pixel_count=pixel_count,
            )
            return frame, next_state, receipt
        except (TypeError, ValueError, RuntimeError) as error:
            self._state = previous_state
            receipt = RuntimeReceipt(
                cycle=previous_state.cycle,
                committed=False,
                stage=RuntimeStage.ROLLBACK,
                frame_digest=previous_state.frame_digest,
                previous_state_digest=previous_digest,
                state_digest=previous_digest,
                reason=str(error),
                tile_count=tile_count,
                pixel_count=pixel_count,
            )
            return None, previous_state, receipt


def frame_rgb_bytes(frame: PhotonicFrame) -> bytes:
    """Serialize quantized RGB pixels in row-major order."""

    output = bytearray()
    for pixel in frame.pixels:
        output.extend(pixel.quantized_rgb)
    return bytes(output)


def mean_irradiance(frame: PhotonicFrame) -> float:
    if not frame.pixels:
        return 0.0
    return sum(pixel.irradiance for pixel in frame.pixels) / len(frame.pixels)


def iter_lattice_samples(
    frame: PhotonicFrame,
    side: int = 1000,
) -> Iterable[tuple[tuple[int, int, int], PixelMeasurement]]:
    """Yield deterministic 3D lattice coordinates for every rendered pixel."""

    for y in range(frame.height):
        for x in range(frame.width):
            pixel = frame.at(x, y)
            coordinate = pixel_to_lattice_coordinate(
                x,
                y,
                pixel.depth,
                frame.width,
                frame.height,
                side,
            )
            yield coordinate, pixel
