"""Bounded radial permeation operator for the Dr Moagi research layer.

The implementation treats permeation as a static boundary-value problem, not as
instantaneous physical propagation.  A locked spherical core of radius ``a``
and boundary value ``phi0`` induces the unique spherically symmetric harmonic
exterior

    Phi(r) = phi0 * a / r,  r > a,

with Phi(a)=phi0 and Phi(r)->0 as r->infinity.

For non-zero wavenumber ``k`` the optional harmonic excitation uses the outgoing
radial Helmholtz continuation

    Phi_k(r) = phi0 * a/r * exp(i*k*(r-a)),  r > a.

Only a finite domain is ever sampled by this module.  Infinite support is a
mathematical property of the analytic field, not a claim of infinite
allocation, zero latency, or non-causal execution.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

Point3D = tuple[float, float, float]


@dataclass(frozen=True)
class PermeationConfig:
    """Numerical contract for a locked spherical core and finite sampler."""

    core_radius: float = 1.0
    core_value: float = 1.0
    max_radius: float = 64.0
    samples: int = 257

    def __post_init__(self) -> None:
        for name in ("core_radius", "core_value", "max_radius"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")

        if self.core_radius <= 0.0:
            raise ValueError("core_radius must be positive")
        if self.core_value == 0.0:
            raise ValueError("core_value must be non-zero")
        if self.max_radius < self.core_radius:
            raise ValueError("max_radius must be at least core_radius")
        if isinstance(self.samples, bool) or not isinstance(self.samples, int):
            raise TypeError("samples must be an integer")
        if self.samples < 2:
            raise ValueError("samples must be at least 2")


@dataclass(frozen=True)
class PermeationSample:
    radius: float
    potential: float
    radial_derivative: float | None


@dataclass(frozen=True)
class PermeationMetrics:
    core_radius: float
    max_radius: float
    core_value: float
    outer_value: float
    outer_to_core_ratio: float
    sampled_points: int


class PermeationField:
    """Analytic locked-core field with bounded sampling and relaxation helpers."""

    def __init__(self, config: PermeationConfig | None = None) -> None:
        self.config = config or PermeationConfig()

    def potential_at_radius(self, radius: float) -> float:
        """Return the static locked-core potential.

        The core is held exactly at ``core_value``.  Outside the core the field
        is the harmonic 1/r continuation.
        """

        r = self._radius(radius)
        a = self.config.core_radius
        if r <= a:
            return float(self.config.core_value)
        return float(self.config.core_value * a / r)

    def potential(self, point: Point3D) -> float:
        return self.potential_at_radius(self._point_radius(point))

    def helmholtz_at_radius(self, radius: float, wavenumber: float) -> complex:
        """Return the outgoing spherical Helmholtz continuation.

        ``k=0`` reduces exactly to the static field.  This is a frequency-domain
        solution; it does not model time-of-flight or causal propagation.
        """

        r = self._radius(radius)
        k = self._finite(wavenumber, "wavenumber")
        if k < 0.0:
            raise ValueError("wavenumber must be non-negative")

        a = self.config.core_radius
        phi0 = self.config.core_value
        if r <= a:
            return complex(phi0, 0.0)
        return complex(phi0 * a / r, 0.0) * cmath.exp(1j * k * (r - a))

    def exterior_radial_derivative(self, radius: float) -> float:
        """Return dPhi/dr for ``r > core_radius``."""

        r = self._radius(radius)
        a = self.config.core_radius
        if r <= a:
            raise ValueError("exterior derivative is defined only for r > core_radius")
        return float(-self.config.core_value * a / (r * r))

    def gradient(self, point: Point3D) -> Point3D:
        """Return the classical gradient away from the spherical boundary.

        Inside the locked core the classical gradient is zero.  Exactly on the
        boundary the derivative jumps and the distributional shell source must
        be used instead, so this method rejects that point.
        """

        x, y, z = self._point(point)
        r = math.sqrt(x * x + y * y + z * z)
        a = self.config.core_radius
        if math.isclose(r, a, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("gradient is discontinuous at the shell boundary")
        if r < a or r == 0.0:
            return (0.0, 0.0, 0.0)

        radial = self.exterior_radial_derivative(r)
        scale = radial / r
        return (scale * x, scale * y, scale * z)

    def radial_laplacian(self, radius: float) -> float:
        """Return the classical radial Laplacian away from the shell.

        The 1/r exterior and constant interior are harmonic.  At the shell the
        source is distributional rather than a finite classical value.
        """

        r = self._radius(radius)
        a = self.config.core_radius
        if math.isclose(r, a, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("Laplacian contains a distributional shell source at r=a")
        return 0.0

    def threshold_radius(self, epsilon: float) -> float:
        """Radius beyond which ``|Phi(r)| <= epsilon``.

        This converts infinite mathematical support into a practical finite
        truncation radius for a requested amplitude threshold.
        """

        eps = self._finite(epsilon, "epsilon")
        if eps <= 0.0:
            raise ValueError("epsilon must be positive")

        a = self.config.core_radius
        magnitude = abs(self.config.core_value)
        if eps >= magnitude:
            return a
        return float(a * magnitude / eps)

    def relax_value(self, radius: float, current: float, gain: float = 1.0) -> float:
        """Move one scalar sample toward the analytic target.

        ``gain=1`` is an exact projection.  ``0 < gain < 1`` is bounded
        relaxation with geometric error reduction, which operationalizes the
        self-healing metaphor without claiming an instantaneous counter-wave.
        """

        value = self._finite(current, "current")
        g = self._finite(gain, "gain")
        if g <= 0.0 or g > 1.0:
            raise ValueError("gain must satisfy 0 < gain <= 1")
        target = self.potential_at_radius(radius)
        return float(value + g * (target - value))

    def sample_profile(self) -> tuple[PermeationSample, ...]:
        """Sample the finite configured domain, including the origin and outer edge."""

        count = self.config.samples
        stop = self.config.max_radius
        step = stop / (count - 1)
        result: list[PermeationSample] = []
        for index in range(count):
            r = step * index
            derivative: float | None
            if r > self.config.core_radius:
                derivative = self.exterior_radial_derivative(r)
            elif math.isclose(
                r, self.config.core_radius, rel_tol=0.0, abs_tol=max(1e-15, step * 1e-12)
            ):
                derivative = None
            else:
                derivative = 0.0
            result.append(
                PermeationSample(
                    radius=r,
                    potential=self.potential_at_radius(r),
                    radial_derivative=derivative,
                )
            )
        return tuple(result)

    def metrics(self) -> PermeationMetrics:
        outer = self.potential_at_radius(self.config.max_radius)
        return PermeationMetrics(
            core_radius=self.config.core_radius,
            max_radius=self.config.max_radius,
            core_value=self.config.core_value,
            outer_value=outer,
            outer_to_core_ratio=abs(outer / self.config.core_value),
            sampled_points=self.config.samples,
        )

    @staticmethod
    def normalized_shell_charge(core_radius: float, core_value: float) -> float:
        """Return Q for ``-Laplacian Phi = Q delta(r-a)/(4*pi*a^2)``.

        With this normalization the static shell solution is

            Phi(r) = Q / (4*pi*max(r, a)),

        and choosing Q = 4*pi*a*phi0 reproduces the locked-boundary field.
        """

        a = PermeationField._finite(core_radius, "core_radius")
        phi0 = PermeationField._finite(core_value, "core_value")
        if a <= 0.0:
            raise ValueError("core_radius must be positive")
        return float(4.0 * math.pi * a * phi0)

    @staticmethod
    def _finite(value: float, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite")
        return result

    @classmethod
    def _radius(cls, radius: float) -> float:
        r = cls._finite(radius, "radius")
        if r < 0.0:
            raise ValueError("radius must be non-negative")
        return r

    @classmethod
    def _point(cls, point: Point3D) -> Point3D:
        if len(point) != 3:
            raise ValueError("point must contain exactly three coordinates")
        x = cls._finite(point[0], "point coordinate")
        y = cls._finite(point[1], "point coordinate")
        z = cls._finite(point[2], "point coordinate")
        return (x, y, z)

    @classmethod
    def _point_radius(cls, point: Point3D) -> float:
        x, y, z = cls._point(point)
        return math.sqrt(x * x + y * y + z * z)
