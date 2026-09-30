"""Coupled 3D geometry -> 2D projection -> latent fixed-point reference kernel.

The authoritative geometry is a hard signed-distance field (SDF). Rendering uses a
separate smooth occupancy surrogate so image-space derivatives remain defined near
the zero level set. This keeps the geometry/material contract distinct from the
optimization surrogate.

The module is dependency-free and intentionally small enough to audit. It is a
reference implementation, not a production renderer or Transformer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

Vec = tuple[float, ...]
Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class WireSpec:
    total_length_mm: float = 1000.0
    diameter_mm: float = 1.0
    inner_gap_mm: float = 2.0
    copper_density: float = 1.0
    smooth_width_mm: float = 0.05

    def __post_init__(self) -> None:
        values = (
            self.total_length_mm,
            self.diameter_mm,
            self.inner_gap_mm,
            self.copper_density,
            self.smooth_width_mm,
        )
        if not all(math.isfinite(v) and v > 0.0 for v in values):
            raise ValueError("all wire parameters must be finite and positive")
        if self.total_length_mm <= math.pi * self.bend_radius_mm:
            raise ValueError("total_length_mm must exceed the U-bend arc length")

    @property
    def radius_mm(self) -> float:
        return 0.5 * self.diameter_mm

    @property
    def centerline_separation_mm(self) -> float:
        return self.inner_gap_mm + self.diameter_mm

    @property
    def bend_radius_mm(self) -> float:
        return 0.5 * self.centerline_separation_mm

    @property
    def bend_curvature_per_mm(self) -> float:
        return 1.0 / self.bend_radius_mm

    @property
    def leg_length_mm(self) -> float:
        return 0.5 * (self.total_length_mm - math.pi * self.bend_radius_mm)


def _clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else hi if value > hi else value


def centerline_point(s_mm: float, spec: WireSpec = WireSpec()) -> Vec3:
    """Return the planar hairpin centerline point at arc length s."""

    s = _clamp(float(s_mm), 0.0, spec.total_length_mm)
    r = spec.bend_radius_mm
    leg = spec.leg_length_mm
    arc = math.pi * r

    if s <= leg:
        return (-r, s, 0.0)
    if s <= leg + arc:
        local = s - leg
        phi = math.pi - local / r
        return (r * math.cos(phi), leg + r * math.sin(phi), 0.0)
    return (r, leg - (s - leg - arc), 0.0)


def _distance_to_segment_2d(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    denom = vx * vx + vy * vy
    t = 0.0 if denom == 0.0 else _clamp((wx * vx + wy * vy) / denom, 0.0, 1.0)
    dx = px - (ax + t * vx)
    dy = py - (ay + t * vy)
    return math.hypot(dx, dy)


def distance_to_centerline(point: Vec3, spec: WireSpec = WireSpec()) -> float:
    """Exact distance to the two straight legs plus planar semicircular centerline."""

    x, y, z = point
    r = spec.bend_radius_mm
    leg = spec.leg_length_mm

    d_left_xy = _distance_to_segment_2d(x, y, -r, 0.0, -r, leg)
    d_right_xy = _distance_to_segment_2d(x, y, r, leg, r, 0.0)

    qx, qy = x, y - leg
    if qy >= 0.0:
        radial = math.hypot(qx, qy)
        d_arc_xy = abs(radial - r)
    else:
        d_arc_xy = min(math.hypot(x + r, y - leg), math.hypot(x - r, y - leg))

    d_xy = min(d_left_xy, d_right_xy, d_arc_xy)
    return math.hypot(d_xy, z)


def sdf(point: Vec3, spec: WireSpec = WireSpec()) -> float:
    """Signed distance to the wire volume: negative inside, zero on the surface."""

    return distance_to_centerline(point, spec) - spec.radius_mm


def hard_density(point: Vec3, spec: WireSpec = WireSpec()) -> float:
    """Authoritative solid-material occupancy."""

    return spec.copper_density if sdf(point, spec) <= 0.0 else 0.0


def soft_density(point: Vec3, spec: WireSpec = WireSpec()) -> float:
    """Smooth rendering surrogate for the hard occupancy boundary."""

    x = sdf(point, spec) / spec.smooth_width_mm
    if x >= 40.0:
        return 0.0
    if x <= -40.0:
        return spec.copper_density
    return spec.copper_density / (1.0 + math.exp(x))


def sdf_gradient(point: Vec3, spec: WireSpec = WireSpec(), eps: float = 1.0e-4) -> Vec3:
    if eps <= 0.0 or not math.isfinite(eps):
        raise ValueError("eps must be finite and positive")
    p = list(point)
    grad = []
    for axis in range(3):
        plus = p.copy()
        minus = p.copy()
        plus[axis] += eps
        minus[axis] -= eps
        plus_point: Vec3 = (plus[0], plus[1], plus[2])
        minus_point: Vec3 = (minus[0], minus[1], minus[2])
        grad.append((sdf(plus_point, spec) - sdf(minus_point, spec)) / (2.0 * eps))
    return (grad[0], grad[1], grad[2])


def surface_normal(point: Vec3, spec: WireSpec = WireSpec(), eps: float = 1.0e-4) -> Vec3:
    g = sdf_gradient(point, spec, eps)
    norm = math.sqrt(sum(v * v for v in g))
    if norm <= 1.0e-12:
        return (0.0, 0.0, 0.0)
    return (g[0] / norm, g[1] / norm, g[2] / norm)


def eikonal_residual(point: Vec3, spec: WireSpec = WireSpec(), eps: float = 1.0e-4) -> float:
    """| ||grad f||_2 - 1 |; expected small away from SDF non-smooth loci."""

    g = sdf_gradient(point, spec, eps)
    return abs(math.sqrt(sum(v * v for v in g)) - 1.0)


def render_pixel(
    x_mm: float,
    y_mm: float,
    spec: WireSpec = WireSpec(),
    *,
    z_extent_mm: float = 1.5,
    samples: int = 96,
    absorption: float = 3.0,
) -> float:
    """Orthographic differentiable silhouette renderer using fixed quadrature."""

    if samples < 2:
        raise ValueError("samples must be >= 2")
    if z_extent_mm <= 0.0 or absorption <= 0.0:
        raise ValueError("z_extent_mm and absorption must be positive")
    dz = 2.0 * z_extent_mm / (samples - 1)
    optical_depth = 0.0
    for i in range(samples):
        z = -z_extent_mm + i * dz
        weight = 0.5 if i in (0, samples - 1) else 1.0
        optical_depth += weight * soft_density((x_mm, y_mm, z), spec) * dz
    return 1.0 - math.exp(-absorption * optical_depth)


def render_bend_crop(
    spec: WireSpec = WireSpec(),
    *,
    width: int = 24,
    height: int = 24,
    half_extent_mm: float = 3.0,
    samples: int = 64,
) -> tuple[tuple[float, ...], ...]:
    """Render a square crop centered on the hairpin bend."""

    if width < 2 or height < 2:
        raise ValueError("width and height must be >= 2")
    if half_extent_mm <= 0.0:
        raise ValueError("half_extent_mm must be positive")
    cy = spec.leg_length_mm
    rows = []
    for j in range(height):
        y = cy - half_extent_mm + 2.0 * half_extent_mm * j / (height - 1)
        row = []
        for i in range(width):
            x = -half_extent_mm + 2.0 * half_extent_mm * i / (width - 1)
            row.append(render_pixel(x, y, spec, samples=samples))
        rows.append(tuple(row))
    return tuple(rows)


def patch_tokens(
    image: Sequence[Sequence[float]], patch: int = 4
) -> tuple[tuple[float, float, float], ...]:
    """Map image patches to compact mean/variance/energy visual tokens."""

    if patch <= 0:
        raise ValueError("patch must be positive")
    h = len(image)
    if h == 0:
        return ()
    w = len(image[0])
    if w == 0 or any(len(row) != w for row in image):
        raise ValueError("image must be rectangular and non-empty")

    tokens = []
    for y0 in range(0, h, patch):
        for x0 in range(0, w, patch):
            values = [
                float(image[y][x])
                for y in range(y0, min(h, y0 + patch))
                for x in range(x0, min(w, x0 + patch))
            ]
            mean = sum(values) / len(values)
            var = sum((v - mean) ** 2 for v in values) / len(values)
            energy = sum(v * v for v in values) / len(values)
            tokens.append((mean, var, energy))
    return tuple(tokens)


def self_attention(tokens: Sequence[Vec], temperature: float | None = None) -> tuple[Vec, ...]:
    """Single-head Q=K=V reference attention over equal-width tokens."""

    if not tokens:
        return ()
    dim = len(tokens[0])
    if dim == 0 or any(len(t) != dim for t in tokens):
        raise ValueError("tokens must have one non-zero shared dimension")
    scale = math.sqrt(dim) if temperature is None else float(temperature)
    if scale <= 0.0 or not math.isfinite(scale):
        raise ValueError("attention scale must be finite and positive")

    result = []
    for q in tokens:
        logits = [sum(a * b for a, b in zip(q, k)) / scale for k in tokens]
        shift = max(logits)
        weights = [math.exp(v - shift) for v in logits]
        denom = sum(weights)
        weights = [w / denom for w in weights]
        result.append(
            tuple(
                sum(weight * token[d] for weight, token in zip(weights, tokens))
                for d in range(dim)
            )
        )
    return tuple(result)


def image_feature_vector(
    spec: WireSpec,
    *,
    width: int = 16,
    height: int = 16,
    patch: int = 4,
    samples: int = 48,
) -> Vec:
    image = render_bend_crop(spec, width=width, height=height, samples=samples)
    attended = self_attention(patch_tokens(image, patch=patch))
    if not attended:
        return ()
    dim = len(attended[0])
    return tuple(sum(t[d] for t in attended) / len(attended) for d in range(dim))


def total_loss(
    spec: WireSpec,
    target_features: Vec,
    *,
    diameter_target_mm: float = 1.0,
    gap_target_mm: float = 2.0,
    feature_weight: float = 1.0,
    physical_weight: float = 0.25,
    render_samples: int = 32,
) -> float:
    """Coupled feature + physical loss for the hairpin reference problem."""

    features = image_feature_vector(spec, samples=render_samples)
    if len(features) != len(target_features):
        raise ValueError("target feature dimension mismatch")
    feature_loss = sum((a - b) ** 2 for a, b in zip(features, target_features))
    physical_loss = (
        (spec.diameter_mm - diameter_target_mm) ** 2
        + (spec.inner_gap_mm - gap_target_mm) ** 2
    )
    return feature_weight * feature_loss + physical_weight * physical_loss


def _with_latent(base: WireSpec, z: Sequence[float]) -> WireSpec:
    if len(z) != 2:
        raise ValueError("latent state must be (diameter_mm, inner_gap_mm)")
    diameter = max(1.0e-4, float(z[0]))
    gap = max(1.0e-4, float(z[1]))
    return WireSpec(
        total_length_mm=base.total_length_mm,
        diameter_mm=diameter,
        inner_gap_mm=gap,
        copper_density=base.copper_density,
        smooth_width_mm=base.smooth_width_mm,
    )


class _CachedLoss:
    """Memoize deterministic loss evaluations within one optimization transaction."""

    def __init__(self, loss_fn: Callable[[Vec], float]) -> None:
        self._loss_fn = loss_fn
        self._cache: dict[Vec, float] = {}
        self.evaluations = 0

    def __call__(self, z: Vec) -> float:
        key = tuple(float(value) for value in z)
        if key not in self._cache:
            self._cache[key] = float(self._loss_fn(key))
            self.evaluations += 1
        return self._cache[key]


def numerical_gradient_hessian(
    loss_fn: Callable[[Vec], float],
    z: Vec,
    eps: float = 2.0e-3,
) -> tuple[Vec, tuple[Vec, ...], float]:
    """Compute central-difference gradient and Hessian from one shared stencil.

    For a two-dimensional latent state this needs nine unique loss evaluations
    instead of separately evaluating the gradient and Hessian stencils.
    """

    if eps <= 0.0 or not math.isfinite(eps):
        raise ValueError("eps must be finite and positive")
    n = len(z)
    if n == 0:
        raise ValueError("z cannot be empty")

    f0 = loss_fn(z)
    plus_values = [0.0] * n
    minus_values = [0.0] * n
    gradient = [0.0] * n
    hessian = [[0.0] * n for _ in range(n)]

    for i in range(n):
        plus = list(z)
        minus = list(z)
        plus[i] += eps
        minus[i] -= eps
        plus_values[i] = loss_fn(tuple(plus))
        minus_values[i] = loss_fn(tuple(minus))
        gradient[i] = (plus_values[i] - minus_values[i]) / (2.0 * eps)
        hessian[i][i] = (plus_values[i] - 2.0 * f0 + minus_values[i]) / (eps * eps)

    for i in range(n):
        for j in range(i + 1, n):
            pp, pm, mp, mm = [list(z) for _ in range(4)]
            pp[i] += eps
            pp[j] += eps
            pm[i] += eps
            pm[j] -= eps
            mp[i] -= eps
            mp[j] += eps
            mm[i] -= eps
            mm[j] -= eps
            value = (
                loss_fn(tuple(pp))
                - loss_fn(tuple(pm))
                - loss_fn(tuple(mp))
                + loss_fn(tuple(mm))
            ) / (4.0 * eps * eps)
            hessian[i][j] = hessian[j][i] = value

    return tuple(gradient), tuple(tuple(row) for row in hessian), f0


def numerical_gradient(loss_fn: Callable[[Vec], float], z: Vec, eps: float = 1.0e-3) -> Vec:
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    out = []
    for i in range(len(z)):
        plus = list(z)
        minus = list(z)
        plus[i] += eps
        minus[i] -= eps
        out.append((loss_fn(tuple(plus)) - loss_fn(tuple(minus))) / (2.0 * eps))
    return tuple(out)


def numerical_hessian(
    loss_fn: Callable[[Vec], float], z: Vec, eps: float = 2.0e-3
) -> tuple[Vec, ...]:
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    n = len(z)
    f0 = loss_fn(z)
    h = [[0.0] * n for _ in range(n)]
    for i in range(n):
        plus = list(z)
        minus = list(z)
        plus[i] += eps
        minus[i] -= eps
        h[i][i] = (loss_fn(tuple(plus)) - 2.0 * f0 + loss_fn(tuple(minus))) / (eps * eps)
        for j in range(i + 1, n):
            pp, pm, mp, mm = [list(z) for _ in range(4)]
            pp[i] += eps
            pp[j] += eps
            pm[i] += eps
            pm[j] -= eps
            mp[i] -= eps
            mp[j] += eps
            mm[i] -= eps
            mm[j] -= eps
            value = (
                loss_fn(tuple(pp))
                - loss_fn(tuple(pm))
                - loss_fn(tuple(mp))
                + loss_fn(tuple(mm))
            ) / (4.0 * eps * eps)
            h[i][j] = h[j][i] = value
    return tuple(tuple(row) for row in h)


def damped_newton_step(
    loss_fn: Callable[[Vec], float],
    z: Vec,
    *,
    damping: float = 1.0e-2,
    trust_radius: float = 0.25,
    derivative_eps: float = 2.0e-3,
    max_backtracks: int = 6,
    backtrack_factor: float = 0.5,
) -> Vec:
    """Bounded two-dimensional Newton step with monotonic backtracking.

    The gradient and Hessian share one finite-difference stencil. A trust radius
    bounds the proposal and a short backtracking line search rejects any step
    that increases the measured objective.
    """

    if len(z) != 2:
        raise ValueError("reference Newton solver expects a two-dimensional latent state")
    if damping <= 0.0 or trust_radius <= 0.0:
        raise ValueError("damping and trust_radius must be positive")
    if max_backtracks < 0:
        raise ValueError("max_backtracks must be non-negative")
    if not 0.0 < backtrack_factor < 1.0:
        raise ValueError("backtrack_factor must lie in (0, 1)")

    g, h, f0 = numerical_gradient_hessian(loss_fn, z, eps=derivative_eps)
    a = h[0][0] + damping
    b = h[0][1]
    cc = h[1][0]
    d = h[1][1] + damping
    det = a * d - b * cc

    if abs(det) <= 1.0e-12:
        step = (
            -g[0] / a if abs(a) > 1.0e-12 else 0.0,
            -g[1] / d if abs(d) > 1.0e-12 else 0.0,
        )
    else:
        step = ((-d * g[0] + b * g[1]) / det, (cc * g[0] - a * g[1]) / det)

    norm = math.hypot(step[0], step[1])
    if norm > trust_radius:
        scale = trust_radius / norm
        step = (step[0] * scale, step[1] * scale)

    scale = 1.0
    for _ in range(max_backtracks + 1):
        candidate = (z[0] + scale * step[0], z[1] + scale * step[1])
        if loss_fn(candidate) <= f0:
            return candidate
        scale *= backtrack_factor
    return z


def fixed_point_residual(
    phi: Callable[[Vec], Vec],
    z: Vec,
    *,
    first: Vec | None = None,
) -> float:
    """||Phi(Phi(z)) - Phi(z)||_2, the idempotent fixed-point certificate."""

    first = phi(z) if first is None else first
    second = phi(first)
    if len(first) != len(second):
        raise ValueError("Phi changed latent dimensionality")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(second, first)))


@dataclass(frozen=True)
class CoupledIteration:
    z_before: Vec
    z_after: Vec
    loss_before: float
    loss_after: float
    fixed_point_residual: float
    step_norm: float
    loss_evaluations: int


@dataclass(frozen=True)
class CoupledOptimization:
    z_initial: Vec
    z_final: Vec
    loss_initial: float
    loss_final: float
    iterations: int
    total_loss_evaluations: int
    converged: bool
    history: tuple[CoupledIteration, ...]


def coupled_iteration(
    base: WireSpec,
    z: Vec,
    target_features: Vec,
    *,
    damping: float = 1.0e-2,
    trust_radius: float = 0.20,
    render_samples: int = 24,
) -> CoupledIteration:
    """Run one memoized 3D -> 2D -> attention -> Newton latent update."""

    def raw_loss(latent: Vec) -> float:
        return total_loss(
            _with_latent(base, latent),
            target_features,
            render_samples=render_samples,
        )

    loss_fn = _CachedLoss(raw_loss)

    def phi(latent: Vec) -> Vec:
        return damped_newton_step(
            loss_fn,
            latent,
            damping=damping,
            trust_radius=trust_radius,
        )

    loss_before = loss_fn(z)
    after = phi(z)
    loss_after = loss_fn(after)
    residual = fixed_point_residual(phi, z, first=after)
    step_norm = math.hypot(after[0] - z[0], after[1] - z[1])
    return CoupledIteration(
        z_before=z,
        z_after=after,
        loss_before=loss_before,
        loss_after=loss_after,
        fixed_point_residual=residual,
        step_norm=step_norm,
        loss_evaluations=loss_fn.evaluations,
    )


def optimize_coupled(
    base: WireSpec,
    initial_z: Vec,
    target_features: Vec,
    *,
    max_iterations: int = 8,
    tolerance: float = 1.0e-4,
    loss_tolerance: float = 1.0e-6,
    initial_damping: float = 1.0e-2,
    initial_trust_radius: float = 0.30,
    min_trust_radius: float = 0.02,
    max_trust_radius: float = 0.60,
    render_samples: int = 24,
) -> CoupledOptimization:
    """Auto-optimize the coupled latent state with adaptive trust and damping.

    Internal fixed-point stability and external objective error are both required
    before the result is marked converged.
    """

    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if tolerance <= 0.0 or loss_tolerance < 0.0:
        raise ValueError("tolerances must be valid")
    if not 0.0 < min_trust_radius <= initial_trust_radius <= max_trust_radius:
        raise ValueError("trust radii must satisfy min <= initial <= max")

    current = tuple(float(value) for value in initial_z)
    if len(current) != 2:
        raise ValueError("initial_z must be (diameter_mm, inner_gap_mm)")

    damping = float(initial_damping)
    trust_radius = float(initial_trust_radius)
    history: list[CoupledIteration] = []
    total_evaluations = 0
    converged = False

    for _ in range(max_iterations):
        iteration = coupled_iteration(
            base,
            current,
            target_features,
            damping=damping,
            trust_radius=trust_radius,
            render_samples=render_samples,
        )
        history.append(iteration)
        total_evaluations += iteration.loss_evaluations
        current = iteration.z_after

        if iteration.loss_after < iteration.loss_before:
            damping = max(1.0e-6, damping * 0.7)
            trust_radius = min(max_trust_radius, trust_radius * 1.2)
        else:
            damping = min(1.0e4, damping * 4.0)
            trust_radius = max(min_trust_radius, trust_radius * 0.5)

        converged = (
            iteration.step_norm <= tolerance
            and iteration.fixed_point_residual <= tolerance
            and iteration.loss_after <= loss_tolerance
        )
        if converged:
            break

    first = history[0]
    last = history[-1]
    return CoupledOptimization(
        z_initial=tuple(float(value) for value in initial_z),
        z_final=current,
        loss_initial=first.loss_before,
        loss_final=last.loss_after,
        iterations=len(history),
        total_loss_evaluations=total_evaluations,
        converged=converged,
        history=tuple(history),
    )


def eikonal_audit(points: Iterable[Vec3], spec: WireSpec = WireSpec()) -> float:
    """Return the worst sampled Eikonal residual."""

    residuals = [eikonal_residual(point, spec) for point in points]
    return max(residuals, default=0.0)
