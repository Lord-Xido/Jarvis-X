"""Bounded Phi_M 3D geometric field operator (opt-in research component).

Phi_M(x,t) = tanh(alpha*cos(omega*||x||)*x + integral_0^t grad_x z(x,tau) dtau).
The source image does not specify z or sigma. This module explicitly chooses a
moving Gaussian scalar potential z and componentwise tanh. The graph bridge is
candidate-only: existing authoritative geometric runtimes are never mutated.
Requires NumPy; SciPy enables acceleration (pip install -e '.[phi_m]').
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .geometric_diffusion_runtime import GeometricGraph


def _scalar(value: float, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or not low <= result <= high:
        raise ValueError(f"{name} must be finite and within [{low}, {high}]")
    return result


@dataclass(frozen=True)
class PhiMConfig:
    alpha: float = 0.92
    omega: float = 2.8
    amplitude: float = 0.55
    spatial_sigma: float = 0.72
    modulation: float = 0.35
    frequency: float = 1.6
    center: tuple[float, float, float] = (0.35, -0.25, 0.16)
    velocity: tuple[float, float, float] = (0.13, -0.08, 0.08)
    quadrature_order: int = 32
    max_time: float = 10.0
    max_points: int = 4096

    def __post_init__(self) -> None:
        for name, lo, hi in (
            ("alpha", 0.0, 10.0), ("omega", 0.0, 100.0),
            ("amplitude", 0.0, 10.0), ("spatial_sigma", 0.01, 100.0),
            ("modulation", 0.0, 1.0), ("frequency", 0.0, 100.0),
            ("max_time", 0.0, 100.0),
        ):
            _scalar(getattr(self, name), name, lo, hi)
        for name in ("center", "velocity"):
            values = getattr(self, name)
            if len(values) != 3:
                raise ValueError(f"{name} must be a three-component vector")
            for item in values:
                _scalar(item, name, -100.0, 100.0)
        for name, upper in (("quadrature_order", 64), ("max_points", 4096)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= upper:
                raise ValueError(f"{name} must be an integer in [1, {upper}]")


def _points(points: object, config: PhiMConfig) -> tuple[np.ndarray, tuple[int, ...]]:
    x = np.asarray(points, dtype=np.float64)
    if x.ndim < 1 or x.shape[-1] != 3 or not 1 <= x.size // 3 <= config.max_points:
        raise ValueError("points must have shape (..., 3) within the configured point budget")
    if not np.isfinite(x).all() or np.max(np.abs(x)) > 1000.0:
        raise ValueError("coordinates must be finite and within [-1000, 1000]")
    return x.reshape(-1, 3), x.shape


def _time(t: float, config: PhiMConfig) -> float:
    return _scalar(t, "time", 0.0, config.max_time)


def _gaussian(
    x: np.ndarray, times: np.ndarray, config: PhiMConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return z, spatial gradient and spatial Hessian, broadcast over times and points."""
    c = np.asarray(config.center)[None, :] + times[:, None] * np.asarray(config.velocity)
    d = x[None, :, :] - c[:, None, :]
    s2 = config.spatial_sigma**2
    amp = config.amplitude * (1.0 + config.modulation * np.sin(config.frequency * times))
    z = amp[:, None] * np.exp(-np.sum(d * d, axis=-1) / (2.0 * s2))
    grad = -z[:, :, None] * d / s2
    hessian = z[:, :, None, None] * (
        d[:, :, :, None] * d[:, :, None, :] / (s2**2) - np.eye(3) / s2
    )
    return z, grad, hessian


def potential(points: object, t: float, config: PhiMConfig = PhiMConfig()) -> np.ndarray:
    x, shape = _points(points, config)
    z, _, _ = _gaussian(x, np.array([_time(t, config)]), config)
    return np.asarray(z[0].reshape(shape[:-1]), dtype=np.float64)


def gradient(points: object, t: float, config: PhiMConfig = PhiMConfig()) -> np.ndarray:
    x, shape = _points(points, config)
    _, grad, _ = _gaussian(x, np.array([_time(t, config)]), config)
    return np.asarray(grad[0].reshape(shape), dtype=np.float64)


def integrated_gradient_hessian_quadrature(
    points: object, t: float, config: PhiMConfig = PhiMConfig()
) -> tuple[np.ndarray, np.ndarray]:
    """Independent Gauss-Legendre numerical reference, O(N*Q) work and memory."""
    x, shape = _points(points, config)
    t = _time(t, config)
    if t == 0.0:
        return np.zeros(shape), np.zeros(shape[:-1] + (3, 3))
    nodes, weights = np.polynomial.legendre.leggauss(config.quadrature_order)
    times = 0.5 * t * (nodes + 1.0)
    _, grad, hessian = _gaussian(x, times, config)
    g = 0.5 * t * np.einsum("q,qni->ni", weights, grad)
    h = 0.5 * t * np.einsum("q,qnij->nij", weights, hessian)
    return g.reshape(shape), h.reshape(shape[:-1] + (3, 3))


def _complex_gaussian_moments(
    a: float, lam: np.ndarray, t: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """I_k=integral_0^t tau**k exp(-a*tau**2+lam*tau) dtau, k=0,1,2.

    erfcx form avoids subtracting exponentially large erf terms. Higher moments
    follow the integration-by-parts recurrence. Requires SciPy for complex erfcx.
    """
    from scipy.special import erfcx

    root = np.sqrt(a)
    u0 = -lam / (2.0 * root)
    u1 = root * t + u0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        endpoint = np.exp(-a * t * t + lam * t)
        i0 = np.sqrt(np.pi) / (2.0 * root) * (erfcx(u0) - endpoint * erfcx(u1))
        i1 = (lam * i0 + 1.0 - endpoint) / (2.0 * a)
        i2 = (i0 + lam * i1 - t * endpoint) / (2.0 * a)
    return i0, i1, i2


def integrated_gradient_hessian_analytic(
    points: object, t: float, config: PhiMConfig = PhiMConfig()
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate temporal Gaussian moments in O(N), not N quadrature samples.

    For moving Gaussian + sinusoidal amplitude, use complex error functions to
    compute 0th, 1st and 2nd temporal moments of the envelope. The v=0 case
    has a simpler exact limit. Numerically ill-conditioned near-zero velocities
    or nonfinite intermediate moments use the quadrature safety fallback.
    """
    x, shape = _points(points, config)
    t = _time(t, config)
    if t == 0.0:
        return np.zeros(shape), np.zeros(shape[:-1] + (3, 3))

    v = np.asarray(config.velocity, dtype=np.float64)
    d0 = x - np.asarray(config.center, dtype=np.float64)
    s2 = config.spatial_sigma**2
    a = float(np.dot(v, v) / (2.0 * s2))
    c = -np.sum(d0 * d0, axis=1) / (2.0 * s2)
    if a == 0.0:
        duration = (
            t if config.frequency == 0.0 else
            t + config.modulation * (1.0 - np.cos(config.frequency * t))
            / config.frequency
        )
        m0 = np.exp(c) * duration
        m1 = np.zeros_like(m0)
        m2 = np.zeros_like(m0)
    else:
        # Recurrences divide by a twice; avoid catastrophic cancellation
        # when the moving source is effectively stationary over [0,t].
        if np.linalg.norm(v) * t / config.spatial_sigma < 0.001:
            return integrated_gradient_hessian_quadrature(x.reshape(shape), t, config)
        b = d0 @ v / s2
        try:
            i_real = _complex_gaussian_moments(a, b, t)
            i_complex = _complex_gaussian_moments(a, b + 1j * config.frequency, t)
        except ImportError as exc:
            raise ImportError("Install scipy for analytic Phi_M integration: pip install scipy") from exc
        moments = np.exp(c)[None, :] * (
            np.stack(i_real).real + config.modulation * np.stack(i_complex).imag
        )
        if not np.isfinite(moments).all():
            return integrated_gradient_hessian_quadrature(x.reshape(shape), t, config)
        m0, m1, m2 = moments

    g = -config.amplitude / s2 * (d0 * m0[:, None] - v * m1[:, None])
    h = config.amplitude * (
        (d0[:, :, None] * d0[:, None, :]) * m0[:, None, None]
        - (d0[:, :, None] * v[None, None, :] + v[None, :, None] * d0[:, None, :])
        * m1[:, None, None]
        + np.outer(v, v)[None, :, :] * m2[:, None, None]
    ) / (s2 * s2) - config.amplitude * m0[:, None, None] * np.eye(3) / s2
    return g.reshape(shape), h.reshape(shape[:-1] + (3, 3))


def integrated_gradient_hessian(
    points: object, t: float, config: PhiMConfig = PhiMConfig(),
    *, method: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Select 'analytic', 'quadrature', or 'auto' (analytic when SciPy exists)."""
    if method not in ("auto", "analytic", "quadrature"):
        raise ValueError("method must be auto, analytic, or quadrature")
    if method == "quadrature":
        return integrated_gradient_hessian_quadrature(points, t, config)
    if method == "analytic":
        return integrated_gradient_hessian_analytic(points, t, config)
    try:
        return integrated_gradient_hessian_analytic(points, t, config)
    except ImportError:
        return integrated_gradient_hessian_quadrature(points, t, config)


def phi_m(points: object, t: float, config: PhiMConfig = PhiMConfig()) -> np.ndarray:
    """Return componentwise tanh of radial warp plus integrated spatial gradient."""
    x, shape = _points(points, config)
    g, _ = integrated_gradient_hessian(x, t, config)
    r = np.linalg.norm(x, axis=1, keepdims=True)
    u = config.alpha * np.cos(config.omega * r) * x + g
    return np.asarray(np.tanh(u).reshape(shape), dtype=np.float64)


def spatial_jacobian(
    points: object, t: float, config: PhiMConfig = PhiMConfig()
) -> np.ndarray:
    """Analytic d(Phi_M)/d(x), using the selected field-integral backend."""
    x, shape = _points(points, config)
    g, h = integrated_gradient_hessian(x, t, config)
    r = np.linalg.norm(x, axis=1)
    scale = config.alpha * np.cos(config.omega * r)
    radial = np.zeros_like(r)
    mask = r > 1e-12
    radial[mask] = -config.alpha * config.omega * np.sin(config.omega * r[mask]) / r[mask]
    du = (scale[:, None, None] * np.eye(3)
          + radial[:, None, None] * x[:, :, None] * x[:, None, :] + h)
    u = scale[:, None] * x + g
    y = np.tanh(u)
    jacobian = (1.0 - y**2)[:, :, None] * du
    return np.asarray(jacobian.reshape(shape[:-1] + (3, 3)), dtype=np.float64)


def temporal_derivative(
    points: object, t: float, config: PhiMConfig = PhiMConfig()
) -> np.ndarray:
    """d(Phi_M)/dt = (1-Phi_M**2) * grad_x z(x,t), up to quadrature error."""
    y = phi_m(points, t, config)
    return np.asarray((1.0 - y**2) * gradient(points, t, config), dtype=np.float64)


def static_source_integral(
    points: object, t: float, config: PhiMConfig = PhiMConfig()
) -> np.ndarray:
    """Independent closed form if the Gaussian center is stationary."""
    if any(component != 0 for component in config.velocity):
        raise ValueError("closed form requires stationary source")
    x, shape = _points(points, config)
    t = _time(t, config)
    d = x - np.asarray(config.center)
    s2 = config.spatial_sigma**2
    duration = (t if config.frequency == 0.0 else
                t + config.modulation * (1 - np.cos(config.frequency * t))
                / config.frequency)
    z0 = config.amplitude * np.exp(-np.sum(d * d, axis=1) / (2.0 * s2))
    return np.asarray((-z0[:, None] * d / s2 * duration).reshape(shape), dtype=np.float64)


def graph_candidate(
    graph: "GeometricGraph",
    t: float,
    config: PhiMConfig = PhiMConfig(),
    *,
    max_position_step: float = 0.25,
    max_rms: float = 0.15,
) -> "GeometricGraph":
    """Produce a bounded candidate graph; preserve topology/features and never commit.

    The caller must independently verify and authorize promotion. This adapter
    neither calls GeometricDiffusionRuntime.step nor mutates its state.
    """
    from .geometric_diffusion_runtime import (
        GeometricGraph, GeometricNode, graph_rms, project_candidate,
    )

    step = _scalar(max_position_step, "max_position_step", 0.0, 100.0)
    limit = _scalar(max_rms, "max_rms", 0.0, 100.0)
    if not 1 <= len(graph.nodes) <= config.max_points:
        raise ValueError("graph exceeds Phi_M point budget")
    positions = np.array([node.position for node in graph.nodes], dtype=np.float64)
    mapped = phi_m(positions, t, config)
    raw = GeometricGraph(
        tuple(GeometricNode(tuple(map(float, pos)), node.feature)
              for pos, node in zip(mapped, graph.nodes)),
        graph.edges,
    )
    projected = project_candidate(
        raw, graph, max_position_step=step, max_feature_step=0.0,
    )
    if graph_rms(graph, projected)[2] > limit:
        raise RuntimeError("Phi_M graph candidate rejected by RMS bound")
    return projected


def numerical_self_test(config: PhiMConfig = PhiMConfig()) -> dict[str, float | bool]:
    """Independent finite differences plus a stationary-source closed-form oracle."""
    rng = np.random.default_rng(20261029)
    x = rng.uniform(-0.8, 0.8, (7, 3))
    t, eps = 1.35, 1e-5
    if config.max_time < t:
        raise ValueError("self-test requires max_time >= 1.35")
    analytical = spatial_jacobian(x, t, config)
    finite_diff = np.empty_like(analytical)
    for axis in range(3):
        displacement = np.zeros_like(x)
        displacement[:, axis] = eps
        finite_diff[:, :, axis] = (
            phi_m(x + displacement, t, config) - phi_m(x - displacement, t, config)
        ) / (2.0 * eps)
    dt = (phi_m(x, t + eps, config) - phi_m(x, t - eps, config)) / (2.0 * eps)
    static = PhiMConfig(**{**vars(config), "velocity": (0.0, 0.0, 0.0)})
    integral, _ = integrated_gradient_hessian(x, t, static)
    oracle = static_source_integral(x, t, static)
    z = phi_m(x, 0.0, config)
    radial = np.tanh(config.alpha * np.cos(config.omega * np.linalg.norm(x, axis=1, keepdims=True)) * x)
    errors = {
        "spatial_jacobian_max_abs_error": float(np.max(np.abs(analytical - finite_diff))),
        "temporal_derivative_max_abs_error": float(np.max(np.abs(temporal_derivative(x, t, config) - dt))),
        "stationary_integral_max_abs_error": float(np.max(np.abs(integral - oracle))),
        "zero_time_max_abs_error": float(np.max(np.abs(z - radial))),
    }
    passes = (errors["spatial_jacobian_max_abs_error"] < 2e-7 and
              errors["temporal_derivative_max_abs_error"] < 2e-7 and
              errors["stationary_integral_max_abs_error"] < 2e-8 and
              errors["zero_time_max_abs_error"] < 1e-14)
    return {**errors, "passed": bool(passes)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time", type=float, default=1.5)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    config = PhiMConfig()
    if args.self_test:
        report = numerical_self_test(config)
        print(json.dumps(report, indent=2, sort_keys=True))
        if not report["passed"]:
            raise SystemExit(1)
    else:
        x = np.array([[0.35, -0.20, 0.50], [0.0, 0.0, 0.0]])
        print(json.dumps({"time": args.time, "input": x.tolist(),
                          "output": phi_m(x, args.time, config).tolist()}, indent=2))


if __name__ == "__main__":
    main()
