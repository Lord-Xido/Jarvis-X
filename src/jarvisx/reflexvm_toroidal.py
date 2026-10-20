"""Seam-safe, reference-anchored 3D fixed-point engine on the flat torus.

The codec T³ -> R⁶ -> T³ is locally isometric and round-trips modulo
floating-point error. It is *not* a learned autoencoder or data compressor.
The iteration contracts each point toward its chosen anchor, not the origin.
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np

TAU = 2.0 * math.pi


def _period(value: float) -> float:
    p = float(value)
    if not math.isfinite(p) or p <= 0.0:
        raise ValueError("period must be finite and positive")
    return p


def _xyz(values: object, label: str) -> np.ndarray:
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all():
        raise ValueError(f"{label} must contain finite coordinates of shape (N, 3)")
    return a


def fold(points: object, period: float = 1000.0) -> np.ndarray:
    return np.remainder(_xyz(points, "points"), _period(period))


def periodic_delta(a: object, b: object, period: float = 1000.0) -> np.ndarray:
    """Signed shortest-arc displacement along each periodic axis."""
    p = _period(period)
    u, v = _xyz(a, "a"), _xyz(b, "b")
    if v.shape[0] not in (1, u.shape[0]):
        raise ValueError("b must have one row or match a's rows")
    return np.remainder(u - v + p / 2.0, p) - p / 2.0


def periodic_max_error(a: object, b: object, period: float = 1000.0) -> np.ndarray:
    return np.max(np.abs(periodic_delta(a, b, period)), axis=-1)


def encode6(points: object, period: float = 1000.0) -> np.ndarray:
    """Locally isometric product-of-circles embedding T³ into R⁶."""
    p = _period(period)
    theta = (TAU / p) * fold(points, p)
    return ((p / TAU) * np.stack((np.cos(theta), np.sin(theta)), axis=-1)).reshape(-1, 6)


def decode6(latent: object, period: float = 1000.0) -> np.ndarray:
    """Inverse on the embedded circle manifold; does not invert arbitrary R⁶ data."""
    p = _period(period)
    z = np.asarray(latent, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] != 6 or not np.isfinite(z).all():
        raise ValueError("latent must be finite with shape (N, 6)")
    pairs = z.reshape(-1, 3, 2)
    if np.any(np.hypot(pairs[..., 0], pairs[..., 1]) == 0.0):
        raise ValueError("zero-length circle coordinate")
    return np.remainder((p / TAU) * np.arctan2(pairs[..., 1], pairs[..., 0]), p)


def toroidal_contraction(
    points: object, anchors: object, period: float = 1000.0, q: float = 0.5
) -> np.ndarray:
    """F_a(x) = a + qR/(2pi) sin(2pi(x-a)/R) (mod R).

    For fixed anchors, its intrinsic flat-torus Lipschitz constant is <= q.
    "anchors" may be (1,3) or (N,3).
    """
    p = _period(period)
    if not math.isfinite(q) or not 0.0 < q < 1.0:
        raise ValueError("q must lie strictly between zero and one")
    x, a = fold(points, p), fold(anchors, p)
    if a.shape[0] not in (1, x.shape[0]):
        raise ValueError("anchors must be (1,3) or (N,3)")
    delta = periodic_delta(x, a, p)
    return fold(a + (q * p / TAU) * np.sin((TAU / p) * delta), p)


def cycle_explicit(
    points: object, anchors: object, period: float = 1000.0, q: float = 0.5
) -> np.ndarray:
    """Numerical E->D->Psi->E->D path (fused path removes E/D identities)."""
    decoded = decode6(encode6(points, period), period)
    refined = toroidal_contraction(decoded, anchors, period, q)
    return decode6(encode6(refined, period), period)


def refine(
    points: object,
    anchors: object,
    period: float = 1000.0,
    q: float = 0.5,
    tol: float = 1e-11,
    max_iters: int = 100,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Iterate to reference anchors; report true periodic error, not only step size."""
    p = _period(period)
    x, a = fold(points, p), fold(anchors, p)
    if a.shape[0] not in (1, x.shape[0]):
        raise ValueError("anchors must be (1,3) or (N,3)")
    if not math.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be finite and positive")
    if type(max_iters) is not int or max_iters < 0:
        raise ValueError("max_iters must be a nonnegative integer")
    if not math.isfinite(q) or not 0.0 < q < 1.0:
        raise ValueError("q must lie strictly between zero and one")
    errors = [float(np.max(periodic_max_error(x, a, p))) if len(x) else 0.0]
    for _ in range(max_iters):
        if errors[-1] <= tol:
            break
        x = toroidal_contraction(x, a, p, q)
        errors.append(float(np.max(periodic_max_error(x, a, p))))
    return x, np.asarray(errors), len(errors) - 1


def benchmark(n: int = 10000, steps: int = 40, explicit: bool = False) -> dict:
    """Time complete trajectories; report point-steps separately."""
    if type(n) is not int or type(steps) is not int or n < 1 or steps < 1:
        raise ValueError("n and steps must be positive integers")
    rng = np.random.default_rng(42)
    anchor = rng.uniform(0.0, 1000.0, (n, 3))
    points = fold(anchor + rng.uniform(-100.0, 100.0, (n, 3)))
    operator = cycle_explicit if explicit else toroidal_contraction
    start = time.perf_counter()
    for _ in range(steps):
        points = operator(points, anchor)
    elapsed = time.perf_counter() - start
    return {
        "points": n,
        "steps": steps,
        "mode": "explicit" if explicit else "fused",
        "seconds": elapsed,
        "trajectories_per_second": n / elapsed,
        "point_steps_per_second": n * steps / elapsed,
        "max_periodic_error": float(np.max(periodic_max_error(points, anchor))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--explicit", action="store_true")
    args = parser.parse_args()
    point = [[1234567.89, 987654.321, 314159.265]]
    anchor = [[500.0, 500.0, 500.0]]
    result, history, steps = refine(point, anchor, tol=1e-12, max_iters=80)
    print(f"refined={result[0]} steps={steps} error={history[-1]:.3g}")
    if args.benchmark:
        print(benchmark(explicit=args.explicit))


if __name__ == "__main__":
    main()
