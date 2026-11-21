"""Public, isolated reference for a periodic 3D diffusion-memory system.

This numerical model is NOT a reversible byte codec or a private DM-PCS runtime.
Requires NumPy; execution is explicit, bounded and does not perform I/O.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Parameters:
    side: int = 16
    alpha: float = 0.99
    gamma: float = 0.05
    diffusion: float = 0.10

    def validate(self) -> None:
        if type(self.side) is not int or self.side < 2 or self.side > 64 or self.side & (self.side - 1):
            raise ValueError("side must be a power of two in [2, 64]")
        if not (0.0 < self.alpha < 1.0 and 0.0 < self.gamma < 1.0):
            raise ValueError("alpha and gamma must be in (0, 1)")
        if not (0.0 < self.diffusion <= 1.0 / 6.0):
            raise ValueError("diffusion must be in (0, 1/6]")


def _array(field: np.ndarray, p: Parameters) -> np.ndarray:
    arr = np.asarray(field, dtype=np.float64)
    if arr.shape != (p.side,) * 3 or not np.isfinite(arr).all():
        raise ValueError("expected finite 3D float field matching side")
    return arr


def laplacian(field: np.ndarray) -> np.ndarray:
    """Six-neighbour periodic Laplacian; its spatial sum is zero."""
    return sum((np.roll(field, s, axis=a) for a in range(3) for s in (-1, 1)), start=np.zeros_like(field)) - 6 * field


def rotate(field: np.ndarray) -> np.ndarray:
    """Push-forward torus symmetry (x,y,z)->(z,y,-x mod side), order four."""
    return np.roll(np.rot90(field, k=-1, axes=(0, 2)), 1, axis=2)


def inverse_rotate(field: np.ndarray) -> np.ndarray:
    return np.rot90(np.roll(field, -1, axis=2), k=1, axes=(0, 2))


def step(v: np.ndarray, m: np.ndarray, e: np.ndarray, p: Parameters = Parameters()):
    """Numerical evolution. Accumulated residual `e` is NOT physical energy."""
    p.validate()
    v, m, e = (_array(x, p) for x in (v, m, e))
    q = v + p.diffusion * laplacian(v)
    residual = p.alpha * (m - q)
    return (rotate(q + p.gamma * residual),
            rotate(p.alpha * m + (1 - p.alpha) * q),
            rotate(e + np.abs(residual)))


def conserved_charge(v: np.ndarray, m: np.ndarray, p: Parameters = Parameters()) -> float:
    p.validate()
    v, m = _array(v, p), _array(m, p)
    return (1 - p.alpha) * float(v.sum()) + p.gamma * p.alpha * float(m.sum())


def direct_limit(v: np.ndarray, m: np.ndarray, p: Parameters = Parameters()):
    """Asymptotic uniform v,m only; excludes finite-time values and residual e."""
    p.validate()
    v, m = _array(v, p), _array(m, p)
    c = ((1 - p.alpha) * v.mean() + p.gamma * p.alpha * m.mean()) / ((1 - p.alpha) + p.gamma * p.alpha)
    return np.full(v.shape, c), np.full(m.shape, c)


def _product(a, b):
    return (a[0]*b[0]+a[1]*b[2], a[0]*b[1]+a[1]*b[3],
            a[2]*b[0]+a[3]*b[2], a[2]*b[1]+a[3]*b[3])


def finite_value_memory_fft(v: np.ndarray, m: np.ndarray, steps: int, p: Parameters = Parameters()):
    """Finite-time value/memory with Fourier 2x2 matrix exponentiation; excludes e."""
    p.validate()
    v, m = _array(v, p), _array(m, p)
    if type(steps) is not int or not 0 <= steps <= 1_000_000:
        raise ValueError("steps out of bounds")
    k = 2 * np.pi * np.fft.fftfreq(p.side)
    beta = 1 + 2 * p.diffusion * (np.cos(k)[:,None,None] + np.cos(k)[None,:,None] + np.cos(k)[None,None,:] - 3)
    one = np.ones_like(beta, dtype=np.complex128)
    zero = np.zeros_like(beta, dtype=np.complex128)
    result = (one, zero, zero, one)
    power = ((1 - p.gamma*p.alpha)*beta, p.gamma*p.alpha*one,
             (1 - p.alpha)*beta, p.alpha*one)
    remaining = steps
    while remaining:
        if remaining & 1:
            result = _product(result, power)
        remaining >>= 1
        if remaining:
            power = _product(power, power)
    vh, mh = np.fft.fftn(v), np.fft.fftn(m)
    vo = np.fft.ifftn(result[0]*vh + result[1]*mh).real
    mo = np.fft.ifftn(result[2]*vh + result[3]*mh).real
    for _ in range(steps % 4):
        vo, mo = rotate(vo), rotate(mo)
    return vo, mo


def morton3(x: int, y: int, z: int, bits: int = 4) -> int:
    if not 1 <= bits <= 6 or any(type(a) is not int or not 0 <= a < (1 << bits) for a in (x, y, z)):
        raise ValueError("invalid Morton coordinate")
    return sum(((x >> i) & 1) << (3*i) | ((y >> i) & 1) << (3*i+1) | ((z >> i) & 1) << (3*i+2) for i in range(bits))
