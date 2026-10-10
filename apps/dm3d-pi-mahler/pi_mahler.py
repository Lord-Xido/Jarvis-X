#!/usr/bin/env python3
"""DM3D–π: mathematically faithful pi-periodic three-torus Mahler cubature.

Exact Jensen elimination for P=A(x,y)+z B(x,y) (not an original theorem).
Bounded, sampled cubature + adaptive resolution. No dense septillion^3 array.
"""
from __future__ import annotations
import argparse
import json
import math
import time
from dataclasses import dataclass
from typing import Sequence
import numpy as np

SIDE = 10**24
VIRTUAL_VOXELS = SIDE**3


@dataclass(frozen=True)
class Term:
    coeff: complex
    exponents: tuple[int, int, int]


@dataclass(frozen=True)
class Laurent3:
    terms: tuple[Term, ...]
    radii: tuple[float, float, float] = (1., 1., 1.)

    def __post_init__(self):
        if not self.terms:
            raise ValueError("at least one term required")
        if not all(math.isfinite(v) and v > 0 for v in self.radii):
            raise ValueError("radii must be finite positive numbers")
        for term in self.terms:
            if len(term.exponents) != 3 or not all(isinstance(k, int) for k in term.exponents):
                raise ValueError("exponents must be integer triples")
            if not math.isfinite(term.coeff.real) or not math.isfinite(term.coeff.imag):
                raise ValueError("coefficients must be finite")

    def evaluate(self, u, v, w):
        total = 0j
        for term in self.terms:
            e0, e1, e2 = term.exponents
            total = total + term.coeff * (self.radii[0] ** e0) * (self.radii[1] ** e1) * (self.radii[2] ** e2) * np.exp(2j * (e0 * u + e1 * v + e2 * w))
        return total

    def jensen_parts(self, u, v):
        if any(t.exponents[2] not in (0, 1) for t in self.terms):
            raise ValueError("Jensen reducer requires only z^0 and z^1")
        a = 0j
        b = 0j
        for term in self.terms:
            p, q, k = term.exponents
            piece = term.coeff * (self.radii[0] ** p) * (self.radii[1] ** q) * np.exp(2j * (p * u + q * v))
            if k == 0:
                a = a + piece
            else:
                b = b + piece * self.radii[2]
        return a, b


def pi_phase_integer(k: int, digits: int = 90) -> dict:
    """Arbitrary-precision virtual coordinate, preserving 10^-24 phase differences."""
    if not isinstance(k, int) or not 0 <= k < SIDE:
        raise ValueError("coordinate out of range")
    import mpmath as mp
    with mp.workdps(max(50, digits)):
        frac = mp.mpf(k) / SIDE
        angle = mp.pi * frac
        return {"coordinate": str(k), "fraction_of_pi": mp.nstr(frac, digits),
                "angle_radians": mp.nstr(angle, digits),
                "phase_real": mp.nstr(mp.cos(2 * angle), digits),
                "phase_imag": mp.nstr(mp.sin(2 * angle), digits)}


def midpoint_samples(n: int) -> np.ndarray:
    if not isinstance(n, int) or n < 2 or n > 1024:
        raise ValueError("n must be 2..1024")
    return (np.arange(n, dtype=float) + .5) * (math.pi / n)


def direct_3d(P: Laurent3, n: int) -> dict:
    if n > 160:
        raise ValueError("full reference limited to n<=160")
    u = midpoint_samples(n)
    x = u[:, None]
    y = u[None, :]
    total = 0.
    evals = 0
    t0 = time.perf_counter()
    with np.errstate(divide="ignore", invalid="ignore"):
        for z in u:
            amplitude = np.abs(P.evaluate(x, y, z))
            if np.any(amplitude == 0):
                raise ValueError("sample hits zero: nonregular integrand, shift grid")
            total += float(np.log(amplitude).sum())
            evals += n * n
    return {"value": total / (n**3), "evaluations": evals,
            "seconds": time.perf_counter() - t0, "n": n, "dimensions": 3}


def jensen_2d(P: Laurent3, n: int) -> dict:
    u = midpoint_samples(n)
    t0 = time.perf_counter()
    with np.errstate(divide="ignore", invalid="ignore"):
        A, B = P.jensen_parts(u[:, None], u[None, :])
        amp = np.maximum(np.abs(A), np.abs(B))
        if np.any(amp == 0):
            raise ValueError("sample hits simultaneous zeros")
        val = float(np.log(amp).mean())
    return {"value": val, "evaluations": n * n,
            "seconds": time.perf_counter() - t0, "n": n, "dimensions": 2}


def resolution_feedback(P: Laurent3, tolerance: float = 1e-6,
                        levels: Sequence[int] = (8, 16, 32, 64, 128, 256)) -> dict:
    """Stop based on observed differences only, not a proof of integration error."""
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if not levels or any(not isinstance(n, int) or n < 2 or n > 1024 for n in levels):
        raise ValueError("levels must contain valid grid sizes")
    prev = None
    history = []
    for n in levels:
        item = jensen_2d(P, n)
        item["delta_from_prior"] = None if prev is None else abs(item["value"] - prev)
        history.append(item)
        if prev is not None and item["delta_from_prior"] < tolerance:
            return {"estimate": item["value"], "converged_heuristically": True,
                    "history": history, "evaluations_total": sum(r["evaluations"] for r in history)}
        prev = item["value"]
    return {"estimate": history[-1]["value"], "converged_heuristically": False,
            "history": history, "evaluations_total": sum(r["evaluations"] for r in history)}


def test_polynomials() -> dict:
    return {
        "smyth": Laurent3((Term(1, (0, 0, 0)), Term(1, (1, 0, 0)),
                             Term(1, (0, 1, 0)), Term(1, (0, 0, 1)))),
        "dominant": Laurent3((Term(1.85, (0, 0, 0)), Term(1, (1, 0, 0)),
                                Term(.5, (0, 1, 0)), Term(.2, (0, 0, 1)))),
        "cross_terms": Laurent3((Term(1, (0, 0, 0)), Term(1, (1, 0, 0)),
                                   Term(1, (0, 1, 0)), Term(.4, (1, 1, 1)),
                                   Term(1, (0, 0, 1)))),
        "radii": Laurent3((Term(1, (0, 0, 0)), Term(.9, (1, 0, 0)),
                             Term(.8, (0, 1, 0)), Term(1.1, (0, 0, 1))),
                            radii=(1.1, .8, .9)),
    }


def smyth_reference():
    import mpmath as mp
    with mp.workdps(60):
        return float(7 * mp.zeta(3) / (2 * mp.pi**2))


def demo(n: int = 32, tolerance: float = 1e-6) -> dict:
    out = {"virtual_axis": str(SIDE), "virtual_sites": str(VIRTUAL_VOXELS),
           "pi_periodic": True, "smyth_reference": smyth_reference(), "cases": {}}
    for name, P in test_polynomials().items():
        baseline = direct_3d(P, n)
        reduced = jensen_2d(P, n)
        loop = resolution_feedback(P, tolerance)
        expected = (math.log(1.85) if name == "dominant"
                    else smyth_reference() if name == "smyth" else None)
        out["cases"][name] = {
            "direct3d": baseline, "jensen2d": reduced, "feedback": loop,
            "known_exact": expected,
            "jensen_error": None if expected is None else abs(reduced["value"] - expected),
            "direct_error": None if expected is None else abs(baseline["value"] - expected)}
    out["virtual_far_corner"] = pi_phase_integer(SIDE - 1, 35)
    return out


def main():
    p = argparse.ArgumentParser(description="DM3D pi-Mahler analytical dimension reduction emulator")
    p.add_argument("command", nargs="?", choices=["demo", "phase"], default="demo")
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--coordinate", type=int, default=SIDE - 1)
    p.add_argument("--out", type=str)
    args = p.parse_args()
    output = pi_phase_integer(args.coordinate) if args.command == "phase" else demo(args.n, args.tol)
    payload = json.dumps(output, indent=2, allow_nan=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
