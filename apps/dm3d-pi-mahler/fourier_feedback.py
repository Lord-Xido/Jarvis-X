#!/usr/bin/env python3
"""DM3D–π: independently corrected Fourier control-variate experiment.

The identity E[f] = E[g] + E[f-g] is exact. The Monte Carlo estimator
is unbiased conditional on an independently fitted model under integrability.
"""
from __future__ import annotations
import argparse
import json
import math
import time
import numpy as np
from pi_mahler import Laurent3, smyth_reference, test_polynomials


def sample_f(P: Laurent3, angles: np.ndarray) -> np.ndarray:
    a, b = P.jensen_parts(angles[:, 0], angles[:, 1])
    return np.log(np.maximum(np.abs(a), np.abs(b)))


def modes(order: int) -> list[tuple[int, int]]:
    if order < 0 or order > 8:
        raise ValueError("order 0..8")
    return [(i, j) for i in range(-order, order + 1)
            for j in range(-order, order + 1)
            if i > 0 or (i == 0 and j > 0)]


def features(samples: np.ndarray, order: int) -> np.ndarray:
    out = [np.ones(len(samples))]
    for i, j in modes(order):
        phase = 2 * (i * samples[:, 0] + j * samples[:, 1])
        out.extend([np.cos(phase), np.sin(phase)])
    return np.stack(out, axis=1)


def fit_on_pilot(P: Laurent3, angles: np.ndarray,
                 orders=(0, 1, 2, 3), val: np.ndarray | None = None):
    if val is None:
        raise ValueError("independent validation required")
    f = sample_f(P, angles)
    fval = sample_f(P, val)
    best = None
    records = []
    for order in orders:
        X = features(angles, order)
        if X.shape[1] >= len(angles):
            continue
        coeff = np.linalg.lstsq(X, f, rcond=1e-10)[0]
        residual = fval - features(val, order) @ coeff
        variance = float(residual.var(ddof=1))
        records.append({"order": order, "validation_residual_variance": variance,
                        "features": X.shape[1]})
        if best is None or variance < best[0] - 1e-12:
            best = (variance, order, coeff)
    if best is None:
        raise ValueError("no feasible Fourier order")
    return best[1], best[2], records


def emulate(P: Laurent3, seed: int = 42, pilot_n: int = 384,
            validation_n: int = 128, test_n: int = 2048):
    if pilot_n < 60 or validation_n < 2 or test_n < 2:
        raise ValueError("sample budget invalid")
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    pilot = rng.random((pilot_n, 2)) * math.pi
    val = rng.random((validation_n, 2)) * math.pi
    test = rng.random((test_n, 2)) * math.pi
    order, coeff, records = fit_on_pilot(P, pilot, val=val)
    truth = sample_f(P, test)
    residual = truth - features(test, order) @ coeff
    corrected = float(coeff[0] + residual.mean())
    se_corrected = float(residual.std(ddof=1) / math.sqrt(test_n))
    # Equal total source-integrand evaluation budget, including fit & selection.
    total = pilot_n + validation_n + test_n
    independent = rng.random((total, 2)) * math.pi
    naive = sample_f(P, independent)
    plain = float(naive.mean())
    se_plain = float(naive.std(ddof=1) / math.sqrt(total))
    elapsed = time.perf_counter() - t0
    return {
        "estimate_feedback": corrected, "estimate_plain": plain,
        "se_feedback": se_corrected, "se_plain": se_plain,
        "variance_efficiency_gain": (se_plain / se_corrected) ** 2,
        "Fourier_order": order, "Fourier_features": len(coeff),
        "control_integral": float(coeff[0]),
        "evaluations_feedback": total, "evaluations_plain": total,
        "pilot": pilot_n, "validation": validation_n, "test": test_n,
        "elapsed_seconds_including_fit_and_both_estimators": elapsed,
        "selection_ledger": records, "independent_holdout": True,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="smyth", choices=list(test_polynomials()))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out")
    a = p.parse_args()
    result = emulate(test_polynomials()[a.case], a.seed)
    result["polynomial_case"] = a.case
    exact = (smyth_reference() if a.case == "smyth"
             else math.log(1.85) if a.case == "dominant" else None)
    result["known_exact"] = exact
    if exact is not None:
        result["observed_feedback_error"] = abs(result["estimate_feedback"] - exact)
        result["observed_plain_error"] = abs(result["estimate_plain"] - exact)
    s = json.dumps(result, indent=2)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(s + "\n")
    print(s)


if __name__ == "__main__":
    main()
