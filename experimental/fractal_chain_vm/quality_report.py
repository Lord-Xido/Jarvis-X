#!/usr/bin/env python3
"""Validate the fixed-reference experiment and report evidence, not claims."""
import csv
import json
import math
import pathlib
import sys

def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: quality_report.py <input.csv> <output.json>")
    source, output = map(pathlib.Path, sys.argv[1:])
    with source.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 5:
        raise ValueError(f"expected five levels, got {len(rows)}")
    measurements = []
    for index, row in enumerate(rows, 1):
        if int(row["levels"]) != index:
            raise ValueError("missing/duplicate level or wrong order")
        floats = {key: float(row[key]) for key in (
            "reference_mse", "constant_0_5_baseline_mse",
            "mean_baseline_mse", "relative_to_0_5",
            "relative_to_mean", "internal_final_mse",
            "decoded_stddev", "psnr_db")}
        if not all(math.isfinite(value) for value in floats.values()):
            raise ValueError("nonfinite quality metric")
        err, mean = floats["reference_mse"], floats["mean_baseline_mse"]
        assert 0 <= err <= 1 and 0 < mean <= 1
        assert math.isclose(floats["relative_to_mean"], err / mean, rel_tol=1e-7)
        assert math.isclose(floats["psnr_db"], -10 * math.log10(err), rel_tol=1e-7, abs_tol=1e-7)
        assert math.isclose(floats["relative_to_0_5"],
                            err / floats["constant_0_5_baseline_mse"], rel_tol=1e-7)
        measurements.append({"level": index, **floats})
    errors = [m["reference_mse"] for m in measurements]
    best = min(range(len(errors)), key=errors.__getitem__)
    improvement = (errors[0] - errors[-1]) / errors[0]
    report = {
        "dataset": "synthetic 16^3 power-8 Mandelbulb, 12 fractal iterations, 4 permeation passes",
        "metric": "MSE to fixed level-0 source volume",
        "levels": measurements,
        "best_level": best + 1,
        "best_mse": errors[best],
        "monotone_nonincreasing": all(b <= a for a,b in zip(errors, errors[1:])),
        "final_vs_first_fractional_improvement": improvement,
        "beats_optimal_constant_baseline": any(e < measurements[0]["mean_baseline_mse"] for e in errors),
        "important_limitation": "No learned reconstruction and no reference-independent proof of compression or auto-optimisation.",
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("FIXED-REFERENCE EXPERIMENT (synthetic 3D field)")
    for m in measurements:
        print(f"LEVEL {m['level']}: MSE={m['reference_mse']:.9g},"
              f" mean-baseline-relative={m['relative_to_mean']:.6g},"
              f" decoded-std={m['decoded_stddev']:.6g},"
              f" PSNR={m['psnr_db']:.5g} dB")
    print(f"BEST LEVEL={best+1}, MONOTONE={report['monotone_nonincreasing']},"
          f" FINAL CHANGE={improvement*100:+.4f}%,"
          f" BEATS OPTIMAL CONSTANT={report['beats_optimal_constant_baseline']}")
    print("Diagnostic checks passed. Passing is not a quality-improvement claim.")

if __name__ == "__main__":
    main()
