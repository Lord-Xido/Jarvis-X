"""Minimal reproducible benchmark harness for the bounded OmegaFold resolver."""

from __future__ import annotations

import json
import platform
import statistics
import time
from typing import Dict, List

from jarvisx.omega_fold import FoldConfig, FoldProblem, resolve, verify_result


def run_benchmark(repeats: int = 1000) -> Dict[str, object]:
    """Measure one documented scalar fixed-point workload."""

    if repeats <= 0:
        raise ValueError("repeats must be positive")

    problem = FoldProblem(
        name="benchmark-scalar-contraction",
        initial_state=(0.0,),
        transition=lambda state: ((state[0] + 2.0) / 2.0,),
        residual=lambda state: abs(state[0] - 2.0),
    )
    config = FoldConfig(max_iterations=64, tolerance=1.0e-9)
    samples: List[int] = []

    for _ in range(repeats):
        start = time.perf_counter_ns()
        result = resolve(problem, config)
        samples.append(time.perf_counter_ns() - start)
        if not verify_result(problem, config, result):
            raise RuntimeError("benchmark produced an unverifiable result")

    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, int(len(ordered) * 0.95))
    return {
        "workload": problem.name,
        "repeats": repeats,
        "precision": {"tolerance": config.tolerance, "quantization_digits": 12},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "timing_ns": {
            "median": statistics.median(samples),
            "p95": ordered[p95_index],
            "minimum": min(samples),
        },
        "claim_boundary": "Measured timings apply only to this workload and environment.",
    }


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), indent=2, sort_keys=True))
