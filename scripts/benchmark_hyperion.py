"""Deterministic synthetic benchmark for the Hyperion audit kernel.

Usage:
    PYTHONPATH=src python scripts/benchmark_hyperion.py --events 10000 --seed 7
"""

from __future__ import annotations

import argparse
import json
import random
import time

from jarvisx.hyperion import HyperionEngine, Observation


def build_fixture(events: int, seed: int) -> tuple[list[Observation], set[str]]:
    randomizer = random.Random(seed)
    observations: list[Observation] = []
    anomalies: set[str] = set()
    value = 10_000.0
    for index in range(events):
        event_id = f"event-{index}"
        timestamp = index * 10
        value += randomizer.uniform(-2.0, 2.0)
        csv_value = value
        cpu_value = value
        label = "entity"
        if index > 30 and index % 257 == 0:
            csv_value += 500.0
            anomalies.add(event_id)
        if index > 30 and index % 389 == 0:
            cpu_value -= 700.0
            anomalies.add(event_id)
        if index > 30 and index % 521 == 0:
            label = None
            csv_value *= 1.4
            anomalies.add(event_id)
        observations.extend(
            [
                Observation(
                    "csv",
                    timestamp,
                    csv_value,
                    "amount",
                    "ZAR",
                    event_id,
                    1.0,
                    True,
                    label,
                ),
                Observation(
                    "cpu",
                    timestamp + 1,
                    cpu_value,
                    "amount",
                    "ZAR",
                    event_id,
                    1.0,
                    True,
                    label,
                ),
            ]
        )
    return observations, anomalies


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=7)
    arguments = parser.parse_args()
    if arguments.events < 1:
        raise SystemExit("--events must be positive")

    observations, known_anomalies = build_fixture(arguments.events, arguments.seed)
    engine = HyperionEngine()
    start = time.perf_counter()
    report = engine.audit(observations)
    elapsed = time.perf_counter() - start

    detected = {point.event_id for point in report.points if point.critical}
    true_positive = len(detected & known_anomalies)
    false_positive = len(detected - known_anomalies)
    false_negative = len(known_anomalies - detected)
    precision = true_positive / (true_positive + false_positive) if detected else 0.0
    recall = (
        true_positive / (true_positive + false_negative)
        if known_anomalies
        else 0.0
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    print(
        json.dumps(
            {
                "events": arguments.events,
                "observations": len(observations),
                "seed": arguments.seed,
                "elapsed_seconds": elapsed,
                "events_per_second": arguments.events / elapsed,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "critical_events": len(detected),
                "known_anomalies": len(known_anomalies),
                "ghs": report.geometric_health_score,
                "verified": report.verify(),
                "model_hash": report.model_hash,
                "configuration_hash": report.configuration_hash,
                "report_digest": report.report_digest,
            },
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
