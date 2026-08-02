"""Minimal deterministic Hyperion audit demonstration."""

from jarvisx.hyperion import HyperionConfig, HyperionEngine, Observation


def observation(
    source: str,
    timestamp_ms: int,
    value: float,
    event_id: str,
) -> Observation:
    return Observation(
        source=source,
        timestamp_ms=timestamp_ms,
        value=value,
        quantity="amount",
        unit="ZAR",
        correlation_id=event_id,
        confidence=1.0,
        label="verified beneficiary",
    )


observations = []
for index, balance in enumerate((0.0, -5_000.0, -10_000.0, -15_000.0, -99_500.0)):
    event_id = f"tx-{index}"
    timestamp = index * 1_000
    observations.extend(
        [
            observation("csv", timestamp, balance, event_id),
            observation("cpu", timestamp + 1, balance, event_id),
        ]
    )

engine = HyperionEngine(HyperionConfig(lower_bound=-100_000.0))
report = engine.audit(observations)

print(f"events={len(report.points)}")
print(f"ghs={report.geometric_health_score:.3f}")
print(f"verified={report.verify()}")
print(f"model={report.model_hash}")
print(f"report={report.report_digest}")
for point in report.points:
    if any(point.flags.values()):
        print(point.event_id, point.cas, dict(point.flags))
