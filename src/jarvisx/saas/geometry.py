"""Mathematical and geometric enterprise-state model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence, Tuple


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


@dataclass(frozen=True)
class EnterprisePoint3D:
    delivery: float
    finance: float
    governance: float
    health: float
    risk: float

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.delivery, self.finance, self.governance)


def engagement_point(features: Mapping[str, float]) -> EnterprisePoint3D:
    """Map operational features into the consultancy's 3D enterprise manifold."""

    progress = clamp(features.get("progress", 0.0))
    quality = clamp(features.get("quality", 0.8))
    schedule = clamp(features.get("schedule", progress))
    budget_util = clamp(features.get("budget_utilization", 0.0))
    collection = clamp(features.get("collection_ratio", 1.0))
    margin = clamp(features.get("margin_ratio", 0.5))
    stated_risk = clamp(features.get("risk", 0.0))
    governance_input = clamp(features.get("governance", 0.8))
    audit = clamp(features.get("audit_completeness", governance_input))

    delivery = clamp(0.45 * progress + 0.30 * quality + 0.25 * schedule)
    finance = clamp(0.40 * (1.0 - budget_util) + 0.35 * collection + 0.25 * margin)
    governance = clamp(
        0.50 * governance_input + 0.25 * audit + 0.25 * (1.0 - stated_risk)
    )
    epsilon = 1e-9
    health = math.exp(
        0.40 * math.log(max(delivery, epsilon))
        + 0.35 * math.log(max(finance, epsilon))
        + 0.25 * math.log(max(governance, epsilon))
    )
    risk = clamp(1.0 - health + 0.25 * stated_risk)
    return EnterprisePoint3D(delivery, finance, governance, health, risk)


def enterprise_centroid(
    weighted_points: Iterable[Tuple[EnterprisePoint3D, float]],
) -> dict:
    points = [(point, max(0.0, float(weight))) for point, weight in weighted_points]
    if not points:
        return {"centroid": (0.0, 0.0, 0.0), "health": 0.0, "dispersion": 0.0}
    total_weight = sum(weight for _, weight in points) or float(len(points))
    normalized = [(point, weight if weight > 0 else 1.0) for point, weight in points]
    total_weight = sum(weight for _, weight in normalized)
    centroid = tuple(
        sum(point.as_tuple()[axis] * weight for point, weight in normalized)
        / total_weight
        for axis in range(3)
    )
    dispersion = math.sqrt(
        sum(
            weight
            * sum((point.as_tuple()[axis] - centroid[axis]) ** 2 for axis in range(3))
            for point, weight in normalized
        )
        / total_weight
    )
    health = sum(point.health * weight for point, weight in normalized) / total_weight
    return {"centroid": centroid, "health": health, "dispersion": dispersion}


def neural_advisory_vector(
    point: EnterprisePoint3D, metrics: Sequence[float]
) -> Tuple[float, ...]:
    """Create the vector consumed by the existing 3D abstraction ANN core."""

    return (
        point.delivery,
        point.finance,
        point.governance,
        point.health,
        point.risk,
        *tuple(float(value) for value in metrics),
    )
