"""Bounded 3D latency-field routing for Jarvis-X inference placement.

The controller maps every candidate endpoint into a three-axis operational field:

    X: network geometry (causality floor + non-propagation network overhead)
    Y: compute geometry (prefill + decode work for the request objective)
    Z: service-load geometry (queueing delay)

It then minimizes a scalar potential over those axes while preserving two important
constraints:

1. measured network telemetry is never allowed to violate the propagation floor;
2. hysteresis prevents small score changes from causing route flapping.

The module is deterministic and dependency-free. It does not move models or start
infrastructure by itself; it selects among endpoint states supplied by the caller.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable

FIBER_SPEED_MPS = 2.0e8


def _finite_non_negative(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _positive(name: str, value: float) -> float:
    result = _finite_non_negative(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be greater than zero")
    return result


@dataclass(frozen=True)
class RequestProfile:
    """Request shape used by the latency predictor."""

    input_tokens: int
    output_tokens: int = 1
    objective: str = "ttft"
    render_ms: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.input_tokens, bool) or not isinstance(self.input_tokens, int):
            raise TypeError("input_tokens must be an integer")
        if self.input_tokens < 0:
            raise ValueError("input_tokens must be non-negative")
        if isinstance(self.output_tokens, bool) or not isinstance(self.output_tokens, int):
            raise TypeError("output_tokens must be an integer")
        if self.output_tokens <= 0:
            raise ValueError("output_tokens must be positive")
        if self.objective not in {"ttft", "completion"}:
            raise ValueError("objective must be 'ttft' or 'completion'")
        _finite_non_negative("render_ms", self.render_ms)


@dataclass(frozen=True)
class EndpointState3D:
    """Measured or estimated state of one inference endpoint."""

    name: str
    one_way_distance_km: float
    network_overhead_ms: float
    queue_ms: float
    prefill_tokens_per_s: float
    decode_tokens_per_s: float
    measured_rtt_ms: float | None = None
    healthy: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty")
        _finite_non_negative("one_way_distance_km", self.one_way_distance_km)
        _finite_non_negative("network_overhead_ms", self.network_overhead_ms)
        _finite_non_negative("queue_ms", self.queue_ms)
        _positive("prefill_tokens_per_s", self.prefill_tokens_per_s)
        _positive("decode_tokens_per_s", self.decode_tokens_per_s)
        if self.measured_rtt_ms is not None:
            _finite_non_negative("measured_rtt_ms", self.measured_rtt_ms)


@dataclass(frozen=True)
class LatencyWeights:
    """Dimensionless weights for the 3D latency potential."""

    network: float = 1.0
    compute: float = 1.0
    queue: float = 1.0

    def __post_init__(self) -> None:
        for name in ("network", "compute", "queue"):
            _finite_non_negative(name, getattr(self, name))
        if self.network + self.compute + self.queue <= 0.0:
            raise ValueError("at least one latency weight must be positive")


@dataclass(frozen=True)
class LatencyEstimate:
    causality_floor_ms: float
    network_ms: float
    queue_ms: float
    prefill_ms: float
    first_decode_ms: float
    decode_tail_ms: float
    render_ms: float
    total_ms: float

    @property
    def compute_ms(self) -> float:
        return self.prefill_ms + self.first_decode_ms + self.decode_tail_ms

    @property
    def point3d(self) -> tuple[float, float, float]:
        return (self.network_ms, self.compute_ms, self.queue_ms)

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["compute_ms"] = self.compute_ms
        payload["point3d"] = self.point3d
        return payload


@dataclass(frozen=True)
class RouteDecision:
    endpoint: EndpointState3D
    estimate: LatencyEstimate
    residual_bias_ms: float
    potential_ms: float
    kept_for_hysteresis: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "endpoint": asdict(self.endpoint),
            "estimate": self.estimate.as_dict(),
            "residual_bias_ms": self.residual_bias_ms,
            "potential_ms": self.potential_ms,
            "kept_for_hysteresis": self.kept_for_hysteresis,
        }


class LatencyField3DOptimizer:
    """Select the lowest-potential endpoint in a bounded 3D latency field."""

    def __init__(
        self,
        endpoints: Iterable[EndpointState3D],
        *,
        weights: LatencyWeights | None = None,
        fiber_speed_mps: float = FIBER_SPEED_MPS,
        hysteresis_ms: float = 2.0,
        feedback_alpha: float = 0.20,
        max_residual_bias_ms: float = 250.0,
    ) -> None:
        endpoint_tuple = tuple(endpoints)
        if not endpoint_tuple:
            raise ValueError("at least one endpoint is required")
        names = [endpoint.name for endpoint in endpoint_tuple]
        if len(set(names)) != len(names):
            raise ValueError("endpoint names must be unique")

        self.endpoints = endpoint_tuple
        self.weights = weights or LatencyWeights()
        self.fiber_speed_mps = _positive("fiber_speed_mps", fiber_speed_mps)
        self.hysteresis_ms = _finite_non_negative("hysteresis_ms", hysteresis_ms)
        alpha = _finite_non_negative("feedback_alpha", feedback_alpha)
        if alpha > 1.0:
            raise ValueError("feedback_alpha must be in [0, 1]")
        self.feedback_alpha = alpha
        self.max_residual_bias_ms = _finite_non_negative(
            "max_residual_bias_ms", max_residual_bias_ms
        )
        self._residual_bias_ms = {name: 0.0 for name in names}

    def estimate(
        self,
        endpoint: EndpointState3D,
        request: RequestProfile,
    ) -> LatencyEstimate:
        distance_m = endpoint.one_way_distance_km * 1_000.0
        floor_ms = 2.0 * distance_m / self.fiber_speed_mps * 1_000.0

        if endpoint.measured_rtt_ms is None:
            transport_ms = floor_ms
        else:
            transport_ms = max(floor_ms, endpoint.measured_rtt_ms)
        network_ms = transport_ms + endpoint.network_overhead_ms

        prefill_ms = request.input_tokens / endpoint.prefill_tokens_per_s * 1_000.0
        first_decode_ms = 1_000.0 / endpoint.decode_tokens_per_s
        decode_tail_ms = 0.0
        if request.objective == "completion" and request.output_tokens > 1:
            decode_tail_ms = (
                (request.output_tokens - 1) / endpoint.decode_tokens_per_s * 1_000.0
            )

        total_ms = (
            network_ms
            + endpoint.queue_ms
            + prefill_ms
            + first_decode_ms
            + decode_tail_ms
            + request.render_ms
        )
        return LatencyEstimate(
            causality_floor_ms=floor_ms,
            network_ms=network_ms,
            queue_ms=endpoint.queue_ms,
            prefill_ms=prefill_ms,
            first_decode_ms=first_decode_ms,
            decode_tail_ms=decode_tail_ms,
            render_ms=request.render_ms,
            total_ms=total_ms,
        )

    def rank(self, request: RequestProfile) -> tuple[RouteDecision, ...]:
        decisions: list[RouteDecision] = []
        for endpoint in self.endpoints:
            if not endpoint.healthy:
                continue
            estimate = self.estimate(endpoint, request)
            weighted = (
                self.weights.network * estimate.network_ms
                + self.weights.compute * estimate.compute_ms
                + self.weights.queue * estimate.queue_ms
            )
            bias = self._residual_bias_ms[endpoint.name]
            decisions.append(
                RouteDecision(
                    endpoint=endpoint,
                    estimate=estimate,
                    residual_bias_ms=bias,
                    potential_ms=weighted + bias,
                )
            )

        if not decisions:
            raise RuntimeError("no healthy inference endpoints are available")
        return tuple(sorted(decisions, key=lambda item: (item.potential_ms, item.endpoint.name)))

    def select(
        self,
        request: RequestProfile,
        *,
        incumbent: str | None = None,
    ) -> RouteDecision:
        ranked = self.rank(request)
        best = ranked[0]
        if incumbent is None or incumbent == best.endpoint.name:
            return best

        current = next((item for item in ranked if item.endpoint.name == incumbent), None)
        if current is None:
            return best

        improvement = current.potential_ms - best.potential_ms
        if improvement <= self.hysteresis_ms:
            return RouteDecision(
                endpoint=current.endpoint,
                estimate=current.estimate,
                residual_bias_ms=current.residual_bias_ms,
                potential_ms=current.potential_ms,
                kept_for_hysteresis=True,
            )
        return best

    def observe(
        self,
        decision: RouteDecision,
        observed_ms: float,
    ) -> float:
        """Update an EWMA residual for the selected endpoint and return the new bias."""

        actual = _finite_non_negative("observed_ms", observed_ms)
        predicted = decision.estimate.total_ms
        residual = actual - predicted
        old = self._residual_bias_ms[decision.endpoint.name]
        updated = (1.0 - self.feedback_alpha) * old + self.feedback_alpha * residual
        limit = self.max_residual_bias_ms
        updated = max(-limit, min(limit, updated))
        self._residual_bias_ms[decision.endpoint.name] = updated
        return updated

    def feedback_state(self) -> dict[str, float]:
        return dict(self._residual_bias_ms)
