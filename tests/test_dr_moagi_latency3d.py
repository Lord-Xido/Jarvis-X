from __future__ import annotations

import pytest

from jarvisx.dr_moagi_latency3d import EndpointState3D, LatencyField3DOptimizer, RequestProfile


def _endpoint(name: str, distance: float, decode: float, *, overhead: float = 0.0):
    return EndpointState3D(
        name=name,
        one_way_distance_km=distance,
        network_overhead_ms=overhead,
        queue_ms=1.0,
        prefill_tokens_per_s=10_000.0,
        decode_tokens_per_s=decode,
    )


def test_causality_floor_is_enforced():
    endpoint = EndpointState3D(
        name="remote",
        one_way_distance_km=5_000.0,
        network_overhead_ms=3.0,
        queue_ms=0.0,
        prefill_tokens_per_s=20_000.0,
        decode_tokens_per_s=100.0,
        measured_rtt_ms=1.0,
    )
    optimizer = LatencyField3DOptimizer([endpoint])
    estimate = optimizer.estimate(endpoint, RequestProfile(input_tokens=0))
    assert estimate.causality_floor_ms == pytest.approx(50.0)
    assert estimate.network_ms == pytest.approx(53.0)


def test_ttft_prefers_near_edge_for_short_request():
    edge = _endpoint("edge", 100.0, 20.0, overhead=2.0)
    remote = _endpoint("remote", 5_000.0, 200.0, overhead=2.0)
    optimizer = LatencyField3DOptimizer([edge, remote], hysteresis_ms=0.0)
    decision = optimizer.select(RequestProfile(input_tokens=20, objective="ttft"))
    assert decision.endpoint.name == "edge"
    assert decision.estimate.point3d == pytest.approx((3.0, 52.0, 1.0))


def test_completion_can_prefer_remote_decode_throughput():
    edge = _endpoint("edge", 100.0, 20.0, overhead=2.0)
    remote = _endpoint("remote", 5_000.0, 200.0, overhead=2.0)
    optimizer = LatencyField3DOptimizer([edge, remote], hysteresis_ms=0.0)
    request = RequestProfile(input_tokens=20, output_tokens=1_000, objective="completion")
    assert optimizer.select(request).endpoint.name == "remote"


def test_feedback_residual_changes_ranking():
    first = _endpoint("first", 50.0, 100.0)
    second = _endpoint("second", 100.0, 100.0, overhead=1.0)
    optimizer = LatencyField3DOptimizer(
        [first, second], hysteresis_ms=0.0, feedback_alpha=1.0, max_residual_bias_ms=100.0
    )
    request = RequestProfile(input_tokens=1)
    decision = optimizer.select(request)
    assert decision.endpoint.name == "first"
    optimizer.observe(decision, decision.estimate.total_ms + 20.0)
    assert optimizer.select(request).endpoint.name == "second"
