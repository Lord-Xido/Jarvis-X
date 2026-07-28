from dataclasses import replace

import pytest

from jarvisx.meta_volume import (
    FrameSignals,
    HardwareTelemetry,
    MetaVolumeConfig,
    ParetoPoint,
    SelfEvolutionaryMetaVolume,
    pareto_front,
)


def make_signals():
    return FrameSignals(
        error=(0.90, 0.10, 0.80, 0.05),
        edge_density=(0.80, 0.10, 0.70, 0.00),
        occupancy=(0.90, 0.00, 0.80, 0.00),
    )


def make_engine(**overrides):
    values = dict(
        region_count=4,
        parameter_count=8,
        learning_rate=0.50,
        lambda_compute=0.02,
        lambda_memory=0.01,
        minimum_active_ratio=0.25,
    )
    values.update(overrides)
    return SelfEvolutionaryMetaVolume(MetaVolumeConfig(**values))


def test_layer_manifest_exposes_twelve_logical_layers():
    engine = make_engine()
    manifest = engine.state.layer_manifest()
    assert len(manifest) == 12
    assert manifest[8]["name"] == "architecture_dna"
    assert manifest[9]["name"] == "step_allocation_map"
    assert manifest[10]["name"] == "sparsity_mask"
    assert manifest[11]["name"] == "meta_gradient_accumulator"


def test_complex_regions_receive_more_depth_and_samples():
    engine = make_engine()
    result = engine.evolve(
        make_signals(),
        HardwareTelemetry(frame_ms=12.0, flops=1.0e11, memory_mb=2048.0),
    )
    assert result.committed
    assert engine.state.architecture.depth[0] > engine.state.architecture.depth[1]
    assert engine.state.step_map[0] > engine.state.step_map[1]
    assert engine.state.architecture.depth[2] > engine.state.architecture.depth[3]
    assert engine.state.step_map[2] > engine.state.step_map[3]
    assert any(item.opcode == "SET_DEPTH" for item in result.instructions)
    assert any(item.opcode == "SET_STEPS" for item in result.instructions)


def test_high_hardware_pressure_reduces_allocations():
    low_pressure = make_engine()
    high_pressure = make_engine(lambda_compute=0.50, lambda_memory=0.50)
    signals = make_signals()

    low = low_pressure.evolve(
        signals,
        HardwareTelemetry(frame_ms=12.0, flops=1.0e11, memory_mb=2048.0),
    )
    high = high_pressure.evolve(
        signals,
        HardwareTelemetry(
            frame_ms=40.0,
            flops=1.0e12,
            memory_mb=23000.0,
            sm_cycles=1.0e9,
        ),
    )

    assert low.committed and high.committed
    assert sum(high.state.architecture.depth) <= sum(low.state.architecture.depth)
    assert sum(high.state.step_map) < sum(low.state.step_map)


def test_empty_scene_prunes_to_declared_floor():
    engine = make_engine(
        learning_rate=0.60,
        lambda_compute=0.20,
        lambda_memory=0.20,
    )
    empty = FrameSignals(
        error=(0.0, 0.0, 0.0, 0.0),
        edge_density=(0.0, 0.0, 0.0, 0.0),
        occupancy=(0.0, 0.0, 0.0, 0.0),
    )
    telemetry = HardwareTelemetry(
        frame_ms=40.0,
        flops=2.0e12,
        memory_mb=23000.0,
        sm_cycles=1.0e9,
    )

    result = engine.evolve(empty, telemetry)
    assert result.committed
    assert engine.snapshot()["active_ratio"] == pytest.approx(0.25)
    assert set(engine.state.architecture.depth) == {engine.config.min_depth}


def test_invalid_candidate_rolls_back_atomically(monkeypatch):
    engine = make_engine()
    before = engine.state
    original = engine._propose

    def invalid(gradients):
        candidate = original(gradients)
        architecture = replace(
            candidate.architecture,
            depth=(99,) + candidate.architecture.depth[1:],
        )
        return replace(candidate, architecture=architecture)

    monkeypatch.setattr(engine, "_propose", invalid)
    result = engine.evolve(
        make_signals(),
        HardwareTelemetry(frame_ms=12.0, flops=1.0e11, memory_mb=2048.0),
    )

    assert not result.committed
    assert engine.state.cycle == before.cycle
    assert engine.state.journal_hash == before.journal_hash
    assert engine.state.architecture == before.architecture
    assert engine.state.rollback_reason == "depth bound exceeded"


def test_replay_is_deterministic():
    left = make_engine()
    right = make_engine()
    signals = make_signals()
    telemetry = HardwareTelemetry(
        frame_ms=12.0,
        flops=1.0e11,
        memory_mb=2048.0,
    )

    for _ in range(3):
        assert left.evolve(signals, telemetry) == right.evolve(signals, telemetry)

    assert left.state == right.state
    assert left.state.journal_hash == right.state.journal_hash


def test_pareto_front_removes_dominated_candidates():
    points = (
        ParetoPoint("fixed-small", 29.0, 100.0, 10.0),
        ParetoPoint("dominated", 28.0, 120.0, 12.0),
        ParetoPoint("quality", 31.0, 150.0, 8.0),
    )
    front = pareto_front(points)
    assert {point.name for point in front} == {"fixed-small", "quality"}
