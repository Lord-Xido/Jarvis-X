from __future__ import annotations

import pytest

from jarvisx.inward_swarm_fixed_point import (
    InwardSwarmConfig,
    InwardSwarmFixedPointEngine,
    clipped_contract_axis,
    hamming_distance_word,
    refine_word_toward_target,
)


def _words(value: int, count: int = 64) -> tuple[int, ...]:
    return (value,) * count


def test_clipped_spatial_contraction_lands_on_center_without_oscillation():
    assert clipped_contract_axis(500_123, 500_000, 1_000) == 500_000
    assert clipped_contract_axis(499_123, 500_000, 1_000) == 500_000
    assert clipped_contract_axis(750_000, 500_000, 1_000) == 749_000


def test_bit_refinement_reduces_hamming_error_by_exactly_one():
    current = 0b111101
    target = 0b001010

    before = hamming_distance_word(current, target)
    refined = refine_word_toward_target(current, target)
    after = hamming_distance_word(refined, target)

    assert after == before - 1


def test_bit_refinement_is_idempotent_at_fixed_point():
    target = 0x1234567
    assert refine_word_toward_target(target, target) == target


def test_engine_joint_objective_contracts_strictly_until_fixed_point():
    config = InwardSwarmConfig(max_iterations=512)
    engine = InwardSwarmFixedPointEngine(config)
    engine.load(
        positions=[
            (0, 999_999, 500_123),
            (999_999, 1, 499_123),
        ],
        states=[_words(0x7FFFFFFF), _words(0x13579BDF)],
        targets=[_words(0), _words(0x02468ACE)],
    )

    bound = engine.theoretical_iterations_remaining()
    reports = engine.run_until_fixed_point()

    assert len(reports) <= bound
    assert reports[-1].converged
    assert all(report.objective_after <= report.objective_before for report in reports)
    assert all(report.strict_contraction for report in reports)
    assert engine.positions() == ((500_000, 500_000, 500_000),) * 2
    assert engine.states() == engine.targets()
    assert engine.status()["converged"] is True


def test_fixed_point_step_is_idempotent_and_does_not_claim_strict_contraction():
    config = InwardSwarmConfig()
    engine = InwardSwarmFixedPointEngine(config)
    engine.load(
        positions=[config.center],
        states=[_words(0x42)],
        targets=[_words(0x42)],
    )

    report = engine.step()

    assert report.converged
    assert not report.strict_contraction
    assert report.objective_before == 0.0
    assert report.objective_after == 0.0


def test_logical_domain_is_sparse_metadata_not_dense_materialization():
    config = InwardSwarmConfig(scale=1_000_000)
    engine = InwardSwarmFixedPointEngine(config)
    engine.load(
        positions=[(123, 456, 789)],
        states=[_words(0)],
        targets=[_words(0)],
    )

    status = engine.status()

    assert status["logical_domain"] == "1000000^3"
    assert status["agents"] == 1
    assert status["materialization"] == "sparse-active-agents-only"


def test_validation_rejects_out_of_range_packed_words():
    engine = InwardSwarmFixedPointEngine(InwardSwarmConfig(word_bits=31))
    with pytest.raises(ValueError, match="packed word"):
        engine.load(
            positions=[(0, 0, 0)],
            states=[_words(1 << 31)],
            targets=[_words(0)],
        )
