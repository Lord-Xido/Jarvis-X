import math

import pytest

from jarvisx.inward_document_swarm import (
    DEFAULT_CAPACITY,
    DEFAULT_SIDE,
    Inward3DDocumentSwarm,
    SwarmState,
)


def test_default_capacity_is_exact_cube() -> None:
    assert DEFAULT_SIDE == 4_843
    assert DEFAULT_CAPACITY == 113_590_865_107
    assert DEFAULT_CAPACITY == DEFAULT_SIDE**3


def test_3d_address_round_trip() -> None:
    swarm = Inward3DDocumentSwarm()
    probes = [0, 1, 1_299_999, swarm.logical_capacity - 1]
    for index in probes:
        assert swarm.xyz_to_linear(*swarm.linear_to_xyz(index)) == index


def test_candidate_cube_contains_125_directions() -> None:
    swarm = Inward3DDocumentSwarm()
    directions = tuple(swarm.candidate_directions(radius=2))
    assert len(directions) == 125
    assert (-2, -2, -2) in directions
    assert (0, 0, 0) in directions
    assert (2, 2, 2) in directions


def test_million_word_schedule_is_bounded_and_virtual() -> None:
    swarm = Inward3DDocumentSwarm(document_words=1_000_000)
    summary = swarm.schedule_summary(SwarmState())

    assert summary["document_tokens"] == 1_300_000
    assert summary["sections"] == 1_000
    assert summary["logical_ticks"] == 1
    assert math.isclose(summary["virtual_ns"], 1.3)
    assert summary["effective_ops"] < summary["logical_capacity"]


def test_recursive_optimisation_does_not_reduce_fitness() -> None:
    swarm = Inward3DDocumentSwarm(seed=7)
    result = swarm.optimise(generations=24)

    assert len(result.history) == 24
    assert result.final_metrics.fitness >= result.initial_metrics.fitness
    assert result.final_metrics.coherence >= result.initial_metrics.coherence
    assert result.final_metrics.verification >= result.initial_metrics.verification
    assert result.final_metrics.stability >= result.initial_metrics.stability


def test_invalid_coordinates_are_rejected() -> None:
    swarm = Inward3DDocumentSwarm()
    with pytest.raises(ValueError):
        swarm.linear_to_xyz(-1)
    with pytest.raises(ValueError):
        swarm.xyz_to_linear(DEFAULT_SIDE, 0, 0)
