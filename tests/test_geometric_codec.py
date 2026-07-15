import math

import pytest

from jarvisx.geometric_codec import (
    GeometricAutoEncoder,
    GeometricRuntime,
    GridGeometry,
    PermutationTransform,
    validate_latent,
)


def test_arithmetic_coordinate_mapping_is_bijective():
    geometry = GridGeometry(4, 3, 2)

    assert geometry.size == 24
    assert geometry.validate_round_trip()

    coordinates = [geometry.arithmetic_to_coordinate(i) for i in range(geometry.size)]
    assert len(set(coordinates)) == geometry.size
    assert [geometry.coordinate_to_arithmetic(c) for c in coordinates] == list(
        range(geometry.size)
    )


def test_exact_encode_decode_cycle():
    codec = GeometricAutoEncoder()
    values = list(range(24))

    latent = codec.encode(values, (4, 3, 2))
    report = validate_latent(latent)

    assert report.passed
    assert codec.decode(latent) == values


def test_permutation_transform_and_inverse_close_cycle():
    codec = GeometricAutoEncoder()
    values = list(range(8))
    transform = PermutationTransform.from_sequence((7, 0, 5, 2, 1, 6, 3, 4))

    latent = codec.encode(values, (2, 2, 2))
    transformed = codec.transform(latent, transform)
    recovered = codec.inverse_transform(transformed, transform)

    assert codec.decode(recovered) == values
    assert transform.inverse().inverse() == transform


def test_runtime_commits_only_valid_geometric_state():
    runtime = GeometricRuntime()
    runtime.load(list(range(8)), (2, 2, 2))

    transform = PermutationTransform.from_sequence((1, 2, 3, 4, 5, 6, 7, 0))
    candidate = runtime.propose(transform)

    assert candidate.validation.passed
    assert not candidate.committed

    committed = runtime.commit()
    assert committed.version == 1
    assert runtime.pending is None
    assert runtime.journal[-1]["operation"] == "COMMIT"
    assert runtime.journal[-1]["validation_passed"]


def test_runtime_rolls_back_pending_candidate():
    runtime = GeometricRuntime()
    initial = runtime.load(list(range(8)), (2, 2, 2))
    initial_hash = initial.digest()

    runtime.propose(PermutationTransform.from_sequence((7, 6, 5, 4, 3, 2, 1, 0)))
    recovered = runtime.rollback()

    assert recovered is not None
    assert recovered.digest() == initial_hash
    assert runtime.pending is None
    assert runtime.journal[-1]["operation"] == "ROLLBACK"


def test_spatial_omega_memory_tracks_local_residual():
    runtime = GeometricRuntime()
    runtime.load([0.0] * 8, (2, 2, 2))

    omega = runtime.update_omega([1.0] + [0.0] * 7, retention=1.0, learning_rate=0.5)

    assert omega[(0, 0, 0)] == pytest.approx(0.5)
    assert sum(abs(value) for value in omega.values()) == pytest.approx(0.5)


def test_invalid_permutation_is_rejected():
    with pytest.raises(ValueError):
        PermutationTransform.from_sequence((0, 1, 1, 3))


def test_non_finite_candidate_fails_validation():
    codec = GeometricAutoEncoder()
    latent = codec.encode([0.0, 1.0, math.inf, 3.0], (1, 2, 2))

    report = validate_latent(latent)

    assert not report.passed
    assert not report.values_finite


def test_runtime_round_trip_identity_under_arbitrary_permutation():
    runtime = GeometricRuntime()
    values = ["zero", "one", "two", "three", "four", "five", "six", "seven"]
    transform = PermutationTransform.from_sequence((4, 0, 7, 2, 6, 1, 5, 3))

    assert runtime.round_trip(values, (2, 2, 2), transform)
