import pytest

from jarvisx.dsmpe_reference import DsmpeConfig, complete_octree_nodes, encode_model, select_model


def test_complete_octree_node_count() -> None:
    assert complete_octree_nodes(0) == 1
    assert complete_octree_nodes(1) == 9
    assert complete_octree_nodes(4) == 4681


def test_encoding_is_deterministic_and_volume_preserving() -> None:
    config = DsmpeConfig(max_depth=4, gamma=0.4, sample_resolution=7)
    first = encode_model(config)
    second = encode_model(config)

    assert first.metrics.digest_sha256 == second.metrics.digest_sha256
    assert first.metrics.partition_volume_error < 1.0e-9
    assert 0.0 <= first.metrics.compression_ratio < 1.0
    assert 0.0 < first.metrics.complexity_ratio <= 1.0
    assert first.metrics.visited_nodes <= first.metrics.total_possible_nodes
    assert first.metrics.boundary_leaves > 0


def test_refinement_reduces_reconstruction_error() -> None:
    coarse = encode_model(DsmpeConfig(max_depth=1, gamma=0.0, sample_resolution=9))
    fine = encode_model(DsmpeConfig(max_depth=4, gamma=0.0, sample_resolution=9))
    assert fine.metrics.reconstruction_rmse < coarse.metrics.reconstruction_rmse


def test_depth_search_selects_observed_minimum() -> None:
    selection = select_model(DsmpeConfig(max_depth=4, gamma=0.35, sample_resolution=7))
    observed = [candidate.metrics.loss for candidate in selection.candidates]
    assert selection.selected.metrics.loss == min(observed)
    assert selection.selected in selection.candidates


def test_decode_rejects_points_outside_root() -> None:
    model = encode_model(DsmpeConfig(max_depth=2, sample_resolution=5))
    with pytest.raises(ValueError, match="outside"):
        model.decode((100.0, 0.0, 0.0))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"root_size": 0.0}, "positive"),
        ({"max_depth": 0}, "at least 1"),
        ({"gamma": 1.1}, "between 0 and 1"),
        ({"sample_resolution": 1}, "at least 2"),
    ],
)
def test_invalid_config_is_rejected(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        DsmpeConfig(**kwargs)
