import json

import pytest

from jarvisx.moagi_3d_geometric_intelligence import (
    EXPONENT10_PER_AXIS,
    SEPTILLION,
    FractalOctreeNode,
    GridTokenCodec,
    MoagiGeometricIntelligence,
    axis_from_virtual,
    axis_to_virtual,
    format_virtual_coord,
    main,
    restore_virtual_coord,
)


def test_virtual_extent_is_symbolic_and_math_is_exact():
    assert SEPTILLION == 10**24
    assert EXPONENT10_PER_AXIS == 24 * 10**24
    assert format_virtual_coord((1, 0, 0))["coordinate_notation"][0].endswith("-1)")


@pytest.mark.parametrize("v", [-1.0, -0.3333333333, -1e-25, -1e-100, 0.0, 1e-100, 1e-25, 0.25, 1.0])
def test_virtual_axis_round_trip(v):
    actual = axis_from_virtual(json.loads(json.dumps(axis_to_virtual(v))))
    assert actual == pytest.approx(v, rel=2e-14, abs=1e-110)


def test_three_axis_roundtrip_and_json_serialization():
    point = (0.25, -0.75, 0.0)
    payload = json.loads(json.dumps(format_virtual_coord(point)))
    assert restore_virtual_coord(payload) == pytest.approx(point, abs=1e-13)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.1, 1.1])
def test_invalid_virtual_inputs(bad):
    with pytest.raises(ValueError):
        axis_to_virtual(bad)


def test_unsafe_virtual_payloads_rejected():
    a = axis_to_virtual(0.5)
    for malformed in ({"sign": 1, "triple_log": float("inf")},
                      {"sign": 0, "triple_log": 1.0},
                      {"sign": 2, "triple_log": 0.1},
                      {"sign": 1, "triple_log": 100.0},
                      {"sign": 1, "triple_log": 0.0}):
        with pytest.raises(ValueError):
            axis_from_virtual(malformed)
    assert axis_from_virtual(a) == pytest.approx(0.5)


def test_sparse_octree_redistributes_points_and_tracks_total():
    tree = FractalOctreeNode(capacity=2, max_depth=7)
    points = [(-0.75, -0.75, -0.75), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.2, 0.2, 0.2)]
    for point in points:
        assert tree.insert(point)
    assert tree.token_count == len(points)
    assert tree.count_nodes() > 1
    assert tree.count_leaves() > 1
    assert tree.validate_counts()
    with pytest.raises(ValueError):
        tree.insert((2.0, 0.0, 0.0))  # outside the allowed chart
    assert tree.token_count == len(points)


def test_octree_boundary_centers_are_not_limited_to_data_chart():
    tree = FractalOctreeNode(center=(1, 1, 1), half_size=1, capacity=1)
    assert tree.insert((1.0, 1.0, 1.0))
    assert tree.insert((1.0, 1.0, 1.0))
    assert tree.count_nodes() > 1
    assert tree.validate_counts()
    with pytest.raises(ValueError):
        FractalOctreeNode(center=(float("inf"), 0, 0))


def test_sparse_octree_handles_exact_boundary_and_clusters():
    tree = FractalOctreeNode(max_depth=5, capacity=1)
    for _ in range(50):
        assert tree.insert((1.0, 1.0, 1.0))
    assert tree.validate_counts()
    assert tree.count_nodes() <= 1 + 5


def test_complete_ordered_token_cloud_roundtrips_unicode_spacing_and_punctuation():
    engine = MoagiGeometricIntelligence()
    text = "The king sees the crown!  Dr Moagi: 3D. 🚀\n"
    encoded = engine.encode_to_virtual(text)
    assert encoded["token_count"] == len(engine.base.encode(text).tokens)
    assert engine.decode_from_virtual(json.loads(json.dumps(encoded)))["decoded_text"] == text
    assert engine.octree.validate_counts()


def test_sentence_point_alone_is_not_a_text_codec():
    engine = MoagiGeometricIntelligence()
    encoded = engine.encode_to_virtual("the king sees the crown")
    with pytest.raises(ValueError, match="full token-cloud"):
        engine.decode_from_virtual(encoded["virtual_sentence"])


def test_calibration_actually_reduces_geometric_error_and_only_commits_nonworse():
    engine = MoagiGeometricIntelligence(decoder_gain=0.82)
    log = engine.auto_evolve(["the king sees the crown", "a different sentence"], iterations=7)
    assert len(log) == 7
    assert log[0]["mse_before"] > 0
    assert log[-1]["mse_after"] < log[0]["mse_before"]
    assert engine.decoder_gain > 0.82
    assert all(r["mse_after"] <= r["mse_before"] + 1e-20 for r in log)
    assert engine.octree.validate_counts()


def test_calibration_does_not_fabricate_improvement_at_perfect_gain():
    engine = MoagiGeometricIntelligence(decoder_gain=1.0)
    log = engine.auto_evolve(["testing"], iterations=2)
    assert max(row["mse_after"] for row in log) < 1e-25
    assert engine.decoder_gain == pytest.approx(1.0)


def test_empty_training_is_invalid_and_capacity_is_bounded():
    with pytest.raises(ValueError):
        MoagiGeometricIntelligence().auto_evolve([""], iterations=1)
    with pytest.raises(ValueError):
        GridTokenCodec(max_tokens=1).encode("two tokens")


def test_cli_manifest_is_local_and_truthful(tmp_path, capsys):
    output = tmp_path / "moagi.json"
    assert main(["--text", "the king sees the crown", "--iterations", "3", "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["token_round_trip"] == [True]
    assert report["octree_counts_valid"] is True
    assert "symbolic" in report["virtual_cube_cells"]
    assert "virtual_axis" in capsys.readouterr().out
