import json
import subprocess
import sys

import pytest
from jarvisx.chrysalis3d import ChrysalisConfig, ChrysalisTheta3D, GridShape


def small_config(**overrides):
    values = {
        "grid": GridShape(width=3, height=2, depth=2),
        "d_model": 16,
        "top_k": 2,
        "expert_rank": 4,
        "num_heads": 4,
        "seed": 1234,
    }
    values.update(overrides)
    return ChrysalisConfig(**values)


def modalities():
    return {
        "text": "motion through a 3d field",
        "image": [0.1, 0.4, -0.2, 0.7, 0.3],
        "audio": [0.2, -0.1, 0.5, 0.9],
        "video": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    }


def test_non_cubic_grid_round_trip_and_depth_major_indexing():
    shape = GridShape(width=3, height=2, depth=4)
    for index in range(shape.positions):
        coordinate = shape.coordinate(index)
        assert shape.index(coordinate) == index
    assert shape.index((2, 1, 3)) == 23


def test_true_top_k_dispatch_and_normalized_gate_weights():
    engine = ChrysalisTheta3D(small_config())
    result = engine.step(modalities())
    assert len(result.activations) == 2
    assert sum(item.weight for item in result.activations) == pytest.approx(1.0)
    assert len({item.index for item in result.activations}) == 2
    assert result.arithmetic.activation_ratio == pytest.approx(2.0 / 12.0)


def test_deterministic_output_and_state_hash():
    first = ChrysalisTheta3D(small_config())
    second = ChrysalisTheta3D(small_config())
    result_a = first.step(modalities())
    result_b = second.step(modalities())
    assert result_a.output == result_b.output
    assert result_a.activations == result_b.activations
    assert result_a.state_hash == result_b.state_hash


def test_recurrent_state_evolves_and_reset_restores_initial_trajectory():
    engine = ChrysalisTheta3D(small_config())
    first = engine.step(modalities())
    second = engine.step(modalities())
    assert first.state_hash != second.state_hash
    engine.reset()
    replay = engine.step(modalities())
    assert replay.state_hash == first.state_hash
    assert replay.output == first.output


def test_experts_are_coordinate_distinct():
    config_a = small_config(top_k=1, seed=222)
    config_b = small_config(top_k=1, seed=333)
    first = ChrysalisTheta3D(config_a).step(modalities())
    second = ChrysalisTheta3D(config_b).step(modalities())
    assert (
        first.activations[0].index != second.activations[0].index
        or first.output != second.output
    )


def test_attention_shape_is_heads_by_modalities():
    engine = ChrysalisTheta3D(small_config())
    result = engine.step(modalities())
    assert len(result.modality_attention) == 4
    for head in result.modality_attention:
        assert len(head) == 4
        assert sum(head) == pytest.approx(1.0)


def test_invalid_configuration_fails_closed():
    with pytest.raises(ValueError):
        ChrysalisConfig(grid=GridShape(2, 2, 2), d_model=15, num_heads=4)
    with pytest.raises(ValueError):
        ChrysalisConfig(grid=GridShape(2, 2, 2), top_k=9)


def test_cli_executes_sequence(tmp_path):
    payload = {
        "config": {
            "width": 2,
            "height": 2,
            "depth": 2,
            "d_model": 16,
            "top_k": 2,
            "expert_rank": 4,
            "num_heads": 4,
            "seed": 99,
        },
        "sequence": [modalities(), modalities()],
    }
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "jarvisx.chrysalis3d_cli",
            "@" + str(path),
            "--summary-only",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    decoded = json.loads(completed.stdout)
    assert len(decoded["results"]) == 2
    assert decoded["snapshot"]["step"] == 2
