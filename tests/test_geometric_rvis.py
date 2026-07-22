import importlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import pytest

from jarvisx.core import CodexVM
from jarvisx.geometric_rvis import (
    GeometricConfig,
    GeometricFeedbackRuntime,
    coordinate_to_index,
    index_to_coordinate,
    quantize_q3,
)
from jarvisx.serialization import json_safe


def test_coordinate_mapping_is_bijective():
    shape = (4, 4, 4)
    for index in range(64):
        coord = index_to_coordinate(index, shape)
        assert coordinate_to_index(coord, shape) == index


def test_coordinate_mapping_rejects_invalid_shapes_and_indices():
    with pytest.raises(ValueError, match="positive integers"):
        coordinate_to_index((0, 0, 0), (4, 0, 4))
    with pytest.raises(ValueError, match="integer"):
        index_to_coordinate(True, (4, 4, 4))


def test_quantization_rejects_non_finite_input():
    with pytest.raises(ValueError, match="finite"):
        quantize_q3(math.inf)


def test_geometric_pyramid_condenses_to_one_voxel():
    runtime = GeometricFeedbackRuntime()
    encoded, _ = runtime.encode([3, 2, 1, 0, -1, -2, -3, -4])
    hierarchy = runtime.condense(encoded)
    assert hierarchy[0].shape == (4, 4, 4)
    assert hierarchy[-1].shape == (1, 1, 1)
    assert len(hierarchy[-1].values) == 1


def test_non_cubic_power_of_two_lattice_reaches_a_valid_root():
    runtime = GeometricFeedbackRuntime(GeometricConfig(shape=(8, 4, 2)))
    encoded, input_length = runtime.encode([3, 2, 1, 0])
    hierarchy = runtime.condense(encoded, input_length)
    assert [level.shape for level in hierarchy] == [
        (8, 4, 2),
        (4, 2, 1),
        (2, 1, 1),
        (1, 1, 1),
    ]
    assert len(runtime.decode(hierarchy)) == runtime.config.volume


def test_sparse_padding_does_not_dominate_geometric_condensation():
    result = GeometricFeedbackRuntime().step([3, 3, 3, 3])
    assert result.committed
    assert result.selected_lane == "identity"
    assert result.output == (3, 3, 3, 3)
    assert result.metrics["best_reconstruction_l1"] == 0.0
    assert result.metrics["active_voxels"] == 4.0


def test_parallel_lanes_are_deterministic():
    values = [3, 1, -1, -3, 2, 0, -2, -4]
    left = GeometricFeedbackRuntime().step(values)
    right = GeometricFeedbackRuntime().step(values)
    assert left.committed and right.committed
    assert left.selected_lane == right.selected_lane
    assert left.candidate_hash == right.candidate_hash
    assert [lane.reconstruction_l1 for lane in left.lanes] == [
        lane.reconstruction_l1 for lane in right.lanes
    ]


def test_concurrent_steps_are_serialized_into_unique_committed_cycles():
    runtime = GeometricFeedbackRuntime()
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(
            executor.map(
                runtime.step,
                ([1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [-1, -1, -1, -1]),
            )
        )
    assert runtime.state.cycle == 4
    assert sorted(result.cycle for result in results) == [1, 2, 3, 4]
    assert len(runtime.journal) == 4


def test_feedback_turns_committed_output_into_next_input():
    runtime = GeometricFeedbackRuntime(GeometricConfig(feedback_cycles=3))
    results = runtime.run_feedback([3, 1, -1, -3])
    assert len(results) == 3
    assert all(result.committed for result in results)
    assert results[1].encoded[:4] == results[0].output
    assert results[-1].state_hash == runtime.state.state_hash


def test_feedback_rejects_zero_and_excessive_cycle_counts():
    runtime = GeometricFeedbackRuntime(GeometricConfig(max_feedback_cycles=4))
    with pytest.raises(ValueError, match="positive integer"):
        runtime.run_feedback([1], cycles=0)
    with pytest.raises(ValueError, match="maximum"):
        runtime.run_feedback([1], cycles=5)


def test_lambda_rejection_preserves_committed_state():
    config = GeometricConfig(max_reconstruction_l1=0)
    runtime = GeometricFeedbackRuntime(config)
    before = runtime.snapshot()
    result = runtime.step([3, -4, 3, -4])
    assert not result.committed
    assert runtime.snapshot() == before
    assert result.state_hash == "GENESIS"


def test_invalid_lambda_budgets_are_rejected_at_configuration_time():
    with pytest.raises(ValueError, match="non-negative"):
        GeometricFeedbackRuntime(GeometricConfig(max_reconstruction_l1=-1))
    with pytest.raises(ValueError, match="fit inside"):
        GeometricFeedbackRuntime(GeometricConfig(max_active_voxels=65))


def test_shell_events_expose_3d_pipeline_without_uncommitted_render():
    result = GeometricFeedbackRuntime().step([1, 2, 3, -4])
    phases = [event["phase"] for event in result.events]
    assert phases[0] == "GEOM_ENCODE"
    assert phases.count("MULTIPARALLEL_LANE") == 4
    assert phases[-1] == "COMMIT"
    assert result.events[-1]["committed"] is True


def test_cycle_reports_memory_before_and_after_without_aliasing():
    runtime = GeometricFeedbackRuntime()
    first = runtime.step([3, 1, -1, -3])
    second = runtime.step(first.output)
    assert first.omega_before == ()
    assert second.omega_before == first.omega_after


def test_codex_vm_projects_geometric_commit_into_registers():
    vm = CodexVM()
    results = vm.geometric_feedback([3, 1, -1, -3], cycles=2)
    final = results[-1]
    assert final.committed
    assert vm.regs["Λ"] == 1
    assert vm.regs["Ψ"] == final.hierarchy[-1].values[0]
    assert vm.regs["𝒮"] == int(final.metrics["best_reconstruction_l1"])
    assert vm.regs["Π"] == sum(final.output)


def test_rejected_cycle_serializes_as_strict_browser_json():
    runtime = GeometricFeedbackRuntime(GeometricConfig(max_reconstruction_l1=0))
    rejected = runtime.step([3, -4, 3, -4])
    payload = json_safe(rejected.to_dict())
    encoded = json.dumps(payload, allow_nan=False)
    assert "Infinity" not in encoded
    assert payload["metrics"]["best_reconstruction_l1"] is None


def test_cli_import_does_not_load_optional_service_modules():
    for module_name in ("jarvisx.api", "jarvisx.web", "jarvisx.node"):
        sys.modules.pop(module_name, None)
    cli = importlib.import_module("jarvisx.cli")
    importlib.reload(cli)
    assert "jarvisx.api" not in sys.modules
    assert "jarvisx.web" not in sys.modules
    assert "jarvisx.node" not in sys.modules


def test_browser_shell_escapes_trace_text_and_bounds_loaded_geometry():
    shell = (Path(__file__).parents[1] / "geometric-rvis-shell.html").read_text()
    assert "function escapeHtml" in shell
    assert "escapeHtml(safe.summary" in shell
    assert "MAX_VOXELS" in shell
    assert "MAX_JSON_BYTES" in shell
