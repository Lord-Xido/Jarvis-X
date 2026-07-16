import json
import math

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


def test_feedback_turns_committed_output_into_next_input():
    runtime = GeometricFeedbackRuntime(GeometricConfig(feedback_cycles=3))
    results = runtime.run_feedback([3, 1, -1, -3])
    assert len(results) == 3
    assert all(result.committed for result in results)
    assert results[1].encoded[:4] == results[0].output
    assert results[-1].state_hash == runtime.state.state_hash


def test_lambda_rejection_preserves_committed_state():
    config = GeometricConfig(max_reconstruction_l1=0)
    runtime = GeometricFeedbackRuntime(config)
    before = runtime.snapshot()
    result = runtime.step([3, -4, 3, -4])
    assert not result.committed
    assert runtime.snapshot() == before
    assert result.state_hash == "GENESIS"


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
