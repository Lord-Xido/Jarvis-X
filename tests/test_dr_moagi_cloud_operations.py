from __future__ import annotations

import pytest

from jarvisx.dr_moagi_cloud_operations import DrMoagiFieldStepExecutor
from jarvisx.dr_moagi_cloud_runtime import ResourceLimits


def test_field_step_operation_uses_canonical_bounded_runtime():
    executor = DrMoagiFieldStepExecutor()
    result = executor.execute(
        {
            "config": {
                "side": 5,
                "alpha": 0.0,
                "lambda_residual": 0.0,
                "eta": 0.0,
                "dt": 0.1,
                "expand_halo": False,
            },
            "field": [{"x": 2, "y": 2, "z": 2, "value": 0.5}],
        },
        ResourceLimits(),
    )

    assert result["operation"] == "dr-moagi-field-step.v1"
    assert result["virtual_cell_count"] == 125
    assert result["active_cell_count"] == 1
    assert result["cycle"] == 1
    assert result["metrics"]["committed"] is True
    assert result["field"] == [
        {"x": 2, "y": 2, "z": 2, "value": pytest.approx(0.5)}
    ]


def test_field_step_rejects_duplicate_sparse_coordinates():
    executor = DrMoagiFieldStepExecutor()
    with pytest.raises(ValueError, match="duplicate sparse coordinate"):
        executor.execute(
            {
                "config": {"side": 5, "expand_halo": False},
                "field": [
                    {"x": 2, "y": 2, "z": 2, "value": 0.5},
                    {"x": 2, "y": 2, "z": 2, "value": 0.6},
                ],
            },
            ResourceLimits(),
        )
