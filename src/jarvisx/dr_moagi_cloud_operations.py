"""Bounded Dr Moagi domain operations exposed through the cloud coordinator."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

from .dr_moagi_cloud_runtime import ResourceLimits
from .dr_moagi_field_runtime import DrMoagiFieldConfig, DrMoagiFieldRuntime, IdentityFieldCodec


class DrMoagiFieldStepExecutor:
    """Execute one candidate/commit field-runtime step from a sparse JSON payload."""

    _CONFIG_FIELDS = frozenset(
        {
            "side",
            "alpha",
            "lambda_residual",
            "eta",
            "dt",
            "value_min",
            "value_max",
            "max_active_cells",
            "expand_halo",
            "prune_epsilon",
            "enforce_conservative_step_bound",
        }
    )

    def execute(self, payload: Mapping[str, Any], limits: ResourceLimits) -> Mapping[str, Any]:
        del limits
        raw_field = payload.get("field")
        if not isinstance(raw_field, list):
            raise ValueError("field must be a list of sparse cells")
        raw_config = payload.get("config", {})
        if not isinstance(raw_config, dict):
            raise ValueError("config must be an object")
        unknown = set(raw_config).difference(self._CONFIG_FIELDS)
        if unknown:
            raise ValueError(f"unknown field config keys: {sorted(unknown)}")

        config = DrMoagiFieldConfig(**raw_config)
        field: dict[tuple[int, int, int], float] = {}
        for index, cell in enumerate(raw_field):
            if not isinstance(cell, dict):
                raise ValueError(f"field[{index}] must be an object")
            required = {"x", "y", "z", "value"}
            if set(cell) != required:
                raise ValueError(f"field[{index}] must contain exactly x, y, z, value")
            coordinate = (
                self._integer(cell["x"], f"field[{index}].x"),
                self._integer(cell["y"], f"field[{index}].y"),
                self._integer(cell["z"], f"field[{index}].z"),
            )
            if coordinate in field:
                raise ValueError(f"duplicate sparse coordinate at field[{index}]")
            field[coordinate] = self._number(cell["value"], f"field[{index}].value")

        runtime = DrMoagiFieldRuntime(IdentityFieldCodec(), config)
        runtime.load(field)
        metrics = runtime.step()
        state = runtime.snapshot()
        cells = [
            {"x": x, "y": y, "z": z, "value": value}
            for (x, y, z), value in sorted(state.items())
        ]
        return {
            "operation": "dr-moagi-field-step.v1",
            "virtual_cell_count": runtime.virtual_cell_count,
            "active_cell_count": runtime.active_cell_count,
            "cycle": runtime.cycle,
            "metrics": asdict(metrics),
            "field": cells,
        }

    @staticmethod
    def _integer(value: object, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        return value

    @staticmethod
    def _number(value: object, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric")
        return float(value)
