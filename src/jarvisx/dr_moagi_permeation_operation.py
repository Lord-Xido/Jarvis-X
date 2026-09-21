"""Cloud-operation adapter for the Dr Moagi permeation transport."""

from __future__ import annotations

from typing import Any, Mapping

from .dr_moagi_cloud_runtime import ResourceLimits
from .dr_moagi_permeation import PermeationConfig, simulate_round_trip


class DrMoagiPermeationExecutor:
    """Execute one bounded software permeation round trip."""

    _CONFIG_FIELDS = frozenset(
        {
            "carrier_hz",
            "propagation_speed_m_s",
            "source_strength",
            "range_m",
            "coherence",
            "omni_weight",
            "quadrupole_weight",
            "axis",
            "receiver_direction",
            "noise_std",
            "noise_seed",
            "max_payload_bytes",
        }
    )

    def execute(self, payload: Mapping[str, Any], limits: ResourceLimits) -> Mapping[str, Any]:
        raw_state = payload.get("state")
        if not isinstance(raw_state, dict):
            raise ValueError("state must be a JSON object")
        raw_config = payload.get("config", {})
        if not isinstance(raw_config, dict):
            raise ValueError("config must be a JSON object")
        unknown = set(raw_config).difference(self._CONFIG_FIELDS)
        if unknown:
            raise ValueError(f"unknown permeation config keys: {sorted(unknown)}")

        config_payload = dict(raw_config)
        for vector_name in ("axis", "receiver_direction"):
            if vector_name in config_payload:
                value = config_payload[vector_name]
                if not isinstance(value, list) or len(value) != 3:
                    raise ValueError(f"{vector_name} must be a three-element list")
                config_payload[vector_name] = tuple(value)

        configured_max = config_payload.get("max_payload_bytes")
        if configured_max is None:
            config_payload["max_payload_bytes"] = min(65_536, limits.max_input_bytes)
        elif isinstance(configured_max, bool) or not isinstance(configured_max, int):
            raise ValueError("max_payload_bytes must be an integer")
        elif configured_max > limits.max_input_bytes:
            raise ValueError("max_payload_bytes exceeds cloud input budget")

        config = PermeationConfig(**config_payload)
        result = simulate_round_trip(raw_state, config)
        return {
            "operation": "permeate-roundtrip.v1",
            **result,
        }
