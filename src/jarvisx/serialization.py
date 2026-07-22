"""Serialization helpers for deterministic browser-safe runtime output."""

import math


def json_safe(value):
    """Recursively replace non-finite floats with JSON null-compatible values."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value
