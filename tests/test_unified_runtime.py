from __future__ import annotations

import math

import pytest

from jarvisx.unified_runtime import UnifiedRuntime, UnifiedRuntimeConfig, UnifiedRuntimeRegistry


def test_unified_runtime_is_deterministic_for_same_history() -> None:
    config = UnifiedRuntimeConfig(max_dimensions=16)
    left = UnifiedRuntime(config)
    right = UnifiedRuntime(config)
    history = ([0.2, -0.4, 0.8, 0.1], [0.2, -0.4, 0.8, 0.1])

    left_states = [left.step(values) for values in history]
    right_states = [right.step(values) for values in history]

    assert [state.state_hash for state in left_states] == [
        state.state_hash for state in right_states
    ]
    assert left_states[-1].h_mmm == pytest.approx(right_states[-1].h_mmm)


def test_unified_runtime_closes_encode_decode_cycle() -> None:
    runtime = UnifiedRuntime(UnifiedRuntimeConfig(latent_block_size=2, max_dimensions=8))
    state = runtime.step([0.5, -0.25, 0.75, -0.5])

    assert len(state.psi) == 4
    assert len(state.latent) == 2
    assert len(state.reconstruction) == 4
    assert state.latent_cycle_mse == pytest.approx(0.0)
    assert state.reconstruction_mse >= 0.0
    assert state.verified
    assert len(state.state_hash) == 64


def test_omega_memory_feeds_next_psi_state() -> None:
    config = UnifiedRuntimeConfig(
        latent_block_size=2,
        omega_retention=0.0,
        psi_memory_gain=1.0,
        max_dimensions=8,
    )
    runtime = UnifiedRuntime(config)
    first = runtime.step([1.0, -0.5, 0.2, 0.9])
    second = runtime.step([1.0, -0.5, 0.2, 0.9])

    assert any(abs(value) > 0.0 for value in first.omega)
    assert second.psi != first.psi
    assert second.cycle == 2


def test_runtime_rejects_dimension_change_and_nonfinite_values() -> None:
    runtime = UnifiedRuntime(UnifiedRuntimeConfig(max_dimensions=4))
    runtime.step([0.0, 1.0])

    with pytest.raises(ValueError, match="dimensionality"):
        runtime.step([0.0, 1.0, 2.0])

    fresh = UnifiedRuntime(UnifiedRuntimeConfig(max_dimensions=4))
    with pytest.raises(ValueError, match="finite"):
        fresh.step([math.inf])


def test_registry_is_bounded_and_resettable() -> None:
    registry = UnifiedRuntimeRegistry(max_sessions=1)
    first = registry.create()
    second = registry.create()

    with pytest.raises(KeyError):
        registry.status(first["session_id"])

    stepped = registry.step(second["session_id"], [0.1, 0.2])
    assert stepped["state"]["cycle"] == 1
    reset = registry.reset(second["session_id"])
    assert reset["state"] is None
    assert registry.delete(second["session_id"])
