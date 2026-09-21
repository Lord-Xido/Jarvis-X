from __future__ import annotations

import pytest

from jarvisx.dr_moagi_multimodal_io import (
    DrMoagiMultimodalConfig,
    DrMoagiMultimodalRuntime,
    IdentityMediumAdapter,
    MediumChannel,
)


def field(value: float):
    return {(1, 1, 1): (value, 0.0)}


def make_runtime(**config_overrides):
    config = DrMoagiMultimodalConfig(
        side=4,
        vector_width=2,
        dt=0.5,
        input_gain=1.0,
        prediction_gain=0.0,
        error_gain=0.5,
        memory_gain=0.0,
        memory_decay=0.9,
        memory_error_gain=0.2,
        **config_overrides,
    )
    return DrMoagiMultimodalRuntime(
        [
            MediumChannel("vision", IdentityMediumAdapter()),
            MediumChannel("audio", IdentityMediumAdapter()),
        ],
        config=config,
    )


def test_multimodal_inputs_fuse_into_one_shared_3d_state():
    runtime = make_runtime()
    runtime.load({})

    result = runtime.step({"vision": field(1.0), "audio": field(0.0)})

    assert result.metrics.committed
    assert result.metrics.channels_seen == 2
    # weighted fusion is 0.5, then dt=0.5 pulls state halfway toward it
    assert runtime.snapshot()[(1, 1, 1)][0] == pytest.approx(0.25)


def test_feedback_residual_updates_state_and_persistent_memory():
    runtime = make_runtime()
    runtime.load({})

    result = runtime.step(
        {"vision": field(0.0)},
        targets={"vision": field(1.0)},
    )

    assert result.metrics.committed
    # error term: dt * error_gain * 1.0 = 0.25
    assert runtime.snapshot()[(1, 1, 1)][0] == pytest.approx(0.25)
    # Omega_(t+1) = rho*Omega_t + eta_omega*E_t
    assert runtime.memory_snapshot()[(1, 1, 1)][0] == pytest.approx(0.2)
    assert result.metrics.distortion_mse == pytest.approx(0.5)


def test_validator_rejection_is_atomic_for_state_and_memory():
    runtime = make_runtime()
    runtime.load(field(0.2), memory=field(0.1))
    state_before = runtime.snapshot()
    memory_before = runtime.memory_snapshot()

    result = runtime.step(
        {"vision": field(1.0)},
        targets={"vision": field(0.8)},
        validator=lambda candidate, metrics: False,
    )

    assert not result.metrics.committed
    assert result.metrics.rejection_reason == "validator rejected candidate"
    assert runtime.snapshot() == state_before
    assert runtime.memory_snapshot() == memory_before
    assert runtime.cycle == 1


def test_projection_clamps_all_channel_values():
    runtime = make_runtime(value_min=-0.5, value_max=0.5)
    runtime.load({})

    runtime.step({"vision": field(9.0)})

    assert runtime.snapshot()[(1, 1, 1)][0] <= 0.5


def test_unknown_channel_fails_closed():
    runtime = make_runtime()

    with pytest.raises(KeyError, match="unknown input channels"):
        runtime.step({"rf": field(1.0)})


def test_output_loopback_is_explicit_not_implied():
    class NoLoopbackAdapter(IdentityMediumAdapter):
        def observe_output(self, output):
            raise RuntimeError("physical sensor unavailable")

    runtime = DrMoagiMultimodalRuntime(
        [MediumChannel("display", NoLoopbackAdapter())],
        config=DrMoagiMultimodalConfig(
            side=4,
            vector_width=2,
            dt=0.1,
            prediction_gain=0.0,
        ),
    )
    runtime.load({})

    # No target means no claim that physical output was observed.
    result = runtime.step({"display": field(0.4)})
    assert result.metrics.committed

    with pytest.raises(RuntimeError, match="physical sensor unavailable"):
        runtime.step(
            {"display": field(0.4)},
            targets={"display": field(0.4)},
        )


def test_active_cell_budget_prevents_dense_materialization():
    runtime = make_runtime(max_active_cells=1)

    with pytest.raises(RuntimeError, match="active-cell budget"):
        runtime.step(
            {
                "vision": {
                    (0, 0, 0): (0.1, 0.0),
                    (1, 1, 1): (0.2, 0.0),
                }
            }
        )
