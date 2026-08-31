import json
import math

import pytest

from jarvisx.dr_moagi_monadic_resonator import (
    MonadicResonator,
    ResonatorConfig,
    linear_relaxation,
    main,
    scaled_identity,
)


def test_zero_dynamics_reduces_to_decode_encode() -> None:
    engine = MonadicResonator(
        encoder=scaled_identity(2.0),
        dynamics=lambda latent, _time: tuple(0.0 for _ in latent),
        decoder=scaled_identity(0.5),
    )

    report = engine.step((1.0, -2.0, 3.0))

    assert report.latent_start == (2.0, -4.0, 6.0)
    assert report.latent_integral == (0.0, 0.0, 0.0)
    assert report.output_state == pytest.approx((1.0, -2.0, 3.0))


def test_constant_field_matches_integral_law_exactly() -> None:
    engine = MonadicResonator(
        encoder=scaled_identity(1.0),
        dynamics=lambda latent, _time: tuple(2.0 for _ in latent),
        decoder=scaled_identity(1.0),
        config=ResonatorConfig(interval=1.0, substeps=4, method="rk4"),
    )

    report = engine.step((1.0, -1.0))

    assert report.latent_integral == pytest.approx((2.0, 2.0))
    assert report.latent_end == pytest.approx((3.0, 1.0))
    assert report.output_state == pytest.approx((3.0, 1.0))
    assert report.derivative_evaluations == 16


def test_rk4_tracks_linear_relaxation_solution() -> None:
    rate = 0.75
    engine = MonadicResonator(
        encoder=scaled_identity(1.0),
        dynamics=linear_relaxation(rate),
        decoder=scaled_identity(1.0),
        config=ResonatorConfig(interval=1.0, substeps=32, method="rk4"),
    )

    report = engine.step((2.0,))

    expected = 2.0 * math.exp(-rate)
    assert report.output_state[0] == pytest.approx(expected, rel=1.0e-7)
    assert report.latent_delta_l2 == pytest.approx(abs(expected - 2.0), rel=1.0e-7)


def test_euler_reports_one_derivative_evaluation_per_substep() -> None:
    engine = MonadicResonator(
        encoder=scaled_identity(1.0),
        dynamics=linear_relaxation(0.1),
        decoder=scaled_identity(1.0),
        config=ResonatorConfig(substeps=7, method="euler"),
    )

    report = engine.step((1.0, 2.0))

    assert report.derivative_evaluations == 7
    assert report.integration_method == "euler"


def test_dynamics_dimension_change_is_rejected() -> None:
    engine = MonadicResonator(
        encoder=scaled_identity(1.0),
        dynamics=lambda _latent, _time: (1.0, 2.0),
        decoder=scaled_identity(1.0),
    )

    with pytest.raises(ValueError, match="dimension changed"):
        engine.step((1.0,))


def test_non_finite_latent_state_is_rejected() -> None:
    engine = MonadicResonator(
        encoder=lambda _state: (float("inf"),),
        dynamics=linear_relaxation(0.1),
        decoder=scaled_identity(1.0),
    )

    with pytest.raises(ValueError, match="finite"):
        engine.step((1.0,))


def test_rollout_recursively_feeds_decoder_output_forward() -> None:
    engine = MonadicResonator(
        encoder=scaled_identity(1.0),
        dynamics=lambda latent, _time: tuple(1.0 for _ in latent),
        decoder=scaled_identity(1.0),
        config=ResonatorConfig(interval=0.5, substeps=2, method="euler"),
    )

    reports = engine.rollout((0.0,), steps=3, t_start=2.0)

    assert [report.output_state[0] for report in reports] == pytest.approx([0.5, 1.0, 1.5])
    assert [report.t_end for report in reports] == pytest.approx([2.5, 3.0, 3.5])


def test_cli_emits_auditable_equation_payload(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "--state",
            "1,2",
            "--steps",
            "1",
            "--rate",
            "0",
            "--substeps",
            "2",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["law_id"] == MonadicResonator.LAW_ID
    assert payload["equation"] == MonadicResonator.EQUATION
    assert payload["steps"][0]["output_state"] == [1.0, 2.0]


def test_configuration_rejects_unbounded_or_invalid_solver_inputs() -> None:
    with pytest.raises(ValueError, match="interval"):
        ResonatorConfig(interval=0.0)
    with pytest.raises(ValueError, match="substeps"):
        ResonatorConfig(substeps=0)
    with pytest.raises(ValueError, match="method"):
        ResonatorConfig(method="bogus")  # type: ignore[arg-type]
