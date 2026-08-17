from __future__ import annotations

import math

import pytest

from jarvisx.dr_moagi_state_equation import (
    DrMoagiEquationConfig,
    DrMoagiStateEquation,
    box_projector,
    merge_predictive_branches,
)


CENTER = (1, 1, 1)


def singleton(value: float):
    return {CENTER: value}


def test_exact_recurrence_then_pi_lambda_projection():
    equation = DrMoagiStateEquation(
        DrMoagiEquationConfig(kappa=2.0, eta_z=0.1, zeta=0.2),
        projector=box_projector(-2.0, 2.0),
    )

    step = equation.step(
        singleton(1.0),
        prediction_branches=[singleton(2.0), singleton(4.0)],
        prediction_weights=[0.25, 0.75],
        error=singleton(0.5),
        memory=singleton(0.25),
        refinement=singleton(1.0),
        latent_gradient=singleton(0.5),
        constraint_gradient=singleton(0.25),
    )

    # P = 0.25*2 + 0.75*4 = 3.5
    # raw = 1 + 3.5 - 0.5 + 0.25 + 2*1 - 0.1*0.5 - 0.2*0.25
    assert step.raw_candidate[CENTER] == pytest.approx(6.15)
    assert step.projected_candidate[CENTER] == pytest.approx(2.0)
    assert step.next_state[CENTER] == pytest.approx(2.0)
    assert step.terms.latent_gradient[CENTER] == pytest.approx(0.5)
    assert step.committed
    assert step.branch_count == 2


def test_predictive_merge_is_convex_and_does_not_scale_with_branch_count():
    support = {CENTER}

    two = merge_predictive_branches(
        [singleton(1.0), singleton(3.0)],
        support=support,
    )
    four = merge_predictive_branches(
        [singleton(1.0), singleton(1.0), singleton(3.0), singleton(3.0)],
        support=support,
    )

    assert two[CENTER] == pytest.approx(2.0)
    assert four[CENTER] == pytest.approx(2.0)


def test_same_space_invariant_rejects_support_mismatch():
    equation = DrMoagiStateEquation()

    with pytest.raises(ValueError, match="share Xi support"):
        equation.step(
            singleton(1.0),
            prediction_branches=[singleton(0.0)],
            error={(0, 0, 0): 0.0},
            memory=singleton(0.0),
            refinement=singleton(0.0),
            latent_gradient=singleton(0.0),
            constraint_gradient=singleton(0.0),
        )


def test_validator_rejection_rolls_back_to_snapshot():
    equation = DrMoagiStateEquation(projector=box_projector(-10.0, 10.0))
    initial = singleton(1.0)

    step = equation.step(
        initial,
        prediction_branches=[singleton(2.0)],
        error=singleton(0.0),
        memory=singleton(0.0),
        refinement=singleton(0.0),
        latent_gradient=singleton(0.0),
        constraint_gradient=singleton(0.0),
        validator=lambda candidate: False,
    )

    assert not step.committed
    assert step.rejection_reason == "validator rejected Pi_Lambda candidate"
    assert step.raw_candidate[CENTER] == pytest.approx(3.0)
    assert step.next_state == initial
    assert initial == singleton(1.0)


def test_non_finite_term_fails_closed_before_projection():
    equation = DrMoagiStateEquation()

    with pytest.raises(ValueError, match="non-finite"):
        equation.step(
            singleton(1.0),
            prediction_branches=[singleton(0.0)],
            error=singleton(math.inf),
            memory=singleton(0.0),
            refinement=singleton(0.0),
            latent_gradient=singleton(0.0),
            constraint_gradient=singleton(0.0),
        )


def test_projection_cannot_escape_authoritative_support():
    def escaping_projector(field):
        projected = dict(field)
        projected[(9, 9, 9)] = 1.0
        return projected

    equation = DrMoagiStateEquation(projector=escaping_projector)

    with pytest.raises(ValueError, match="share Xi support"):
        equation.step(
            singleton(1.0),
            prediction_branches=[singleton(0.0)],
            error=singleton(0.0),
            memory=singleton(0.0),
            refinement=singleton(0.0),
            latent_gradient=singleton(0.0),
            constraint_gradient=singleton(0.0),
        )


def test_predictive_branch_contract_rejects_zero_or_invalid_weights():
    with pytest.raises(ValueError, match="at least one predictive branch"):
        merge_predictive_branches([], support={CENTER})

    with pytest.raises(ValueError, match="positive sum"):
        merge_predictive_branches(
            [singleton(1.0), singleton(2.0)],
            support={CENTER},
            weights=[0.0, 0.0],
        )
