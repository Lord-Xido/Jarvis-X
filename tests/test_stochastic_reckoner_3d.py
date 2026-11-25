import math

import pytest

from jarvisx.stochastic_reckoner_3d import BeliefGrid3D


def test_stochastic_transition_and_boundary():
    g = BeliefGrid3D(3, success=0.75)
    b = g.predict(g.point_mass((0, 0, 0)), (1, 0, 0))
    assert b[g.index[(1, 0, 0)]] == pytest.approx(0.75)
    assert b[g.index[(0, 0, 0)]] == pytest.approx(0.25)
    assert math.fsum(b) == pytest.approx(1)
    assert g.predict(g.point_mass((0, 0, 0)), (-1, 0, 0)) == g.point_mass((0, 0, 0))


def test_bayes_update_matches_analytic_ratio():
    g = BeliefGrid3D(2, success=0.5)
    prior = g.predict(g.point_mass((0, 0, 0)), (1, 0, 0))
    b = g.observe(prior, (1, 0, 0), 1.0)
    assert b[g.index[(1, 0, 0)]] == pytest.approx(1 / (1 + math.exp(-0.5)))
    assert math.fsum(b) == pytest.approx(1)


def test_risk_gate_and_goal():
    g = BeliefGrid3D(3, success=1)
    b = g.point_mass((0, 0, 0))
    result = g.plan(b, (2, 0, 0), {(1, 0, 0)}, horizon=2, budget=100)
    assert result.plan is not None
    assert result.plan.actions[0] != (1, 0, 0)
    assert result.plan.risk_upper == 0
    assert not result.exhausted
    clear = g.plan(b, (2, 0, 0), set(), horizon=2, budget=100)
    assert clear.plan is not None
    assert clear.plan.actions == ((1, 0, 0), (1, 0, 0))


def test_initial_risk_and_budget_fail_closed():
    g = BeliefGrid3D(2)
    b = g.point_mass((0, 0, 0))
    blocked = g.plan(b, (1, 0, 0), {(0, 0, 0)})
    assert blocked.plan is None and blocked.evaluations == 0
    limited = g.plan(b, (1, 0, 0), set(), horizon=3, budget=1)
    assert limited.plan is None and limited.exhausted and limited.evaluations == 1


def test_invalid_inputs():
    g = BeliefGrid3D(2)
    with pytest.raises(ValueError):
        g.predict([float("nan")] * 8, (0, 0, 0))
    with pytest.raises(ValueError):
        g.predict([], (0, 0, 0))
    with pytest.raises(ValueError):
        g.observe(g.point_mass((0, 0, 0)), (1, 0, 0), 0)
    with pytest.raises(ValueError):
        g.plan(g.point_mass((0, 0, 0)), (1, 0, 0), set(), budget=0)


def test_risk_bound_accumulates_over_time():
    g = BeliefGrid3D(3, success=0.5)
    b = g.point_mass((0, 0, 0))
    result = g.plan(b, (2, 0, 0), {(1, 0, 0)}, horizon=2, budget=100, risk_limit=0.4)
    assert result.plan is not None
    assert result.plan.actions[0] != (1, 0, 0)
    assert result.plan.risk_upper <= 0.4
