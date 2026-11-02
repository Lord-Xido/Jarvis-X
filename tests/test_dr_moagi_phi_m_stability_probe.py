"""Bounded 3D iterated-map stability and two-cycle conformance checks."""
import pytest

from jarvisx.dr_moagi_phi_m_stability_probe import StabilityConfig, probe_stability


def test_two_cycle_is_not_misidentified_as_fixed_point():
    report = probe_stability(StabilityConfig(points=32, steps=180, times=(1.5,)))
    item = report["results"][0]
    assert item["terminal_one_step_max_residual"] > 0.1
    assert item["terminal_two_step_max_residual"] < 1e-6
    assert item["classification"] == "sampled_period_two_convergence"
    assert item["terminal_max_two_step_spectral_radius"] < 1.0


def test_initial_jacobian_is_not_globally_contractive():
    report = probe_stability(StabilityConfig(points=32, steps=20, times=(1.5,)))
    assert report["results"][0]["initial_sample_max_jacobian_singular_value"] > 1.0


@pytest.mark.parametrize("kwargs", [
    {"points": 1}, {"steps": 1}, {"steps": 10000},
    {"tolerance": float("nan")}, {"times": (float("inf"),)},
])
def test_invalid_dynamics_budgets_rejected(kwargs):
    with pytest.raises(ValueError):
        StabilityConfig(**kwargs)
