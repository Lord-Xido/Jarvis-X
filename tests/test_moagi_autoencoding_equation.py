import math

import pytest

from jarvisx.moagi_autoencoding_equation import (
    MoagiObjectiveConfig,
    contractive_penalty,
    evaluate_autoencoding_objective,
    expectation_value,
    gaussian_basis,
    gradient_step,
    normalize_wavefunction,
    phase_angle,
    reconstruction_loss,
    refinement_energy,
    shannon_entropy,
    wavefunction_components,
)


def test_reconstruction_loss_matches_half_squared_l2():
    assert reconstruction_loss((1.0, 2.0), (0.0, 4.0)) == pytest.approx(2.5)


def test_shannon_entropy_uses_normalized_distribution_in_bits():
    assert shannon_entropy((0.5, 0.5)) == pytest.approx(1.0)
    assert shannon_entropy((1.0, 0.0)) == pytest.approx(0.0)

    with pytest.raises(ValueError, match="sum to one"):
        shannon_entropy((0.4, 0.4))


def test_contractive_penalty_uses_full_encoder_jacobian():
    jacobian = ((1.0, 2.0), (3.0, 4.0))
    assert contractive_penalty(jacobian) == pytest.approx(15.0)


def test_refinement_energy_combines_entropy_and_jacobian_terms():
    omega = refinement_energy(
        probabilities=(0.5, 0.5),
        encoder_jacobian=((1.0, 0.0),),
        entropy_weight=2.0,
        jacobian_weight=3.0,
    )
    assert omega == pytest.approx(3.5)


def test_gaussian_basis_has_correct_n_dimensional_normalizer():
    value = gaussian_basis((0.0, 0.0), (0.0, 0.0), 1.0)
    assert value == pytest.approx(1.0 / (2.0 * math.pi))


def test_phase_angle_uses_quadrant_correct_atan2():
    assert phase_angle(complex(-1.0, 1.0)) == pytest.approx(3.0 * math.pi / 4.0)


def test_wavefunction_components_preserve_complex_phase_and_normalize():
    psi = wavefunction_components(
        x=(0.0,),
        means=((0.0,), (1.0,)),
        sigmas=(1.0, 1.0),
        amplitudes=(1.0 + 0.0j, 1.0 + 0.0j),
        phases=(0.0, math.pi / 2.0),
    )
    assert len(psi) == 2
    state = normalize_wavefunction(psi)
    assert sum(abs(value) ** 2 for value in state) == pytest.approx(1.0)
    assert abs(psi[0].imag) < 1e-12
    assert psi[1].imag > 0.0


def test_expectation_value_is_real_for_hermitian_operator():
    psi = (1.0 + 0.0j, 1.0j)
    hamiltonian = ((1.0, 0.0), (0.0, 3.0))
    assert expectation_value(psi, hamiltonian) == pytest.approx(2.0)


def test_expectation_value_rejects_non_hermitian_operator():
    with pytest.raises(ValueError, match="Hermitian"):
        expectation_value((1.0, 0.0), ((1.0, 1.0), (0.0, 1.0)))


def test_objective_is_scalar_real_and_auditable():
    encoder = lambda x: (x[0] + x[1],)
    decoder = lambda z: (z[0] / 2.0, z[0] / 2.0)
    config = MoagiObjectiveConfig(lambda_refinement=0.2, eta_quantum=0.1)

    result = evaluate_autoencoding_objective(
        (1.0, 3.0),
        encoder,
        decoder,
        probabilities=(0.25, 0.75),
        encoder_jacobian=((1.0, 1.0),),
        wavefunction=(1.0 + 0.0j, 0.0 + 1.0j),
        hamiltonian=((1.0, 0.0), (0.0, 3.0)),
        config=config,
    )

    assert result.latent == pytest.approx((4.0,))
    assert result.reconstruction == pytest.approx((2.0, 2.0))
    assert result.reconstruction_loss == pytest.approx(1.0)
    assert isinstance(result.total, float)
    assert math.isfinite(result.total)
    assert result.total == pytest.approx(
        result.reconstruction_loss
        + config.lambda_refinement * result.refinement
        + config.eta_quantum * result.quantum_expectation
    )


def test_nonzero_quantum_weight_requires_explicit_state_and_operator():
    with pytest.raises(ValueError, match="eta_quantum"):
        evaluate_autoencoding_objective(
            (1.0,),
            lambda x: x,
            lambda z: z,
            config=MoagiObjectiveConfig(eta_quantum=1.0),
        )


def test_gradient_step_updates_parameters_not_input_state():
    updated = gradient_step((1.0, -2.0), (0.5, -1.0), 0.1)
    assert updated == pytest.approx((0.95, -1.9))
