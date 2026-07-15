import math

import pytest

from jarvisx.d_muad import (
    AdamScalarState,
    DMUADConfig,
    LossWeights,
    adam_scalar_step,
    aggregate_loss,
    constitutional_boundaries,
    conv3d_macs,
    derive_contract,
    encoder_macs,
    padded_extent,
    separable_embedding_macs,
)


def test_padding_is_minimal_and_divisible_by_eight():
    assert padded_extent(1) == 8
    assert padded_extent(8) == 8
    assert padded_extent(9) == 16
    assert padded_extent(33) == 40


def test_contract_derives_canonical_shapes():
    contract = derive_contract((2, 4, 9, 17, 25))

    assert contract.padded == (2, 4, 16, 24, 32)
    assert contract.padding == (7, 7, 7)
    assert contract.embedded == (2, 64, 16, 24, 32)
    assert contract.h1 == (2, 256, 8, 12, 16)
    assert contract.h2 == (2, 512, 4, 6, 8)
    assert contract.h3 == (2, 1024, 4, 6, 8)
    assert contract.latent == (2, 1024)
    assert contract.volumetric_output == (2, 5, 16, 24, 32)


def test_contract_explicitly_rejects_global_invertibility_claim():
    contract = derive_contract((1, 4, 8, 8, 8))

    assert contract.compressive
    assert not contract.exact_global_inverse_possible
    assert contract.raw_elements_per_sample == 2048
    assert contract.latent_elements_per_sample == 1024


def test_dense_bottleneck_cardinality_is_resolution_dependent():
    contract = derive_contract((1, 4, 32, 32, 32))

    expected_base_values = 1024 * 8 * 8 * 8
    assert contract.dense_bottleneck_weights == 1024 * expected_base_values
    assert contract.dense_bottleneck_weights == 536_870_912


def test_wrong_channel_count_is_rejected():
    with pytest.raises(ValueError, match="requires 4 input channels"):
        derive_contract((1, 3, 8, 8, 8))


def test_arithmetic_budget_functions_are_positive_and_shape_dependent():
    small = derive_contract((1, 4, 8, 8, 8))
    large = derive_contract((1, 4, 16, 16, 16))

    assert separable_embedding_macs(small) > 0
    assert encoder_macs(small) > 0
    assert separable_embedding_macs(large) > separable_embedding_macs(small)
    assert encoder_macs(large) > encoder_macs(small)


def test_conv3d_mac_formula():
    output = (2, 16, 4, 5, 6)
    assert conv3d_macs(output, 8, (3, 3, 3)) == 2 * 16 * 4 * 5 * 6 * 8 * 27


def test_corrected_loss_includes_rendering_and_eikonal_terms():
    loss = aggregate_loss(
        sdf=2.0,
        appearance=4.0,
        physics=5.0,
        eikonal=3.0,
        rendering=6.0,
    )

    assert loss == pytest.approx(2.0 + 0.5 * 4.0 + 0.1 * 5.0 + 0.1 * 3.0 + 6.0)


def test_loss_rejects_non_finite_or_negative_values():
    with pytest.raises(ValueError):
        aggregate_loss(math.inf, 0.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        aggregate_loss(-1.0, 0.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        LossWeights(rendering=-1.0)


def test_adam_first_step_matches_bias_corrected_recurrence():
    state = AdamScalarState(parameter=1.0)
    updated = adam_scalar_step(state, gradient=2.0, learning_rate=1e-4)

    expected = 1.0 - 1e-4 * 2.0 / (math.sqrt(4.0) + 1e-8)
    assert updated.parameter == pytest.approx(expected)
    assert updated.first_moment == pytest.approx(0.2)
    assert updated.second_moment == pytest.approx(0.004)
    assert updated.step == 1


def test_constitutional_boundaries_are_explicit():
    boundaries = constitutional_boundaries()

    assert "not a global exact inverse" in boundaries["inverse"]
    assert "almost everywhere" in boundaries["differentiability"]
    assert "shape" in boundaries["arithmetic_budget"]


def test_configuration_rejects_non_positive_dimensions():
    with pytest.raises(ValueError):
        DMUADConfig(latent_dim=0)
