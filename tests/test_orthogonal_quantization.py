import math

import pytest

from jarvisx.orthogonal_quantization import (
    dct2_orthonormal_basis,
    orthogonal_quantization_trace,
    uniform_error_bound,
)


def test_canonical_two_point_dct_trace() -> None:
    basis = dct2_orthonormal_basis(2)
    trace = orthogonal_quantization_trace((1.5, 1.9), basis, 0.1)

    assert trace.quantized_values == (24, -3)
    assert trace.transformed_values == pytest.approx(
        (2.4041630560342613, -0.2828427124746189)
    )
    assert trace.dequantized_values == pytest.approx((2.4, -0.3))
    assert trace.reconstructed_values == pytest.approx(
        (1.4849242404917498, 1.9091883092036783)
    )
    assert trace.spatial_residual == pytest.approx(
        (0.0150757595082502, -0.0091883092036784)
    )
    assert trace.residual_norm == pytest.approx(0.0176551277, rel=1e-8)
    assert trace.deterministic_bound == pytest.approx(0.1 * math.sqrt(2.0) / 2.0)
    assert trace.gate_ratio < 0.25
    assert trace.committed is True


def test_incorrectly_normalised_dct_fails_closed() -> None:
    bad_basis = (
        (1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)),
        (0.5, -0.5),
    )

    with pytest.raises(ValueError, match="not orthonormal"):
        orthogonal_quantization_trace((1.5, 1.9), bad_basis, 0.1)


def test_nonuniform_steps_use_root_sum_square_bound() -> None:
    basis = dct2_orthonormal_basis(2)
    trace = orthogonal_quantization_trace((1.5, 1.9), basis, (0.05, 0.2))

    expected = 0.5 * math.sqrt(0.05**2 + 0.2**2)
    assert trace.deterministic_bound == pytest.approx(expected)
    assert trace.residual_norm <= trace.deterministic_bound


def test_half_step_ties_are_rounded_away_from_zero() -> None:
    identity = ((1.0, 0.0), (0.0, 1.0))
    trace = orthogonal_quantization_trace((0.05, -0.05), identity, 0.1)

    assert trace.quantized_values == (1, -1)
    assert trace.reconstructed_values == pytest.approx((0.1, -0.1))
    assert trace.residual_norm == pytest.approx(math.sqrt(0.05**2 + 0.05**2))
    assert trace.residual_norm == pytest.approx(trace.deterministic_bound)


def test_uniform_bound_validation() -> None:
    assert uniform_error_bound(0.1, 2) == pytest.approx(0.07071067811865477)

    with pytest.raises(ValueError):
        uniform_error_bound(0.0, 2)
    with pytest.raises(ValueError):
        uniform_error_bound(0.1, 0)


def test_invalid_shapes_and_nonfinite_values_are_rejected() -> None:
    basis = dct2_orthonormal_basis(2)

    with pytest.raises(ValueError, match="square"):
        orthogonal_quantization_trace((1.0, 2.0), ((1.0,),), 0.1)
    with pytest.raises(ValueError, match="finite"):
        orthogonal_quantization_trace((1.0, math.inf), basis, 0.1)
    with pytest.raises(ValueError, match="positive"):
        orthogonal_quantization_trace((1.0, 2.0), basis, -0.1)
    with pytest.raises(ValueError, match="one step per coefficient"):
        orthogonal_quantization_trace((1.0, 2.0), basis, (0.1,))


def test_general_orthogonal_basis_preserves_quantization_norm() -> None:
    # Normalized 4x4 Hadamard matrix.
    scale = 0.5
    basis = (
        (scale, scale, scale, scale),
        (scale, -scale, scale, -scale),
        (scale, scale, -scale, -scale),
        (scale, -scale, -scale, scale),
    )
    trace = orthogonal_quantization_trace((0.3, -0.7, 1.2, 0.4), basis, 0.1)

    transform_norm = math.sqrt(sum(value * value for value in trace.transform_residual))
    assert trace.residual_norm == pytest.approx(transform_norm, abs=1e-12)
    assert trace.residual_norm <= trace.deterministic_bound
