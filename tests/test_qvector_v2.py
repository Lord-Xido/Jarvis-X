import pytest

from jarvisx.qvector3d import Q_ONE, Q32_MAX, QVector3Q16, QVectorField3D, q16_from_float
from jarvisx.qvector_v2 import (
    PackedQVectorField3D,
    QAccumulator64,
    QArithmeticStatus,
    QBoundaryMode,
    QNumericPolicy,
    QRoundMode,
    QScalarKernel3D,
    QVectorFieldOps3D,
    requantize_product,
)


def field_from_function(shape, function):
    sx, sy, sz = shape
    vectors = []
    for z in range(sz):
        for y in range(sy):
            for x in range(sx):
                vectors.append(function(x, y, z))
    return QVectorField3D.from_vectors(vectors, shape)


def test_nearest_even_requantization_is_bit_exact() -> None:
    status = QArithmeticStatus()
    policy = QNumericPolicy(rounding=QRoundMode.NEAREST_EVEN)
    assert requantize_product(Q_ONE + Q_ONE // 2, policy, status) == 2
    status.clear()
    assert requantize_product(2 * Q_ONE + Q_ONE // 2, policy, status) == 2
    assert status.inexact


def test_accumulator64_saturates_and_sets_sticky_status() -> None:
    status = QArithmeticStatus()
    policy = QNumericPolicy(accumulator_saturate=True)
    accumulator = QAccumulator64((1 << 63) - 2)
    accumulator.add(100, policy, status)
    assert accumulator.value == (1 << 63) - 1
    assert status.accumulator_saturated


def test_packed_vector_field_preserves_exact_binary_contract() -> None:
    source = QVectorField3D.from_vectors(
        [(1.5, -2.25, 3.125), (4.0, 5.0, -6.0)],
        (2, 1, 1),
    )
    packed = PackedQVectorField3D.from_field(source)
    assert packed.raw_bytes == 24
    assert packed.to_field() == source
    assert packed.digest == source.digest
    packed.set(1, 0, 0, QVector3Q16.from_floats(7.0, 8.0, 9.0))
    assert packed.at(1, 0, 0).to_floats() == (7.0, 8.0, 9.0)


def test_packed_field_tiles_bound_resident_working_set() -> None:
    packed = PackedQVectorField3D((5, 4, 3))
    tiles = list(packed.iter_tiles((2, 2, 2)))
    assert len(tiles) == 3 * 2 * 2
    assert tiles[0] == ((0, 0, 0), (2, 2, 2))
    assert tiles[-1] == ((4, 2, 2), (5, 4, 3))


def test_directional_derivative_of_linear_vector_field() -> None:
    field = field_from_function((3, 1, 1), lambda x, y, z: (x, 2 * x, -3 * x))
    derivative = QVectorFieldOps3D().directional_derivative(field, 0)
    assert derivative.at(1, 0, 0).to_floats() == (1.0, 2.0, -3.0)


def test_divergence_of_identity_position_field_is_three_at_interior() -> None:
    field = field_from_function((3, 3, 3), lambda x, y, z: (x, y, z))
    divergence = QVectorFieldOps3D().divergence(field)
    assert divergence.at(1, 1, 1).to_floats() == (3.0, 3.0, 3.0)


def test_curl_of_planar_rotation_field_is_positive_z_two() -> None:
    field = field_from_function((3, 3, 1), lambda x, y, z: (-y, x, 0))
    curl = QVectorFieldOps3D().curl(field)
    assert curl.at(1, 1, 0).to_floats() == (0.0, 0.0, 2.0)


def test_laplacian_of_constant_field_is_zero() -> None:
    field = QVectorField3D.from_vectors([(2.5, -4.0, 7.25)] * 27, (3, 3, 3))
    laplacian = QVectorFieldOps3D(boundary=QBoundaryMode.CLAMP).laplacian(field)
    assert all(vector == QVector3Q16.zero() for vector in laplacian.vectors)


def test_identity_convolution_is_exact() -> None:
    field = field_from_function((3, 2, 2), lambda x, y, z: (x + y, y + z, z - x))
    ops = QVectorFieldOps3D()
    convolved = ops.convolve(field, QScalarKernel3D.identity())
    assert convolved == field
    assert not ops.status.saturated
    assert not ops.status.accumulator_saturated


def test_convolution_uses_checked_wide_accumulator() -> None:
    field = QVectorField3D.from_raw([(Q32_MAX, Q32_MAX, Q32_MAX)] * 27, (3, 3, 3))
    kernel = QScalarKernel3D(tuple([Q32_MAX] * 27), (3, 3, 3))
    ops = QVectorFieldOps3D(policy=QNumericPolicy(accumulator_saturate=True))
    result = ops.convolve(field, kernel)
    assert result.at(1, 1, 1).x == Q32_MAX
    assert ops.status.accumulator_saturated
    assert ops.status.saturated


def test_non_saturating_policy_fails_closed_on_q32_overflow() -> None:
    field = QVectorField3D.from_vectors([(32767.0, 0.0, 0.0)], (1, 1, 1))
    kernel = QScalarKernel3D((q16_from_float(2.0),), (1, 1, 1))
    ops = QVectorFieldOps3D(policy=QNumericPolicy(saturate=False))
    with pytest.raises(OverflowError):
        ops.convolve(field, kernel)
