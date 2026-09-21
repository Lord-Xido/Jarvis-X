from __future__ import annotations

import math
import pytest

from jarvisx.dr_moagi_code_rate_continuum import (
    CodeRateContinuum,
    ContinuumConfig,
    inward_kernel,
    norm3,
    rotate_contract,
    spherical_shell_volume,
)


def test_rotation_contraction_norm() -> None:
    vector = (3.0, -4.0, 2.0)
    gamma = 0.37
    nxt = rotate_contract(vector, gamma, math.pi / 5.0)
    assert norm3(nxt) == pytest.approx(gamma * norm3(vector))


def test_kernel_shell_weight_stays_finite() -> None:
    radius = 1.0
    weights = []
    for _ in range(32):
        nxt = 0.5 * radius
        weight = inward_kernel(radius, 1.0) * spherical_shell_volume(radius, nxt)
        assert math.isfinite(weight)
        assert weight > 0.0
        weights.append(weight)
        radius = nxt
    assert weights[-1] < weights[0]


def test_finite_continuum_receipt() -> None:
    rates = {256: 800000.0, 1024: 1200000.0, 4096: 1000000.0}
    config = ContinuumConfig(
        levels=6,
        gamma=0.5,
        epsilon0=0.5,
        beta=0.0,
        candidates=(256, 1024, 4096),
    )
    receipt = CodeRateContinuum(config).run(lambda chunk: rates[chunk])
    assert receipt.accepted_levels == 6
    assert receipt.finite_lambda > 0.0
    assert receipt.empirical_lambda_infinity_upper > receipt.finite_lambda
    assert all(level.selected_chunk_lines == 1024 for level in receipt.levels)


def test_epsilon_nonincreasing() -> None:
    config = ContinuumConfig(
        levels=6,
        gamma=0.6,
        epsilon0=0.2,
        epsilon_floor=1e-5,
        lambda0=1e-7,
        beta=0.0,
        candidates=(1024,),
    )
    receipt = CodeRateContinuum(config).run(lambda _: 1000000.0)
    eps = [level.epsilon for level in receipt.levels]
    assert all(value >= config.epsilon_floor for value in eps)
    assert all(a >= b for a, b in zip(eps, eps[1:]))
