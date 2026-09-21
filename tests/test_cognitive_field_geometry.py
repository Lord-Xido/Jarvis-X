from __future__ import annotations

import math

import pytest

from jarvisx.cognitive_field_geometry import (
    CognitiveFieldConfig,
    CognitiveFieldVerifier,
    CognitiveGeometryError,
    InformationalStress,
    LatentGeometry,
    computational_field_residual,
    holonomy_residual,
)


IDENTITY = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)


def test_identity_transport_has_zero_holonomy() -> None:
    assert holonomy_residual((1.0, -2.0, 3.0), [IDENTITY, IDENTITY]) == 0.0


def test_nonclosing_transport_has_positive_holonomy() -> None:
    stretch_x = (
        (2.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )

    residual = holonomy_residual((1.0, 0.0, 0.0), [stretch_x])

    assert residual == pytest.approx(1.0)


def test_balanced_computational_field_has_zero_residual() -> None:
    config = CognitiveFieldConfig(lambda_z=0.25, kappa_z=2.0)
    stress = InformationalStress(
        input_density=2.0,
        residual_pressure=3.0,
        memory_pressure=4.0,
        interaction_flux=(0.5, 0.25, 0.125),
    )
    metric = IDENTITY
    source = stress.tensor
    curvature = tuple(
        tuple(
            config.kappa_z * source[row][col] - config.lambda_z * metric[row][col]
            for col in range(3)
        )
        for row in range(3)
    )
    geometry = LatentGeometry(metric=metric, curvature_proxy=curvature)

    residual = computational_field_residual(geometry, stress, config)

    assert all(
        abs(residual[row][col]) < 1e-12
        for row in range(3)
        for col in range(3)
    )


def test_verifier_accepts_consistent_geometry_and_identity_loop() -> None:
    verifier = CognitiveFieldVerifier(
        CognitiveFieldConfig(
            max_geometry_residual=1e-9,
            max_holonomy_residual=1e-9,
            max_reconstruction_error=0.01,
            max_fixed_point_delta=0.01,
        )
    )
    stress = InformationalStress(1.0, 2.0, 3.0)
    geometry = LatentGeometry(
        metric=IDENTITY,
        curvature_proxy=stress.tensor,
    )

    receipt = verifier.verify(
        geometry=geometry,
        stress=stress,
        latent=(0.2, -0.4, 0.8),
        transport_loop=[IDENTITY],
        reconstruction_error=0.001,
        fixed_point_delta=0.002,
    )

    assert receipt.accepted is True
    assert receipt.geometry_residual == pytest.approx(0.0)
    assert receipt.holonomy_residual == pytest.approx(0.0)
    assert receipt.claim_status == "computational_geometry_only"
    assert receipt.action_proxy == pytest.approx(math.sqrt(0.001**2 + 0.002**2))


def test_verifier_rejects_transport_inconsistency() -> None:
    verifier = CognitiveFieldVerifier(
        CognitiveFieldConfig(
            max_geometry_residual=1e-9,
            max_holonomy_residual=0.1,
            max_reconstruction_error=1.0,
            max_fixed_point_delta=1.0,
        )
    )
    stress = InformationalStress(1.0, 1.0, 1.0)
    geometry = LatentGeometry(metric=IDENTITY, curvature_proxy=IDENTITY)
    stretch_x = (
        (1.5, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )

    receipt = verifier.verify(
        geometry=geometry,
        stress=stress,
        latent=(1.0, 0.0, 0.0),
        transport_loop=[stretch_x],
        reconstruction_error=0.0,
        fixed_point_delta=0.0,
    )

    assert receipt.accepted is False
    assert receipt.holonomy_residual == pytest.approx(0.5)


def test_metric_must_be_symmetric_positive_definite() -> None:
    with pytest.raises(CognitiveGeometryError, match="symmetric"):
        LatentGeometry(
            metric=((1.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            curvature_proxy=IDENTITY,
        )

    with pytest.raises(CognitiveGeometryError, match="positive definite"):
        LatentGeometry(
            metric=((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
            curvature_proxy=IDENTITY,
        )


def test_negative_verification_errors_fail_closed() -> None:
    verifier = CognitiveFieldVerifier()
    geometry = LatentGeometry(metric=IDENTITY, curvature_proxy=IDENTITY)
    stress = InformationalStress(1.0, 1.0, 1.0)

    with pytest.raises(CognitiveGeometryError, match="non-negative"):
        verifier.verify(
            geometry=geometry,
            stress=stress,
            latent=(0.0, 0.0, 0.0),
            transport_loop=[],
            reconstruction_error=-1.0,
            fixed_point_delta=0.0,
        )
