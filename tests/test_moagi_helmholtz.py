from __future__ import annotations

import pytest

from jarvisx.moagi_helmholtz import (
    IdentityGeometryRefiner,
    Mesh,
    MoagiHelmholtzConfig,
    MoagiHelmholtzEngine,
    ReferenceArchiveCodec,
    ReferenceConditioner,
    ReferenceGeometryCodec,
    ReferenceRenderer,
    SideInfoInverseModel,
    mesh_rms,
)


def triangle() -> Mesh:
    return Mesh(
        vertices=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        faces=((0, 1, 2),),
    )


def engine(**config_overrides: object) -> MoagiHelmholtzEngine:
    config = MoagiHelmholtzConfig(**config_overrides)
    codec = ReferenceGeometryCodec()
    return MoagiHelmholtzEngine(
        conditioner=ReferenceConditioner(),
        encoder=codec,
        decoder=codec,
        refiner=IdentityGeometryRefiner(),
        renderer=ReferenceRenderer(),
        archive_codec=ReferenceArchiveCodec(),
        inverse_model=SideInfoInverseModel(),
        config=config,
    )


def test_reference_cycle_round_trips_geometry() -> None:
    source = triangle()
    runtime = engine()

    state = runtime.step({"text": "triangle", "audio": [0.0, 1.0]}, source)

    assert state.refined_mesh == source
    assert state.reconstructed_mesh == source
    assert state.metrics.generation_rms == 0.0
    assert state.metrics.cycle_rms == 0.0
    assert state.metrics.anchor_rms == 0.0
    assert state.metrics.archive_bytes > 0
    assert runtime.anchor == source
    assert runtime.state == state


def test_anchor_is_immutable_across_cycles() -> None:
    first = triangle()
    second = Mesh(
        vertices=((0.1, 0.0, 0.0), (1.1, 0.0, 0.0), (0.1, 1.0, 0.0)),
        faces=first.faces,
    )
    runtime = engine()

    runtime.step({"text": "first"}, first)
    state = runtime.step({"text": "second"}, second)

    assert runtime.anchor == first
    assert state.metrics.anchor_rms == pytest.approx(0.1 / (3**0.5))


def test_validator_rejection_is_transactional() -> None:
    runtime = engine()
    accepted = runtime.step({"text": "accepted"}, triangle())

    with pytest.raises(RuntimeError, match="validator"):
        runtime.step(
            {"text": "rejected"},
            triangle(),
            validator=lambda candidate: candidate.cycle < 2,
        )

    assert runtime.state == accepted
    assert runtime.anchor == triangle()


def test_archive_budget_rejects_before_commit() -> None:
    runtime = engine(max_archive_bytes=1)

    with pytest.raises(RuntimeError, match="archive exceeds"):
        runtime.step({"text": "too large"}, triangle())

    assert runtime.state is None
    assert runtime.anchor is None


def test_vertex_budget_is_enforced() -> None:
    runtime = engine(max_vertices=2)

    with pytest.raises(RuntimeError, match="vertex budget"):
        runtime.step({}, triangle())


def test_mesh_validation_rejects_bad_indices() -> None:
    with pytest.raises(ValueError, match="outside"):
        Mesh(vertices=((0.0, 0.0, 0.0),), faces=((0, 1, 2),))


def test_mesh_rms_requires_matching_topology() -> None:
    source = triangle()
    other = Mesh(
        vertices=source.vertices,
        faces=((0, 2, 1),),
    )
    assert mesh_rms(source, other) == float("inf")


def test_cycle_tolerance_rejects_nonmatching_inverse() -> None:
    class ShiftedDecoder(ReferenceGeometryCodec):
        def decode(self, latent: object, condition: object) -> Mesh:
            mesh = super().decode(latent, condition)
            if condition == {"phase": "inverse"}:
                return Mesh(
                    vertices=tuple((x + 1.0, y, z) for x, y, z in mesh.vertices),
                    faces=mesh.faces,
                )
            return mesh

    class ShiftInverse(SideInfoInverseModel):
        def infer(self, frames: object, side_info: object) -> tuple[object, object]:
            latent, _condition = super().infer(frames, side_info)  # type: ignore[arg-type]
            return latent, {"phase": "inverse"}

    codec = ShiftedDecoder()
    runtime = MoagiHelmholtzEngine(
        conditioner=ReferenceConditioner(),
        encoder=codec,
        decoder=codec,
        refiner=IdentityGeometryRefiner(),
        renderer=ReferenceRenderer(),
        archive_codec=ReferenceArchiveCodec(),
        inverse_model=ShiftInverse(),
        config=MoagiHelmholtzConfig(max_cycle_rms=0.1),
    )

    with pytest.raises(RuntimeError, match="cycle reconstruction"):
        runtime.step({}, triangle())

    assert runtime.state is None
