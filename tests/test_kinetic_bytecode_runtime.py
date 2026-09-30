import pytest

from jarvisx.kinetic_bytecode_runtime import (
    KineticBytecodeRuntime,
    KineticConfig,
    KineticInstruction,
    KineticOp,
    PipelineStage,
    RegionDescriptor,
    canonical_million_power_space,
)


def test_symbolic_extent_matches_requested_scale_without_materialization() -> None:
    space = canonical_million_power_space()
    assert space.base == 1_000_000
    assert space.exponent == 1_000_000
    assert space.log10_axis == pytest.approx(6_000_000.0)
    assert space.log10_volume == pytest.approx(18_000_000.0)
    assert 19_931_568 <= space.approximate_axis_bits <= 19_931_570


def test_g3d_packet_traverses_pipeline_and_commits() -> None:
    runtime = KineticBytecodeRuntime(
        space=canonical_million_power_space(),
        regions=[RegionDescriptor(7)],
        program=[
            KineticInstruction(
                KineticOp.G3D,
                7,
                observation=2.0,
                prediction=2.5,
                omega=0.25,
                immediate=0.5,
                verification_score=0.99,
            )
        ],
    )
    final = runtime.run()
    assert final.commits == 1
    assert final.rollbacks == 0
    assert final.inflight == 0
    assert final.resident_regions == ()
    assert runtime.committed[7] == pytest.approx(5.75)
    stages = [stage for _, packet_id, stage in runtime.trace if packet_id == 0]
    assert stages == list(PipelineStage)[:12]


def test_verification_failure_rolls_back_without_commit() -> None:
    runtime = KineticBytecodeRuntime(
        space=canonical_million_power_space(),
        regions=[RegionDescriptor(9)],
        program=[
            KineticInstruction(
                KineticOp.DELTA,
                9,
                observation=1.0,
                immediate=4.0,
                verification_score=0.1,
            )
        ],
    )
    final = runtime.run()
    assert final.commits == 0
    assert final.rollbacks == 1
    assert 9 not in runtime.committed
    assert final.resident_regions == ()


def test_projection_applies_before_verification_and_commit() -> None:
    runtime = KineticBytecodeRuntime(
        space=canonical_million_power_space(),
        regions=[RegionDescriptor(1)],
        program=[
            KineticInstruction(
                KineticOp.DELTA,
                1,
                observation=0.0,
                immediate=100.0,
                verification_score=1.0,
            )
        ],
        config=KineticConfig(projection_limit=3.0),
    )
    runtime.run()
    assert runtime.committed[1] == 3.0


def test_residency_bound_stalls_but_eventually_drains() -> None:
    runtime = KineticBytecodeRuntime(
        space=canonical_million_power_space(),
        regions=[RegionDescriptor(1), RegionDescriptor(2)],
        program=[
            KineticInstruction(KineticOp.DELTA, 1, immediate=1.0),
            KineticInstruction(KineticOp.DELTA, 2, immediate=2.0),
        ],
        config=KineticConfig(max_resident_regions=1),
    )
    final = runtime.run()
    assert final.commits == 2
    assert final.stalls > 0
    assert runtime.committed == {1: 1.0, 2: 2.0}


def test_external_validator_can_reject_candidate() -> None:
    runtime = KineticBytecodeRuntime(
        space=canonical_million_power_space(),
        regions=[RegionDescriptor(3)],
        program=[KineticInstruction(KineticOp.DELTA, 3, immediate=1.0)],
        validator=lambda packet, instruction: False,
    )
    final = runtime.run()
    assert final.rollbacks == 1
    assert final.commits == 0
