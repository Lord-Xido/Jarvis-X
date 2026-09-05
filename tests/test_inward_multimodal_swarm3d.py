import math

import pytest

from jarvisx.inward_multimodal_swarm3d import (
    ElectricalAnalogueConfig,
    InwardMultimodalSwarm3D,
    Modality,
    Particle3D,
    Swarm3DConfig,
    electrical_step,
    local_riemannian_gradient,
    riemannian_metric_inverse,
)


class IdentityCodec:
    def encode(self, value: object):
        values = tuple(float(x) for x in value)  # type: ignore[arg-type]
        return values[0], values[1], values[2]

    def decode(self, position):
        return position


class ContractiveCodec:
    def encode(self, value: object):
        values = tuple(float(x) for x in value)  # type: ignore[arg-type]
        return 0.5 * values[0], 0.5 * values[1], 0.5 * values[2]

    def decode(self, position):
        return position


def test_rank_one_metric_inverse_and_riemannian_gradient_are_exact():
    inverse = riemannian_metric_inverse((2.0, 0.0, 0.0), alpha=1.0)
    expected = (
        (0.2, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    for row, expected_row in zip(inverse, expected):
        assert row == pytest.approx(expected_row)

    gradient = local_riemannian_gradient(
        (1.0, 1.0, 0.0),
        (2.0, 0.0, 0.0),
        alpha=1.0,
    )
    assert gradient == pytest.approx((0.2, 1.0, 0.0))


def test_parallel_swarm_coupling_reduces_consensus_error():
    runtime = InwardMultimodalSwarm3D(
        {Modality.TEXT: IdentityCodec()},
        Swarm3DConfig(
            dt=0.1,
            inward_gain=0.0,
            swarm_gain=0.5,
            feature_mix_gain=0.0,
            max_steps=10,
        ),
    )
    particles = (
        Particle3D((-1.0, 0.0, 0.0), (1.0, 0.0), Modality.TEXT, source_id="a"),
        Particle3D((1.0, 0.0, 0.0), (1.0, 0.0), Modality.TEXT, source_id="b"),
    )

    state = runtime.initial_state(particles)
    candidate = runtime.step(state)

    assert candidate.metrics.consensus_error < state.metrics.consensus_error
    assert candidate.particles[0].position[0] > -1.0
    assert candidate.particles[1].position[0] < 1.0


def test_inward_decode_reencode_loop_converges_for_contractive_codec():
    runtime = InwardMultimodalSwarm3D(
        {Modality.TEXT: ContractiveCodec()},
        Swarm3DConfig(
            dt=0.2,
            inward_gain=0.8,
            swarm_gain=0.0,
            feature_mix_gain=0.0,
            max_steps=200,
            tolerance=1.0e-4,
        ),
    )
    particle = Particle3D((0.5, 0.0, 0.0), (1.0,), Modality.TEXT)

    result = runtime.relax((particle,))

    assert result.converged
    assert result.state.metrics.fixed_point_error <= runtime.config.tolerance
    assert abs(result.state.particles[0].position[0]) < 3.0e-4


def test_multimodal_encoding_uses_one_shared_feature_width():
    runtime = InwardMultimodalSwarm3D(
        {
            Modality.TEXT: IdentityCodec(),
            Modality.IMAGE: IdentityCodec(),
        },
        feature_encoder=lambda modality, value, position: (
            position[0],
            position[1],
            position[2],
            1.0 if modality is Modality.TEXT else -1.0,
        ),
    )

    particles = runtime.encode_modalities(
        {
            Modality.TEXT: ((0.1, 0.2, 0.3),),
            Modality.IMAGE: ((-0.1, 0.2, 0.3),),
        }
    )

    assert len(particles) == 2
    assert all(len(particle.feature) == 4 for particle in particles)
    assert {particle.modality for particle in particles} == {Modality.TEXT, Modality.IMAGE}


def test_consensus_can_decode_through_multiple_modality_heads():
    runtime = InwardMultimodalSwarm3D(
        {
            Modality.TEXT: IdentityCodec(),
            Modality.IMAGE: IdentityCodec(),
        }
    )
    particles = (
        Particle3D((0.2, 0.4, 0.6), (1.0,), Modality.TEXT),
    )

    generated = runtime.decode_consensus(particles)

    assert generated[Modality.TEXT] == pytest.approx((0.2, 0.4, 0.6))
    assert generated[Modality.IMAGE] == pytest.approx((0.2, 0.4, 0.6))


def test_rc_analogue_preserves_pair_mean_and_relaxes_voltage_difference():
    config = ElectricalAnalogueConfig(
        capacitance=1.0,
        feedback_conductance=0.0,
        coupling_conductance=0.5,
        dt=0.1,
    )
    voltages = ((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    adjacency = ((0.0, 1.0), (1.0, 0.0))

    candidate = electrical_step(voltages, voltages, adjacency, config)

    assert candidate[0][0] > -1.0
    assert candidate[1][0] < 1.0
    assert math.isclose(candidate[0][0] + candidate[1][0], 0.0, abs_tol=1.0e-12)
    assert abs(candidate[1][0] - candidate[0][0]) < 2.0


def test_runtime_rejects_unbounded_or_incompatible_inputs():
    with pytest.raises(ValueError, match="positive integer"):
        Swarm3DConfig(max_steps=0)

    runtime = InwardMultimodalSwarm3D({Modality.TEXT: IdentityCodec()})
    with pytest.raises(KeyError, match="no codec"):
        runtime.encode_modalities({Modality.AUDIO: ((0.0, 0.0, 0.0),)})

    with pytest.raises(ValueError, match="shared feature width"):
        runtime.initial_state(
            (
                Particle3D((0.0, 0.0, 0.0), (1.0,), Modality.TEXT),
                Particle3D((0.1, 0.0, 0.0), (1.0, 2.0), Modality.TEXT),
            )
        )
