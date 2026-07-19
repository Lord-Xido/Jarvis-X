import asyncio

import numpy as np
import pytest

from jarvisx.mm3d_omega4 import (
    CloudNode,
    MM3DConfig,
    MM3DEngine,
    OmegaEntry,
    OmegaMemory,
    Voxel,
    Z8QCA,
)


def tiny_config(**overrides):
    values = dict(
        xi_size=8,
        manifold_dim=256,
        latent_size=4,
        codebook_size=128,
        codebook_dim=8,
        projection_rank=8,
        metric_rank=4,
        exploration_depth=2,
        render_image_size=8,
        render_video_frames=2,
        render_audio_samples=128,
        omega_capacity=8,
        max_reference_bytes=32 * 1024 * 1024,
        seed=12345,
    )
    values.update(overrides)
    return MM3DConfig(**values)


def test_voxel_is_exactly_384_bits_and_round_trips():
    voxel = Voxel(
        token_embedding=np.arange(16, dtype=np.int8),
        visual_feature=np.arange(12, dtype=np.int8),
        audio_spectral=np.arange(8, dtype=np.int8),
        motion_vector=np.arange(6, dtype=np.int8),
        attention_weight=np.float32(0.25),
        ethical_flag=np.uint8(1),
    )
    payload = voxel.to_bytes()
    restored = Voxel.from_bytes(payload)

    assert len(payload) == 48
    assert np.array_equal(restored.token_embedding, voxel.token_embedding)
    assert restored.attention_weight == voxel.attention_weight
    assert restored.ethical_flag == 1


def test_qca_preserves_mod8_uint8_state_deterministically():
    left = Z8QCA(8, seed=7)
    right = Z8QCA(8, seed=7)

    left.update()
    right.update()

    assert left.state.dtype == np.uint8
    assert np.array_equal(left.state, right.state)
    assert int(left.state.min()) >= 0
    assert int(left.state.max()) < 8


def test_config_separates_conceptual_target_from_operational_allocation():
    config = tiny_config()
    config.validate()

    assert config.conceptual_parameters_total == 50_000_000_000_000
    assert config.conceptual_parameters_active == 250_000_000_000
    assert config.estimated_reference_bytes() < config.max_reference_bytes

    impossible = MM3DConfig(
        xi_size=256,
        manifold_dim=65536,
        latent_size=32,
        codebook_size=262144,
        codebook_dim=256,
        projection_rank=1024,
        metric_rank=1024,
        max_reference_bytes=512 * 1024 * 1024,
    )
    with pytest.raises(ValueError, match="allocation guard"):
        impossible.validate()


def test_cycle_is_deterministic_and_ledger_verifies():
    config = tiny_config()
    psi = np.linspace(-1.0, 1.0, config.manifold_dim, dtype=np.float32)
    left = MM3DEngine(config)
    right = MM3DEngine(config)

    left_result = left.cycle(psi)
    right_result = right.cycle(psi)

    assert np.array_equal(left_result.latent_code, right_result.latent_code)
    assert np.array_equal(left_result.safe_state, right_result.safe_state)
    assert left_result.xi_state_hash == right_result.xi_state_hash
    assert left.omega.verify()
    assert right.omega.verify()
    assert left_result.actual_parameter_count < left_result.conceptual_parameter_count
    assert isinstance(left_result.summary()["target_met"], bool)


def test_empty_input_is_rejected_by_lambda_policy():
    engine = MM3DEngine(tiny_config())
    with pytest.raises(RuntimeError, match="Lambda constraint"):
        engine.cycle(np.array([], dtype=np.float32))


def test_distributed_projection_matches_sequential_state():
    config = tiny_config()
    psi = np.linspace(-0.5, 0.5, config.manifold_dim, dtype=np.float32)
    sequential = MM3DEngine(config)
    distributed = MM3DEngine(config)
    distributed.add_cloud_node(CloudNode(0))
    distributed.add_cloud_node(CloudNode(1))

    expected = sequential.cycle(psi)
    actual = asyncio.run(distributed.distributed_cycle(psi))
    distributed.close()

    assert np.array_equal(actual.latent_code, expected.latent_code)
    assert np.allclose(actual.safe_state, expected.safe_state, atol=1e-6)
    assert distributed.omega.verify()


def test_omega_retained_window_remains_verifiable():
    memory = OmegaMemory(capacity=2)
    for sequence in range(4):
        memory.append(
            OmegaEntry(
                sequence=sequence,
                instruction_hash=f"instruction-{sequence}",
                voxel_region=(0, 0, 0, 1, 1, 1),
                ethical_flag=0,
                result_hash=f"result-{sequence}",
            )
        )

    assert len(memory.chain) == 2
    assert memory.anchor_hash != OmegaMemory.GENESIS
    assert memory.verify()
